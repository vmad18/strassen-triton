"""
note, idk if this was the correct version of the code, crossing my fingers
"""

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging

import argparse
import dataclasses
import gc
import glob
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.utils.checkpoint import checkpoint

from fixed_strassen import strassen_matmul_n_layers 

import wandb

###
# NanoGPT speedrun training script
# Heavily inspired by / code borrowed from NanoGPT and https://github.com/KellerJordan/modded-nanogpt
###

# speedrun is to <= this val_loss. A val loss of <3.278 is good evidence that >95% of runs attain below 3.28
SPEEDRUN_TARGET = 3.28

# -----------------------------------------------------------------------------
# Memory Snapshotting
# https://pytorch.org/blog/understanding-gpu-memory-1/#capturing-memory-snapshots
# https://pytorch.org/memory_viz

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
PROFILING_DIR = Path('./profiling')
MEM_PROFILING_DIR = PROFILING_DIR / 'mem'
PERF_PROFILING_DIR = PROFILING_DIR / 'perf'


class LinearStrassen(nn.Linear):
    def __init__(self,
                 features: int, 
                 out_features: int,
                 bias: bool = False,
                 dtype=torch.float32,
                 depth: int = 1,
                 device: str = "cuda") -> None:
        is_power_of_two = (features > 0) and ((features & (features - 1)) == 0)
        assert is_power_of_two, "Features must be a power of 2."
        
        super().__init__(features, features, bias=bias, device=device, dtype=dtype)
        self.depth = depth
        self.to(device=device, dtype=dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_dims, S, D_in = input_tensor.shape 

        input_tensor = input_tensor.view(-1, D_in)
        
        assert D_in == self.in_features, f"Input feature dimension mismatch. Expected {self.in_features}, got {D_in}"
        assert batch_dims * S == D_in, f"Input Tensor must be square! Got {batch_dims * S} x {D_in}"

        D = self.in_features
        input_reshaped = input_tensor.reshape(D, D) 
        M = D

        W_T = self.weight.T 
        
        if M < D: 
            pad_rows_x = D - M
            X_padded = F.pad(input_reshaped, (0, 0, 0, pad_rows_x), "constant", 0)
        else: 
            X_padded = input_reshaped # X_padded is (D,D)

        out_padded = strassen_matmul_n_layers(X_padded, W_T, n_depth=self.depth)

        out_unpadded = out_padded[:M, :D]

        out = out_unpadded.view(D, self.out_features)
        if self.bias is not None:
            out = out + self.bias  
            
        return out.view(batch_dims, S, D_in)


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError()

    print0('Starting snapshot record_memory_history', console=True)
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError()

    print0('Stopping snapshot record_memory_history', console=True)
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError()

    host_name = socket.gethostname()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_prefix = f'{host_name}_{timestamp}'
    MEM_PROFILING_DIR.mkdir(exist_ok=True)
    file_path = MEM_PROFILING_DIR / f'{file_prefix}.pickle'

    try:
        print(f'Saving snapshot to local file: {file_path}')
        torch.cuda.memory._dump_snapshot(str(file_path))
    except Exception as e:
        print(f'ERROR: Failed to capture memory snapshot {e}')
        return


def report_mem_consumption() -> str:
    mem_used = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
    mem_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 * 1024 * 1024)
    return f'vram:{mem_used:.1f}/{mem_total:.1f}GB'


# -----------------------------------------------------------------------------
# Performance profiling
@contextmanager
def maybe_profile(do_profile: bool, rank: int = 0):
    if not do_profile:
        yield None
        return

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=1),
        profile_memory=False,
        with_stack=True,
    ) as prof:
        yield prof

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    PERF_PROFILING_DIR.mkdir(exist_ok=True)
    trace_file_path = PERF_PROFILING_DIR / f'trace_{timestamp}_rank{rank}.json'
    prof.export_chrome_trace(str(trace_file_path))


# -----------------------------------------------------------------------------
# Muon optimizer
# Reference: https://kellerjordan.github.io/posts/muon/
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1)) ** 0.5  # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


def count_parameters(model: nn.Module):
    embedding_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'wte' in name)
    non_embedding_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'wte' not in name)
    return embedding_params + non_embedding_params, embedding_params, non_embedding_params


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def softcap(x, cap=1):
    return cap * F.tanh(x / cap)


flex_attention = torch.compile(flex_attention, dynamic=False)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base_theta=10000):  # TODO: increase base theta to 500k (as per llama3)
        super().__init__()
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, block_mask):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # changes direction but not magnitude
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y: torch.Tensor = flex_attention(q, k, v, block_mask=block_mask)  # type: ignore # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / (2 * config.n_layer) ** 0.5

    def forward(self, x, block_mask):
        x = x + self.attn_scale * self.attn(norm(x), block_mask)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eot_token = tiktoken.get_encoding('gpt2')._special_tokens['<|endoftext|>']  # 50256

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    def get_attn_mask(self, idx):
        # idx: shape (b, t)
        # Track document boundaries by detecting EOT tokens and doing a cumulative sum
        # This gives each token a "document ID" - tokens in the same document have the same ID
        documents = (idx == self.eot_token).cumsum(dim=1)  # dim=1 for sequence dimension, not batch

        def document_mask(b, h, q_idx, kv_idx):
            # Only allow attention between tokens in the same document
            return documents[b, q_idx] == documents[b, kv_idx]

        def causal_mask(b, h, q_idx, kv_idx):
            # Standard causal mask - only attend to past tokens
            return q_idx >= kv_idx

        def sliding_window_mask(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= 1024

        # Combine document and causal masks - tokens can only attend to past tokens in the same document
        return and_masks(document_mask, causal_mask, sliding_window_mask)

    def get_block_mask(self, idx):
        # compute block mask once per batch to ammortize the cost of this computation
        B, T = idx.size()
        attn_mask = self.get_attn_mask(idx)
        return create_block_mask(attn_mask, B=B, H=None, Q_LEN=T, KV_LEN=T, _compile=True)

    def forward(self, idx, targets):
        # forward the GPT model itself
        block_mask = self.get_block_mask(idx)
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x, block_mask)
        x = norm(x)

        # compute loss
        logits = softcap(self.lm_head(x), cap=30)  # tanh logit softcap
        logits = logits.float()  # use tf32/fp32 for logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print('ERROR: magic number mismatch in the data .bin file!')
        print('---> HINT: Are you passing in a correct file with --input_bin?')
        print('---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README')
        print('---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try')
        exit(1)
    assert header[1] == 1, 'unsupported version'
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
        assert header[1] == 1, 'unsupported version'
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, 'number of tokens read does not match header?'
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f'did not find any files that match the pattern {filename_pattern}'

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


# -----------------------------------------------------------------------------
# int main


def main(hparam_overrides=None, model_overrides=None, trial: optuna.Trial | None = None):
    
    parsed_args = {}
    if trial is None:
        parser = argparse.ArgumentParser(description='Train a GPT model.')
        parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging.')
        parser.add_argument('--notes', type=str, default=None, help='Notes for the run, will be logged to wandb.')
        parsed_args = parser.parse_args()

    @dataclass
    class Hyperparameters:
        input_bin: str = 'src/data/fineweb10B/fineweb_train_*.bin'
        input_val_bin: str = 'src/data/fineweb10B/fineweb_val_*.bin'
        batch_size = 2 * 8  # global batch size, in sequences
        device_batch_size: int = 1  # batch size, in sequences, per device
        sequence_length: int = 8192 # 32_768  # sequence length, in tokens
        num_iterations: int = 3584  # number of iterations to run
        learning_rate: float = 0.000177  # muon
        emb_learning_rate: float = 0.00198  # adam
        warmup_iters: int = 0
        warmdown_iters: int = 938
        weight_decay: float = 0.0
        val_loss_every: int = 128
        val_tokens: int = 10_485_760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons (2^21 * 5)
        disable_wandb: bool = False
        memory_snapshot_steps: int = -1  # -1 to disable
        do_profile: bool = False

    @dataclass
    class GPTConfig:
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested by @Grad62304977.
        # this originates from Karpathy's experiments. this might be more relevant for H100 runs.
        vocab_size: int = 50304
        n_layer: int = 12
        n_head: int = 16
        n_embd: int = 8192

        def __post_init__(self):
            assert self.n_embd % self.n_head == 0
            self.head_dim = self.n_embd // self.n_head

    args = Hyperparameters()
    args.disable_wandb = args.disable_wandb  # or parsed_args.disable_wandb

    # Override with any provided args
    if hparam_overrides is not None:
        print(f'overriding hparams with: {hparam_overrides}')
        args = dataclasses.replace(args, **hparam_overrides)
    
    # Add FSDP import at the top of the file (not shown here, but needed):
    # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    # from torch.distributed.fsdp import MixedPrecision
    
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of FSDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), 'for now i think we need CUDA for FSDP'
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # ... (args error checking and convenience variables remain the same)
    # args error checking and convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)
    tokens_per_fwdbwd = B * T * ddp_world_size * train_accumulation_steps

    def print0(s, console=False):
        if master_process:
            with open(logfile, 'a') as f:
                if console:
                    print(s)
                print(s, file=f)

    
    # begin logging
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs('logs', exist_ok=True)
        logfile = f'logs/{run_id}.txt'
        print0(logfile, console=True)
        # initialize wandb
        if not args.disable_wandb:
            wandb.init(project='nanogpt-speedrun', name=str(run_id), config=args, notes=parsed_args.notes)

    print0(f'Running pytorch {torch.version.__version__}')
    print(f'using device: {device}')
    print0(f'Hyperparameters: {args}', console=True)
    print0(f'{tokens_per_fwdbwd=} ({B=} {T=} {ddp_world_size=} {train_accumulation_steps=} {args.batch_size=})', console=True)

    # begin by printing this file (the Python code)
    print0('=' * 100)
    print0(code)
    print0('=' * 100)
    # log information about the hardware/software environment this is running on
    print0(f'Running Python {sys.version}')
    print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
    print0(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0('=' * 100)



    # init the model from scratch
    model_config = GPTConfig()
    if model_overrides is not None:
        print0(f'overriding model config with: {model_overrides}', console=True)
        model_config = dataclasses.replace(model_config, **model_overrides)
    print0(f'Model Config: {model_config}', console=True)
    model = GPT(model_config)
    model = model.train().cuda()
    total_params, embedding_params, non_embedding_params = count_parameters(model)
    print0(f'Total params: {total_params / 1e6:.2f}M, embedding params: {embedding_params / 1e6:.2f}M, non-embedding params: {non_embedding_params / 1e6:.2f}M', console=True)
    
    # FSDP setup instead of DDP
    print0('Setting up FSDP...', console=True)
    
    # Define auto wrap policy for transformer blocks
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={Block},  # Your transformer block class
        min_num_params=1e4,  # Minimum parameters to wrap a layer
    )
    
    # Mixed precision config for FSDP
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=ddp_local_rank,
        sync_module_states=True,  # Sync parameters across ranks
        use_orig_params=True,  # Needed for optimizer state dict
    )
    
    print0('Compiling the model...', console=True)
    # torch._dynamo.config.capture_scalar_outputs = True
    fsdp_model = torch.compile(fsdp_model, dynamic=False)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    mem_used = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    print0(f'Initing model required {mem_used:.1f}GB vram', console=True)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    print0(
        f'Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files',
        console=True,
    )
    print0(
        f'Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files',
        console=True,
    )

    x, y = train_loader.next_batch()

    # init the optimizer(s) - note using fsdp_model now
    optimizer1 = torch.optim.AdamW(
        fsdp_model.parameters(),  # Changed from model.lm_head.parameters()
        lr=args.emb_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # For FSDP, we need to get the transformer parameters differently
    # This is a simplified approach - you might need to adjust based on your specific needs
    transformer_params = []
    for name, param in fsdp_model.named_parameters():
        if 'transformer.h' in name:
            transformer_params.append(param)
    
    optimizer2 = Muon(
        transformer_params,
        lr=args.learning_rate,
        momentum=0.95,
    )
    optimizers = [optimizer1, optimizer2]

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    if args.memory_snapshot_steps > 0 and master_process:
        start_record_memory_history()

    tokens_seen = 0
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    val_loss = float('inf')
    with maybe_profile(args.do_profile, rank=ddp_rank) as prof:
        for step in range(args.num_iterations + 1):
            last_step = step == args.num_iterations
            # This effectively ignores timing first 10 steps, which are slower for weird reasons.
            # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
            # steps with dummy data first, and then re-initialize the model and reset the loader.
            if step == 10:
                training_time_ms = 0
                t0 = time.perf_counter()
            timed_steps = float('nan') if step <= 11 else (step - 10) + 1  # <= 11 to avoid bug in val

            # Optionally capture memory snapshot
            if step == args.memory_snapshot_steps and master_process:
                export_memory_snapshot()
                stop_record_memory_history()

            # once in a while evaluate the validation dataset
            if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
                # stop the clock
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.perf_counter() - t0)
                # run validation batches
                fsdp_model.eval()  # Changed from ddp_model
                val_loader.reset()
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    with ctx:  # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                        loss = fsdp_model(x_val, y_val)  # Changed from ddp_model
                        val_loss += loss.detach()
                        del loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
                # log val loss
                print0(
                    f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (timed_steps - 1):.2f}ms',
                    console=True,
                )
                if master_process and not args.disable_wandb:
                    wandb.log(
                        {
                            'val_loss': val_loss,
                            'train_time': training_time_ms,
                            'step': step,
                            'tokens_seen': tokens_seen,
                            'step_avg': training_time_ms / (timed_steps - 1),
                        }
                    )

                if master_process and trial:
                    # report intermediate values to optuna for the purpose of pruning trials
                    # optuna pruning does not support multi-objective optimization
                    trial.report(val_loss, step)
                    if trial.should_prune():
                        print0('Trial pruned!', console=True)
                        dist.destroy_process_group()
                        if not args.disable_wandb:
                            wandb.finish(exit_code=-1)
                        model.cpu()
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                        raise optuna.TrialPruned

                # start the clock again
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # if we hit the speedrun target, we're done
                if val_loss <= SPEEDRUN_TARGET:
                    print0(f'Speedrun target achieved at step {step} with val_loss {val_loss:.4f} at time {training_time_ms:.2f}ms !')

            # bit confusing: we want to make sure to eval on 0th iteration
            # but also after the very last iteration. so we loop for step <= num_iterations
            # instead of just < num_iterations (one extra due to <=), only to do
            # the validation/sampling one last time, and then we break right here as we're done.
            if last_step:
                break

            # --------------- TRAINING SECTION BEGIN -----------------
            fsdp_model.train()  # Changed from ddp_model
            for i in range(1, train_accumulation_steps + 1):
                # forward pass
                with ctx:
                    loss = fsdp_model(x, y)  # Changed from ddp_model
                # advance the dataset for the next batch
                x, y = train_loader.next_batch()
                # backward pass
                train_loss = loss.detach()
                loss.backward()  # FSDP handles gradient synchronization automatically
            
            # step the optimizer
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
            # null the gradients
            fsdp_model.zero_grad(set_to_none=True)  # Changed from ddp_model

            # --------------- TRAINING SECTION END -------------------
            # everything that follows now is just diagnostics, prints, logging, etc.

            # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
            if master_process:
                tokens_seen += tokens_per_fwdbwd
                approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
                tokens_per_second = tokens_seen / (approx_time / 1000) if approx_time > 0 else 0
                mem_str = '' if step > 10 else report_mem_consumption()
                print0(
                    f'step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms tokens_seen:{tokens_seen:.2e} tokens/sec:{tokens_per_second:.2e} {mem_str}',
                    console=True,
                )
                if not args.disable_wandb:
                    wandb.log(
                        {
                            'train_loss': train_loss.item(),
                            'train_time': approx_time,
                            'step': step + 1,
                            'step_avg': approx_time / timed_steps,
                            'tokens_seen': tokens_seen,
                            'lr_schedule': get_lr(step),
                            'tokens_per_second': tokens_per_second,
                        }
                    )

            if prof:
                prof.step()

    # -------------------------------------------------------------------------
    print0(f'peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')
    # clean up nice
    dist.destroy_process_group()
    if master_process and not args.disable_wandb:
        wandb.finish()

    # free up memory in case we are running trials with optuna
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return val_loss, training_time_ms


if __name__ == '__main__':
    main()