import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch import Module 

from ..fixed_strassen import strassen_matmul_two_layers, _strassen_base  

from typing import Optional, Tuple

from einops import rearrange
from math import sqrt, log, log2

import logging as logger

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN = True
except:
    logger.info("Cannot use Flash Attention, using regular attention")
    FLASH_ATTN = False


def scaled_dot_product(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scaling_norm: Optional[float] = None,
        mask: Optional[Tensor] = None,
        drop_r: float = 0.,
        training: bool = False) -> Tensor:
    if not FLASH_ATTN:
        norm = scaling_norm if not (scaling_norm is None) else 1. / sqrt(q.shape[-1])
        mask = torch.zeros((1, 1, q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.float32) if mask is None else mask

        attends = (q @ k.transpose(-2, -1)).float() * norm + mask

        attn = F.softmax(attends, dtype=torch.float32, dim=-1)
        attn = F.dropout(attn, p=drop_r, training=training)
        return attn @ v.to(torch.float32)

    return flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                           dropout=drop_r, softmax_scale=scaling_norm, causal=True, window_size=(-1, -1),
                           alibi_slopes=None, deterministic=True)

def causal_mask(
        S: int,
        W: int,
        shift: int | None = 1,
        value: float | None = None 
    ) -> Tensor:
    shift = shift if shift != None else 1
    value = value if value != None else float("-inf") # torch.finfo(torch.get_default_dtype()).min
    mask = torch.full((S, W), fill_value=value).triu(diagonal=shift)
    return mask

class RoPE(Module):

    def __init__(self,
                 head_dim: int, 
                 rope_base: int,
                 max_tokens: int, 
                 scaling: float = 1.,
                 device: str = "cuda") -> None:
        super().__init__()

        self.dim = head_dim
        self.base = rope_base
        self.scaling = scaling
        self.max_tokens = max_tokens

        self.device = device

    def comp_rots(self) -> Tensor:
        theta = torch.exp(
            -log(self.base) * torch.arange(0, self.dim, 2, device=self.device, dtype=torch.int64) / self.dim)[
            None, ...]

        m = torch.arange(self.max_tokens, device=self.device, dtype=torch.float32)[..., None].float()
        freqs = m * theta
        mag = torch.ones_like(freqs, device=self.device)

        return torch.polar(mag, freqs)

    @staticmethod
    def pass_qk(q: Tensor, 
                k: Tensor, 
                shift: int = 0) -> Tuple[Tensor, Tensor]:
        return q, k

    def _to_complex(self, x: Tensor) -> Tensor:
        return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    def _to_real(self, x: Tensor) -> Tensor:
        return torch.view_as_real(x)

    @torch.no_grad()
    def forward(self,
                q: Tensor,
                k: Tensor,
                shift: int = 0) -> Tuple[Tensor, Tensor]:
        *_, s, d = q.shape

        dtype = q.dtype

        q, k = q.float(), k.float()

        rots = self.comp_rots()[shift : shift + s].reshape(1, 1, s, d // 2)  # self.rotations[shift:shift+s]

        _q = self._to_complex(q) * rots
        _k = self._to_complex(k) * rots

        rq = self._to_real(_q).to(dtype)
        rk = self._to_real(_k).to(dtype)

        return rq.reshape(*rq.shape[:-2], d).to(dtype), rk.reshape(*rk.shape[:-2], d).to(dtype)


class CausalAttention(Module):
    def __init__(self,
                 dim: int,
                 heads: int,
                 max_tokens: int, 
                 bias: bool = False,
                 base: int = int(1e4),
                 drop_r: float = .1,
                 device: str = "cuda",
                 layer_idx: Optional[int] = None):
        super().__init__()

        self.h = heads

        self.to_q = nn.Linear(dim, dim, bias=bias, device=device)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=bias, device=device)
        self.o_proj = nn.Linear(dim, dim, device=device)
    
        self.drop_r = drop_r

        self.rope = RoPE(head_dim = dim // heads, rope_base=base, max_tokens = max_tokens)

        self.layer_idx = layer_idx

    def forward(self,
                x: Tensor,
                ctx: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                shift: int = 0, ) -> Tensor:
        b, s, *_ = x.shape

        if ctx is None:
            qkv = (self.to_q(x), *self.to_kv(x).chunk(2, -1))
        else:
            qkv = (self.to_q(x), *self.to_kv(ctx).chunk(2, -1))
        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )
        q, k = self.rope.forward(q, k, shift=shift)
        x = scaled_dot_product(q, k, v, mask=mask, drop_r=self.drop_r)
        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)
        return F.dropout(x, p=self.drop_r, training=True)


class FeedForward(Module):

    def __init__(self, 
                 dim: int,
                 expansion_scale: int, 
                 bias: bool = True,
                 gate: bool = True,
                 nl = F.silu,
                 device: str = "cuda",
                 layer_idx: Optional[int] = None):
        super().__init__()
        

        hidden = int(dim * expansion_scale)

        self.proj_up = nn.Linear(dim, hidden, bias=bias, device=device)
        self.gate = nn.Linear(dim, hidden, bias=bias, device=device) \
            if gate else nn.Identity()
        self.proj_down = nn.Linear(hidden, dim, bias=bias, device=device)
       
        self.nl = nl
        self.layer_idx = layer_idx

    def forward(self, x: Tensor) -> Tensor:
        o, gate = self.proj_up(x), self.gate(x)
        return self.proj_down(self.nl(o) * gate)



class LinearStrassen(nn.Linear):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 dtype=None, 
                 depth: int = 2, 
                 device: str = "cuda") -> None:
        assert in_features == out_features and (log2(in_features)).is_integer(), "Transform must be square and a power of 2"
        
        super().__init__(in_features, out_features, bias, device, dtype)
        self.depth = depth 


    def forward(self, input: Tensor) -> Tensor:
        if self.depth == 1:
            out = _strassen_base(input, self.weight.T)
        else:
            out = strassen_matmul_two_layers(input, self.weight.T)
        if self.bias is not None:
            out += self.bias 
        return out 






