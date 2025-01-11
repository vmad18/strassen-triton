import sys

import torch
import torch.distributed as dist
import time
from strassen import run_strassen_2_layer_fp32_accum, run_strassen_fp32_accum, run_matmul_fp32_accum

class DataDistributed:
    def __init__(self, world_size=4):
        self.world_size = world_size
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group('nccl')
        
        self.rank = dist.get_rank()
        
    def __call__(self, a: torch.Tensor, b: torch.Tensor, func, three: bool = True) -> torch.Tensor:
        assert len(a.shape) == len(b.shape) == 2, "Inputs must be 2D matrices"
        M, K = a.shape
        K_, N = b.shape
        assert K == K_, "inner dims must match"
        
        local_M = M // self.world_size
        start_idx = self.rank * local_M
        end_idx = start_idx + local_M
        local_a = a[start_idx:end_idx].cuda()
        local_b = b.cuda()
        
        local_c = torch.empty((local_M, N), dtype=torch.float16, device='cuda')
        
        if three:
            func(local_a, local_b, local_c)
        else:
            local_c = func(local_a, local_b)

        gathered_c = [torch.empty_like(local_c) for _ in range(self.world_size)]
        dist.all_gather(gathered_c, local_c)
        
        final_c = torch.cat(gathered_c, dim=0)
        return final_c

def benchmark_matmul(
        M: int,
        N: int,
        K: int,
        num_warmup: int = 10,
        num_runs: int = 100,
        device: str = "cuda"
) -> dict:

    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.zeros((M, N), device=device, dtype=torch.float32)

    gt_mm = torch.matmul(a, b)
    dd = DataDistributed()

    for _ in range(num_warmup):
        dd(a, b, run_strassen_2_layer_fp32_accum)
        dd(a, b, run_strassen_fp32_accum)
        dd(a, b, run_matmul_fp32_accum)
        dd(a, b, torch.matmul, False)

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]


    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_strassen_2_layer_fp32_accum)
        # run_strassen_2_layer_fp32_accum(a, b, c, 64)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    triton_strassen2_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    max_diff = torch.max(torch.abs(c - gt_mm)).item()


    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        c.zero_()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_strassen_fp32_accum)
        # run_strassen_fp32_accum(a, b, c)
        end_events[i].record()

    torch.cuda.synchronize()        
    torch.cuda.empty_cache()
    triton_strassen_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]


    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    triton_mm_times = []
    for i in range(num_runs):
        c.zero_()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_matmul_fp32_accum)
        # run_matmul_fp32_accum(a, b, c)
        end_events[i].record()

    torch.cuda.synchronize()        
    torch.cuda.empty_cache()
    triton_mm_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    torch_times = []
    for i in range(num_runs):
        c.zero_()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, torch.matmul)
        # torch.matmul(a, b)
        end_events[i].record()

    torch.cuda.synchronize()        
    torch.cuda.empty_cache()
    torch_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    triton_strassen2_avg = sum(triton_strassen2_times) / len(triton_strassen2_times)
    triton_strassen_avg = sum(triton_strassen_times) / len(triton_strassen_times)
    triton_mm_avg = sum(triton_mm_times) / len(triton_mm_times)
    torch_avg = sum(torch_times) / len(torch_times)

    triton_strassen2_min = min(triton_strassen2_times)
    triton_strassen_min = min(triton_strassen_times)
    triton_mm_min = min(triton_mm_times)
    torch_min = min(torch_times)

    results = {
        "triton_strassen2_mean_ms": triton_strassen2_avg * 1000,
        "triton_strassen_mean_ms": triton_strassen_avg * 1000,
        "triton_mm_mean_ms": triton_mm_avg * 1000,
        "torch_mean_ms": torch_avg * 1000,
        "triton_strassen2_min_ms": triton_strassen2_min * 1000,
        "triton_strassen_min_ms": triton_strassen_min * 1000,
        "triton_mm_min_ms": triton_mm_min * 1000,
        "torch_min_ms": torch_min * 1000,
        "triton_strassen2_tflops": (2 * M * N * K) / (triton_strassen2_min * 1e12 + 1e-6),
        "triton_strassen_tflops": (2 * M * N * K) / (triton_strassen_min * 1e12 + 1e-6),
        "triton_mm_tflops": (2 * M * N * K) / (triton_mm_min * 1e12 + 1e-6),
        "torch_tflops": (2 * M * N * K) / (torch_min * 1e12 + 1e-6),
        "strassen2_max_diff": max_diff,
        "strassen2_mean_speedup": torch_avg / triton_strassen2_avg,
        "strassen_mean_speedup": torch_avg / triton_strassen_avg,
        "triton_mm_mean_speedup": torch_avg / triton_mm_avg
    }

    return results


def profile_mats():

    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(f"{'Size':>6}| {'Strassen2':>10} | {'Strassen':>10} | {'Triton_MM':>10} | {'PyTorch':>10} | {'Strassen2_Speedup':>8} | {'Strassen_Speedup':>8} | {'Triton Speedup':>8} | {'Max Diff':>8} | {'TF/s':>6}")
    print("-" * 100)

    for size in sizes:
        results = benchmark_matmul(
            M=size, N=size, K=size,
            num_warmup=10,
            num_runs=50
        )

        print(f"{size:>6} | "
              f"{results['triton_strassen2_mean_ms']:>10.3f} | "
              f"{results['triton_strassen_mean_ms']:>10.3f} | "
              f"{results['triton_mm_mean_ms']:>10.3f} | "
              f"{results['torch_mean_ms']:>10.3f} | "
              f"{results['strassen2_mean_speedup']:>8.3f}x | "
              f"{results['strassen_mean_speedup']:>8.3f}x | "
              f"{results['triton_mm_mean_speedup']:>8.3f}x | "
              f"{results['strassen2_max_diff']:>10.3e} | "
              f"{results['triton_strassen2_tflops']:>8.3f} ") # f"{results['triton_mm_tflops']:>8.3f} "


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print("\nProfiling different matrix sizes:")
    profile_mats()
