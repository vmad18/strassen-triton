import sys
import os
import torch
import torch.distributed as dist
import time
from strassen import run_strassen2, run_matmul_fp32_accum, run_strassen, strassen_matmul_two_layers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DataDistributed:
    def __call__(self, a: torch.Tensor, b: torch.Tensor, func, three: bool = False) -> torch.Tensor:
        # Move tensors to GPU
        a = a.cuda()
        b = b.cuda()
        if three:
            c = torch.empty((*a.shape[:-1], b.shape[-1]), dtype=torch.float16, device='cuda')
            func(a, b, c)
            return c
        else:
            # For functions like torch.matmul that only take 2 arguments
            return func(a, b)

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
        dd(a, b, strassen_matmul_two_layers, three=False)
        dd(a, b, run_strassen, three=True)
        dd(a, b, run_matmul_fp32_accum, three=True)
        dd(a, b, torch.matmul, three=False)  # Explicitly set three=False for torch.matmul

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, strassen_matmul_two_layers, three=False)
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
        dd(a, b, run_strassen, three=True)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    triton_strassen_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        c.zero_()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_matmul_fp32_accum, three=True)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    triton_mm_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    for i in range(num_runs):
        c.zero_()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, torch.matmul, three=False)
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
    sizes = [4096, 8192, 16384, 32768, 65536]

    print(
        f"{'Size':>6}| {'Strassen (torch.compile)':>10} | {'Strassen (Triton)':>10} | {'Triton_MM':>10} | {'PyTorch':>10} | {'Strassen_Speedup (torch.compile)':>8} | {'Strassen_Speedup (Triton)':>8} | {'Triton Speedup':>8} | {'Max Diff':>8} | {'TF/s':>6}")
    print("-" * 150)

    for size in sizes:
        results = benchmark_matmul(
            M=size, N=size, K=size,
            num_warmup=1,
            num_runs=2
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
              f"{results['triton_strassen2_tflops']:>8.3f} ")  # f"{results['triton_mm_tflops']:>8.3f} "


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print("\nProfiling different matrix sizes:")
    profile_mats()

