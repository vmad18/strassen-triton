import torch
import time
from strassen import run_strassen_fp32_accum, run_matmul_fp32_accum

def benchmark_matmul(
        M: int = 512,
        N: int = 512,
        K: int = 512,
        num_warmup: int = 10,
        num_runs: int = 100,
        device: str = "cuda"
) -> dict:
    torch.manual_seed(3331)

    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.zeros((M, N), device=device, dtype=torch.float32)

    torch_c = torch.matmul(a, b)

    for _ in range(num_warmup):
        c.zero_()
        run_strassen_fp32_accum(a, b, c)
        run_matmul_fp32_accum(a, b, c)
        torch.matmul(a, b)

    torch.cuda.synchronize()

    triton_strassen_times = []
    for _ in range(num_runs):
        c.zero_()
        start = time.perf_counter()
        run_matmul_fp32_accum(a, b, c)
        torch.cuda.synchronize()
        end = time.perf_counter()
        triton_strassen_times.append(end - start)

    triton_mm_times = []
    for _ in range(num_runs):
        c.zero_()
        start = time.perf_counter()
        run_strassen_fp32_accum(a, b, c)
        torch.cuda.synchronize()
        end = time.perf_counter()
        triton_mm_times.append(end - start)


    torch_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        torch_times.append(end - start)

    max_diff = torch.max(torch.abs(c - torch_c)).item()

    triton_strassen_avg = sum(triton_strassen_times) / len(triton_strassen_times)
    triton_mm_avg = sum(triton_mm_times) / len(triton_mm_times)
    torch_avg = sum(torch_times) / len(torch_times)

    triton_strassen_min = min(triton_strassen_times)
    triton_mm_min = min(triton_mm_times)
    torch_min = min(torch_times)

    results = {
        "triton_strassen_mean_ms": triton_strassen_avg * 1000,
        "triton_mm_mean_ms": triton_mm_avg * 1000,
        "torch_mean_ms": torch_avg * 1000,
        "triton_strassen_min_ms": triton_strassen_min * 1000,
        "triton_mm_min_ms": triton_mm_min * 1000,
        "torch_min_ms": torch_min * 1000,
        "triton_strassen_tflops": (2 * M * N * K) / (triton_strassen_min * 1e12),
        "triton_mm_tflops": (2 * M * N * K) / (triton_mm_min * 1e12),
        "torch_tflops": (2 * M * N * K) / (torch_min * 1e12),
        "strassen_max_diff": max_diff,
        "strassen_mean_speedup": torch_avg / triton_strassen_avg,
        "triton_mm_mean_speedup": torch_avg / triton_mm_avg
    }

    return results


def profile_mats():

    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(f"{'Size':>6} | {'Strassen':>10} | {'Triton_MM':>10} | {'PyTorch':>10} | {'Strassen_Speedup':>8} | {'Triton Speedup':>8} | {'Max Diff':>8} | {'TF/s':>6}")
    print("-" * 100)

    for size in sizes:
        results = benchmark_matmul(
            M=size, N=size, K=size,
            num_warmup=5,
            num_runs=50
        )

        print(f"{size:>6} | "
              f"{results['triton_strassen_mean_ms']:>10.3f} | "
              f"{results['triton_mm_mean_ms']:>10.3f} | "
              f"{results['torch_mean_ms']:>10.3f} | "
              f"{results['strassen_mean_speedup']:>8.3f}x | "
              f"{results['triton_mm_mean_speedup']:>8.3f}x | "
              f"{results['strassen_max_diff']:>10.3e} | "
              f"{results['triton_strassen_tflops']:>8.3f} ") # f"{results['triton_mm_tflops']:>8.3f} "


if __name__ == "__main__":
    print("\nProfiling different matrix sizes:")
    profile_mats()