import torch
import time

from strassen import strassen_matmul_two_layers
from fixed_strassen import strassen_matmul_n_layers

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DataDistributed:
    """
    Helper class to call matrix multiplication functions.
    Assumes input tensors 'a' and 'b' are already on the target CUDA device.
    """
    def __call__(self, a: torch.Tensor, b: torch.Tensor, func, three: bool = False) -> torch.Tensor:
        if three:
            c_out = torch.empty((*a.shape[:-1], b.shape[-1]), dtype=torch.float16, device=a.device)
            func(a, b, c_out) 
            return c_out
        else:
            return func(a, b)

def _execute_runs(
    a: torch.Tensor,
    b: torch.Tensor,
    func_to_benchmark,
    num_runs: int,
    dd_instance: DataDistributed
) -> list[float]:
    """
    Helper function to execute benchmark runs for a given matmul function and record timings.
    """
    run_times_ms = []
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000) 
        start_events[i].record()
        _ = dd_instance(a, b, func_to_benchmark, three=False)
        end_events[i].record()

    torch.cuda.synchronize() 
    
    for i in range(num_runs):
        run_times_ms.append(start_events[i].elapsed_time(end_events[i]))
        
    return run_times_ms

def benchmark_matmul(
    M: int, 
    N: int, 
    K: int, 
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda"
) -> dict:
    """
    Benchmarks different matrix multiplication implementations.
    For square matrices, M, N, and K will be the same (the 'size').
    """
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)

    gt_mm = torch.matmul(a, b) 
    dd = DataDistributed()
    
    benchmark_targets = [
        {"func": strassen_matmul_n_layers, "key": "strassen1", "name": "Strassen1 (torch.compile)"},
        {"func": strassen_matmul_two_layers, "key": "strassen2", "name": "Strassen2 (torch.compile)"},
        {"func": torch.matmul, "key": "torch", "name": "PyTorch"},
    ]

    print(f"  Warm-up ({num_warmup} runs per function)...")
    for _ in range(num_warmup):
        for target in benchmark_targets:
            _ = dd(a, b, target["func"], three=False)
    torch.cuda.synchronize()

    results = {} 

    if any(target['key'] == 'strassen1' for target in benchmark_targets):
        print("  Performing correctness check for Strassen1...")
        result_strassen1 = dd(a, b, strassen_matmul_n_layers, three=False)
        torch.cuda.synchronize()
        max_diff_s1 = torch.max(torch.abs(result_strassen1 - gt_mm)).item()
        results["strassen1_max_diff"] = max_diff_s1
        print(f"  Max difference for strassen_matmul_n_layers (Strassen1): {max_diff_s1:.3e}")

    if any(target['key'] == 'strassen2' for target in benchmark_targets):
        print("  Performing correctness check for Strassen2...")
        result_strassen2 = dd(a, b, strassen_matmul_two_layers, three=False)
        torch.cuda.synchronize()
        max_diff_s2 = torch.max(torch.abs(result_strassen2 - gt_mm)).item()
        results["strassen2_max_diff"] = max_diff_s2
        print(f"  Max difference for strassen_matmul_two_layers (Strassen2): {max_diff_s2:.3e}")

    all_times_ms = {}
    print(f"  Benchmarking ({num_runs} runs per function)...")
    for target in benchmark_targets:
        print(f"    Benchmarking {target['name']}...")
        times_ms = _execute_runs(a, b, target["func"], num_runs, dd)
        all_times_ms[target["key"]] = times_ms
        torch.cuda.empty_cache() 

    torch_times = all_times_ms.get("torch")
    if not torch_times:
        print("Warning: PyTorch benchmark (key 'torch') not found. Speedups will be NaN.")
        torch_avg_ms = float('nan')
    else:
        torch_avg_ms = sum(torch_times) / len(torch_times)

    for target in benchmark_targets:
        key = target["key"]
        current_times_ms = all_times_ms.get(key)

        if not current_times_ms:
            avg_ms = float('nan')
            min_ms = float('nan')
        else:
            avg_ms = sum(current_times_ms) / len(current_times_ms)
            min_ms = min(current_times_ms)
        
        results[f"{key}_mean_ms"] = avg_ms
        results[f"{key}_min_ms"] = min_ms
        
        if not torch.isnan(torch.tensor(min_ms)):
             results[f"{key}_tflops"] = (2 * M * N * K) / (min_ms / 1000.0 * 1e12 + 1e-9)
        else:
            results[f"{key}_tflops"] = float('nan')

        if key != "torch" and not torch.isnan(torch.tensor(torch_avg_ms)) and not torch.isnan(torch.tensor(avg_ms)) and avg_ms != 0:
            results[f"{key}_mean_speedup"] = torch_avg_ms / avg_ms
        elif key != "torch":
             results[f"{key}_mean_speedup"] = float('nan')
    
    if "strassen1_max_diff" not in results: results["strassen1_max_diff"] = float('nan')
    if "strassen2_max_diff" not in results: results["strassen2_max_diff"] = float('nan')

    return results

def profile_mats():
    """
    Profiles matrix multiplication for various square matrix sizes.
    Prints results with repeated headers for each size.
    """
    sizes = [4096, 8192, 16384, 32768, 65536]
    
    header_string = (
        f"{'Size (N)':>9s}| {'S1 ms':>10s} | {'S2 ms':>10s} | {'PyTorch ms':>12s} | "
        f"{'S1 Speedup':>10s} | {'S2 Speedup':>10s} | "
        f"{'S1 MaxDiff':>10s} | {'S2 MaxDiff':>10s} | "
        f"{'S1 TFLOPs':>10s} | {'S2 TFLOPs':>10s}"
    )
    separator_string = "-" * (len(header_string)) 

    for size_n in sizes:
        M, N, K = size_n, size_n, size_n

        if size_n > 16384:
            current_num_runs = 100
            current_num_warmup = 10
        elif size_n > 8192:
            current_num_runs = 200
            current_num_warmup = 10
        else:
            current_num_runs = 500
            current_num_warmup = 10

        print(f"\nProfiling for square matrices of size: {N}x{N} (M={M}, N={N}, K={K})")
        print(f"Warmup: {current_num_warmup} runs, Timed: {current_num_runs} runs")
        

        results = benchmark_matmul(
            M=M, N=N, K=K,
            num_warmup=current_num_warmup,
            num_runs=current_num_runs
        )
        
        print(header_string)
        print(separator_string)
        print(f"{N:>9} | "
              f"{results.get('strassen1_mean_ms', float('nan')):>9.3f} | "
              f"{results.get('strassen2_mean_ms', float('nan')):>9.3f} | "
              f"{results.get('torch_mean_ms', float('nan')):>11.3f} | "
              f"{results.get('strassen1_mean_speedup', 0.0):>9.3f}x | "
              f"{results.get('strassen2_mean_speedup', 0.0):>9.3f}x | "
              f"{results.get('strassen1_max_diff', float('nan')):>10.3e} | "
              f"{results.get('strassen2_max_diff', float('nan')):>10.3e} | "
              f"{results.get('strassen1_tflops', float('nan')):>9.3f} | "
              f"{results.get('strassen2_tflops', float('nan')):>9.3f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("\nStarting Matrix Multiplication Profiling...")
        profile_mats()
