import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import math 

from triton.testing import do_bench


USE_AGG: bool = True
USE_TF32: bool = True


if USE_AGG:
    from fixed_strassen_fp32_agg import strassen_matmul_n_layers, agg_dtype
    # agg_dtype = torch.float64 # this does not work, have to manually change it in the fixed_strassen_fp32_agg.py code 
    print(f"==> Using high precision aggregation with {agg_dtype}")
else:
    from fixed_strassen import strassen_matmul_n_layers
    print("==> Not using aggregation")


if USE_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # get tf32 to work on amd gpus
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    os.environ["HIPBLASLT_ALLOW_TF32"] = "1"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    print("==> Using TF32")
else:
    print("==> NOT using TF32")



torch._dynamo.reset()

torch.backends.cudnn.benchmark = True


class LinearStrassen(nn.Linear):
    def __init__(self,
                 features: int, 
                 bias: bool = False,
                 dtype=torch.float32,
                 depth: int = 1,
                 device: str = "cuda") -> None:
        is_power_of_two = (features > 0) and ((features & (features - 1)) == 0)
        assert is_power_of_two, "Features must be a power of 2."
        
        super().__init__(features, features, bias=bias, device=device, dtype=dtype)
        self.depth = depth
        self.to(device=device, dtype=dtype)
        self.base_dtype = dtype

    def forward(self, input_tensor: Tensor) -> Tensor:
        *batch_dims, D_in = input_tensor.shape 
        
        assert D_in == self.in_features, f"Input feature dimension mismatch. Expected {self.in_features}, got {D_in}"
        assert len(batch_dims) == 1 and batch_dims[0] == self.in_features, \
            f"For NxN input benchmark, expected input shape ({self.in_features}, {self.in_features}), got {input_tensor.shape}"

        D = self.in_features
        input_reshaped = input_tensor.reshape(D, D) 
        M = D

        W_T = self.weight.T 
        
        if M < D: 
            pad_rows_x = D - M
            X_padded = F.pad(input_reshaped, (0, 0, 0, pad_rows_x), "constant", 0)
        else: 
            X_padded = input_reshaped # X_padded is (D,D)

        out_padded = strassen_matmul_n_layers(X_padded, W_T, n_depth=self.depth, base_dtype=self.base_dtype)

        out_unpadded = out_padded[:M, :D]

        out = out_unpadded.view(D, self.out_features)
        if self.bias is not None:
            out = out + self.bias  
        return out

@torch.no_grad()
def benchmark_linear_layers(
    N_dim: int,           
    strassen_depth: int,
    use_bias: bool = False,
    dtype=torch.float32,
    device: str = "cuda",
    num_warmup: int = 25, 
    num_runs: int = 100
) -> dict:
    """
    Benchmarks LinearStrassen (NxN input) against nn.Linear (NxN input).
    N_dim must be a power of 2.
    """
    # torch._dynamo.reset()

    assert (N_dim > 0) and ((N_dim & (N_dim - 1)) == 0), "N_dim (features) must be a power of 2."

    results = {
        "N_dim": N_dim,
        "strassen_depth": strassen_depth,
        "bias": use_bias
    }

    # Input tensor is (N_dim, N_dim)
    input_x = torch.randn(N_dim, N_dim, device=device, dtype=dtype)

    try:
        linear_std = nn.Linear(N_dim, N_dim, bias=use_bias, device=device, dtype=dtype)
        linear_strassen = LinearStrassen(features=N_dim, bias=use_bias, dtype=dtype, 
                                         depth=strassen_depth, device=device)
        
        
        # linear_std.compile(mode="max-autotune")
        # linear_strassen.compile(mode="max-autotune")
       
        linear_std = torch.compile(
                                linear_std,
                                backend="inductor",
                                mode="max-autotune",
                                # fullgraph=True,
                                # dynamic=False,
                            )

        linear_strassen = torch.compile(
                        linear_strassen,
                        backend="inductor",
                        mode="max-autotune",
                        # fullgraph=True,
                        # dynamic=False,
                    )

       # linear_std = torch.compile(linear_std, mode = "max-autotune-no-cudagraphs")
        # linear_strassen = torch.compile(linear_strassen, mode = "max-autotune-no-cudagraphs")

        linear_std.eval()
        linear_strassen.eval()

        linear_strassen.weight.data.copy_(linear_std.weight.data)
        if use_bias and linear_std.bias is not None: # Check if bias actually exists
            linear_strassen.bias.data.copy_(linear_std.bias.data)

        linear_std.eval()
        linear_strassen.eval()
    except Exception as e:
        print(f"  Error initializing models for N_dim={N_dim}: {e}")
        results['error_init'] = str(e)
        return results

    print(f"  Performing correctness check (N_dim={N_dim}, Depth={strassen_depth})...")
    try:
        with torch.no_grad():
            out_std = linear_std(input_x).clone()
            out_strassen = linear_strassen(input_x).clone()

        abs_diff = torch.abs(out_std - out_strassen)
        results['max_abs_diff'] = torch.max(abs_diff).item()
        results['allclose'] = torch.allclose(out_std, out_strassen)
        print(f"    Max Abs Diff: {results['max_abs_diff']:.3e}, Allclose: {results['allclose']}")
    except Exception as e:
        print(f"    Error during correctness check: {e}")
        results['max_abs_diff'] = float('nan')
        results['allclose'] = False
        results['error_correctness'] = str(e)

    print(f"  JIT Warm-up (1 run per function)...")
    try:
        with torch.no_grad():
            _ = linear_std(input_x).clone()
            _ = linear_strassen(input_x).clone()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"    Error during JIT warmup: {e}")
        results['error_warmup'] = str(e)

    print(f"  Benchmarking with do_bench (warmup={num_warmup}, runs={num_runs})...")
    flops = 2 * (N_dim ** 3) 
    if use_bias:
        flops += N_dim * N_dim 

    with torch.no_grad():
        try:
            fn_std = lambda: linear_std(input_x).clone()
            times_std_ms_list = do_bench(fn_std, warmup=num_warmup, rep=num_runs, return_mode="all")
            if times_std_ms_list:
                results['std_mean_ms'] = sum(times_std_ms_list) / len(times_std_ms_list)
                results['std_min_ms'] = min(times_std_ms_list)
                results['std_tflops'] = flops / (results['std_min_ms'] / 1000.0) / 1e12 if results['std_min_ms'] > 0 else float('nan')
            else:
                results['std_mean_ms'] = results['std_min_ms'] = results['std_tflops'] = float('nan')
            print(f"    nn.Linear mean: {results['std_mean_ms']:.4f} ms")
        except Exception as e:
            print(f"    Error benchmarking nn.Linear: {e}")
            results['std_mean_ms'] = results['std_min_ms'] = results['std_tflops'] = float('nan')
            results['error_bench_std'] = str(e)

        try:
            fn_strassen = lambda: linear_strassen(input_x).clone()
            times_strassen_ms_list = do_bench(fn_strassen, warmup=num_warmup, rep=num_runs, return_mode="all")
            if times_strassen_ms_list:
                results['strassen_mean_ms'] = sum(times_strassen_ms_list) / len(times_strassen_ms_list)
                results['strassen_min_ms'] = min(times_strassen_ms_list)
                # For Strassen TFLOPs, we still use the standard FLOP count for apples-to-apples comparison of "useful work"
                results['strassen_tflops'] = flops / (results['strassen_min_ms'] / 1000.0) / 1e12 if results['strassen_min_ms'] > 0 else float('nan')
            else:
                results['strassen_mean_ms'] = results['strassen_min_ms'] = results['strassen_tflops'] = float('nan')
            print(f"    LinearStrassen mean: {results['strassen_mean_ms']:.4f} ms")
        except Exception as e:
            print(f"    Error benchmarking LinearStrassen: {e}")
            results['strassen_mean_ms'] = results['strassen_min_ms'] = results['strassen_tflops'] = float('nan')
            results['error_bench_strassen'] = str(e)
            
    if not torch.isnan(torch.tensor(results.get('std_mean_ms', float('nan')))) and \
       not torch.isnan(torch.tensor(results.get('strassen_mean_ms', float('nan')))) and \
       results.get('strassen_mean_ms', 0) != 0:
        results['speedup_mean'] = results['std_mean_ms'] / results['strassen_mean_ms']
    else:
        results['speedup_mean'] = float('nan')
        
    torch.cuda.empty_cache()
    return results

def profile_linear_layers():
    print("Starting Linear Layer Profiling (NxN Features, NxN Input)...\n")
    # Configurations: (N_dimension, strassen_depth)
    # N_dimension must be a power of 2. This will be used for in_features, out_features, and batch_size.
    configurations = [
        # (N_dim, strassen_depth)
        # (64, 1), 
        # (128, 1), 
        # (256, 1), (256, 2), 
        # (512, 1), (512, 2), 
        (1024, 1), (1024, 2),
        (2048, 1), (2048, 2),
        (4096, 1), (4096, 2),
        (8192, 1), (8192, 2),  (8192, 3), # (8192, 4),
        (8192*2, 1), (8192*2, 2), (8192*2, 3), # (8192*2, 4),
        (8192*4, 1), (8192*4, 2), (8192*4, 3), (8192*4, 4)# , (8192*4, 5),
    ]

    all_results_data = []
    
    num_warmup_base = 25 
    num_runs_base = 100

    header = (
        f"{'N_dim':>7s} | {'Depth':>5s} | "
        f"{'Std Mean(ms)':>14s} | {'Strassen Mean(ms)':>18s} | {'Speedup':>9s} | "
        f"{'Std TFLOPs':>12s} | {'Strassen TFLOPs':>17s} | "
        f"{'MaxAbsDiff':>12s} | {'Allclose':>8s}"
    )
    
    for N_dim, depth in configurations:
        print(f"\nProfiling: N_dim={N_dim}, Strassen Depth={depth}")
        
        current_runs = num_runs_base
        current_warmup = num_warmup_base
        if N_dim >= 1024:
            current_runs = 350
            current_warmup = 10
        if N_dim >= 2048:
            current_runs = 350
            current_warmup = 10
        if N_dim >= 4096: # For very large cases, fewer runs
            current_runs = 50
            current_warmup = 10

        res = benchmark_linear_layers(
            N_dim=N_dim, 
            strassen_depth=depth,
            num_warmup=current_warmup,
            num_runs=current_runs
        )
        all_results_data.append(res)

        print(header)
        print("-" * len(header))
        
        if res.get('error_init') or res.get('error_correctness') or res.get('error_warmup') or res.get('error_bench_std') or res.get('error_bench_strassen'):
            error_msg = res.get('error_init', '') or res.get('error_correctness', '') or res.get('error_warmup', '') or res.get('error_bench_std', '') or res.get('error_bench_strassen', 'Unknown Error')
            print(f"{N_dim:>7} | {depth:>5} | {'ERROR: ' + str(error_msg).splitlines()[0]:<100}") # Print first line of error
        else:
            print(
                f"{res.get('N_dim', 'N/A'):>7} | {res.get('strassen_depth', 'N/A'):>5} | "
                f"{res.get('std_mean_ms', float('nan')):>14.4f} | {res.get('strassen_mean_ms', float('nan')):>18.4f} | "
                f"{res.get('speedup_mean', float('nan')):>8.2f}x | "
                f"{res.get('std_tflops', float('nan')):>12.2f} | {res.get('strassen_tflops', float('nan')):>17.2f} | "
                f"{res.get('max_abs_diff', float('nan')):>12.3e} | {str(res.get('allclose', 'N/A')):>8s}"
            )

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        profile_linear_layers()
