"""
Performance Benchmarking for Matrix Multiplication Implementations

Compares the performance of the three matrix multiplication implementations:
1. Naive Triton implementation
2. Grouped Triton implementation (with memory access optimization)
3. PyTorch's native implementation

The benchmark measures performance in GB/s (gigabytes per second) across different matrix sizes.
"""

import triton
import triton.testing

import torch
from strassen import run_strassen_2_layer_fp32_accum, run_winograd_strassen, run_matmul_fp32_accum, run_old_winograd_strassen


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'],
        x_vals=[2 ** i for i in range(10, 16, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['strassen2', 'strassen', 'triton', 'old-winograd'],
        line_names=['Strassen(depth=2)', 'Strassen(depth=1)', 'Triton', 'StrassenOld(depth=1)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],
        ylabel='GB/s',
        plot_name='matmul-performance',
        args={},
    ))
def benchmark_matrix_size(square_matrix_size, provider):
    sz = square_matrix_size
    bsz = 2
    a = torch.rand((bsz, sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((bsz, sz, sz), device='cuda', dtype=torch.float32)
    c = torch.zeros((bsz, sz, sz), device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'strassen2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_strassen_2_layer_fp32_accum(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'strassen':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'old-winograd':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_old_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_matmul_fp32_accum(a, b, c),
            quantiles=quantiles
        )

    gbps = lambda ms: 12 * sz * sz / (ms * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 10, 12, 14],
        x_log=True,
        line_arg='provider',
        line_vals=['strassen2', 'strassen', 'triton', 'old-winograd'],
        line_names=['Strassen(depth=2)', 'Strassen(depth=1)', 'Triton', 'StrassenOld(depth=1)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],
        ylabel='GB/s',
        plot_name='matmul-batch-performance',
        args={},
    ))
def benchmark_batch_size(batch_size, provider):
    """Benchmark performance with different batch sizes"""
    matrix_size = 8192  # Fixed matrix size
    a = torch.rand((batch_size, matrix_size, matrix_size), device='cuda', dtype=torch.float32)
    b = torch.rand((batch_size, matrix_size, matrix_size), device='cuda', dtype=torch.float32)
    c = torch.zeros((batch_size, matrix_size, matrix_size), device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'strassen2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_strassen_2_layer_fp32_accum(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'strassen':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'old-winograd':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_old_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_matmul_fp32_accum(a, b, c),
            quantiles=quantiles
        )

    # Adjust GB/s calculation for batch size
    gbps = lambda ms: 12 * batch_size * matrix_size * matrix_size / (ms * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'],
        x_vals=[2 ** i for i in range(5, 12, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['strassen2', 'strassen', 'triton', 'old-winograd'],
        line_names=['Strassen(depth=2)', 'Strassen(depth=1)', 'Triton', 'StrassenOld(depth=1)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],
        ylabel='FLOPs',
        plot_name='matmul-performance',
        args={},
    ))
def benchmark_flops(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    c = torch.zeros((sz, sz), device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'strassen2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_strassen_2_layer_fp32_accum(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'strassen':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    elif provider == 'old-winograd':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_old_winograd_strassen(a, b, c),
            quantiles=quantiles
        )
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_matmul_fp32_accum(a, b, c),
            quantiles=quantiles
        )

    flops = lambda ms: 2 * sz ** 3 / (ms * 1e6)
    return flops(ms), flops(max_ms), flops(min_ms)

if __name__ == '__main__':
    benchmark_matrix_size.run(save_path='./plots', show_plots=True, print_data=True)
    benchmark_batch_size.run(save_path='./plots', show_plots=True, print_data=True)
    # benchmark_flops.run(save_path='./plots', show_plots=True, print_data=True)
