import triton
import triton.language as tl
import torch

@triton.jit
def strassen_kernel_fp32_accum(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        A_stride_m, A_stride_k,
        BLOCK_SIZE: tl.constexpr,
        HALF_BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    base_m = pid_m * BLOCK_SIZE
    base_n = pid_n * BLOCK_SIZE

    acc_11 = tl.zeros((HALF_BLOCK, HALF_BLOCK), dtype=tl.float32)
    acc_12 = tl.zeros((HALF_BLOCK, HALF_BLOCK), dtype=tl.float32)
    acc_21 = tl.zeros((HALF_BLOCK, HALF_BLOCK), dtype=tl.float32)
    acc_22 = tl.zeros((HALF_BLOCK, HALF_BLOCK), dtype=tl.float32)

    row_offs1 = base_m + tl.arange(0, HALF_BLOCK)
    row_offs2 = base_m + HALF_BLOCK + tl.arange(0, HALF_BLOCK)
    col_offs1 = base_n + tl.arange(0, HALF_BLOCK)
    col_offs2 = base_n + HALF_BLOCK + tl.arange(0, HALF_BLOCK)

    for k in range(0, K, BLOCK_SIZE):
        k_offs1 = k + tl.arange(0, HALF_BLOCK)
        k_offs2 = k + HALF_BLOCK + tl.arange(0, HALF_BLOCK)

        # load A matrix sections
        a_ptrs_11 = A_ptr + row_offs1[:, None] * A_stride_m + k_offs1[None, :]
        a_ptrs_12 = A_ptr + row_offs1[:, None] * A_stride_m + k_offs2[None, :]
        a_ptrs_21 = A_ptr + row_offs2[:, None] * A_stride_m + k_offs1[None, :]
        a_ptrs_22 = A_ptr + row_offs2[:, None] * A_stride_m + k_offs2[None, :]

        A_11 = tl.load(a_ptrs_11, mask=(row_offs1[:, None] < M) & (k_offs1[None, :] < K), other=0.)
        A_12 = tl.load(a_ptrs_12, mask=(row_offs1[:, None] < M) & (k_offs2[None, :] < K), other=0.)
        A_21 = tl.load(a_ptrs_21, mask=(row_offs2[:, None] < M) & (k_offs1[None, :] < K), other=0.)
        A_22 = tl.load(a_ptrs_22, mask=(row_offs2[:, None] < M) & (k_offs2[None, :] < K), other=0.)

        # load B matrix sections
        b_ptrs_11 = B_ptr + k_offs1[:, None] * A_stride_m + col_offs1[None, :]
        b_ptrs_12 = B_ptr + k_offs1[:, None] * A_stride_m + col_offs2[None, :]
        b_ptrs_21 = B_ptr + k_offs2[:, None] * A_stride_m + col_offs1[None, :]
        b_ptrs_22 = B_ptr + k_offs2[:, None] * A_stride_m + col_offs2[None, :]

        B_11 = tl.load(b_ptrs_11, mask=(k_offs1[:, None] < K) & (col_offs1[None, :] < N), other=0.)
        B_12 = tl.load(b_ptrs_12, mask=(k_offs1[:, None] < K) & (col_offs2[None, :] < N), other=0.)
        B_21 = tl.load(b_ptrs_21, mask=(k_offs2[:, None] < K) & (col_offs1[None, :] < N), other=0.)
        B_22 = tl.load(b_ptrs_22, mask=(k_offs2[:, None] < K) & (col_offs2[None, :] < N), other=0.)

        # accumulate dot products for each quarter
        M1 = tl.dot(A_11 + A_22, B_11 + B_22)
        M2 = tl.dot(A_21 + A_22, B_11)
        M3 = tl.dot(A_11, B_12 - B_22)
        M4 = tl.dot(A_22, B_21 - B_11)
        M5 = tl.dot(A_11 + A_12, B_22)
        M6 = tl.dot(A_21 - A_11, B_11 + B_12)
        M7 = tl.dot(A_12 - A_22, B_21 + B_22)

        acc_11 += M1 + M4 - M5 + M7
        acc_12 += M3 + M5
        acc_21 += M2 + M4
        acc_22 += M1 - M2 + M3 + M6

    c_ptrs_11 = C_ptr + row_offs1[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_12 = C_ptr + row_offs1[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_21 = C_ptr + row_offs2[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_22 = C_ptr + row_offs2[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k

    C_11 = tl.load(c_ptrs_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_12 = tl.load(c_ptrs_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_21 = tl.load(c_ptrs_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_22 = tl.load(c_ptrs_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N), other=0.)

    tl.store(c_ptrs_11, acc_11 + C_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_12, acc_12 + C_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_21, acc_21 + C_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_22, acc_22 + C_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N))


@triton.jit
def matmul_kernel_fp32_accum(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        A_stride_m, A_stride_k,
        B_stride_k, B_stride_n,
        BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        k_offs = k + tl.arange(0, BLOCK_SIZE)

        a_ptrs = A_ptr + off_m[:, None] * A_stride_m + k_offs[None, :] * A_stride_k
        a_mask = (off_m[:, None] < M) & (k_offs[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * B_stride_k + off_n[None, :] * B_stride_n
        b_mask = (k_offs[:, None] < K) & (off_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_block, b_block)

    c_ptrs = C_ptr + off_m[:, None] * A_stride_m + off_n[None, :] * A_stride_k
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    c_current = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(c_ptrs, acc + c_current, mask=c_mask)


def run_strassen_fp32_accum(A, B, C, BLOCK_SIZE=64):
    M, N = C.shape
    K = A.shape[1]
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    strassen_kernel_fp32_accum[grid](A, B, C, M, N, K,
                                   A.stride(0), A.stride(1),
                                   BLOCK_SIZE, BLOCK_SIZE // 2)


def run_matmul_fp32_accum(A, B, C, BLOCK_SIZE=64):
    M, N = C.shape
    K = A.shape[1]
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N
    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    matmul_kernel_fp32_accum[grid](A, B, C, M, N, K,
                                   A.stride(0), A.stride(1),
                                   B.stride(0), B.stride(1),
                                   BLOCK_SIZE)

def strassens_test() -> None:
    torch.manual_seed(3331)

    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a).cuda()

    gt_mm = a @ b

    print("gt mat mul")
    print(gt_mm)
    print("strassen's mat mul")
    run_strassen_fp32_accum(a, b, c)
    print("A matrix")
    print(a)
    print("B matrix")
    print(b)
    print("C matrix")
    print(c)
    assert torch.allclose(gt_mm, c, atol=1e-6), "did not match gt"

def matmul_test() -> None:
    torch.manual_seed(3331)

    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a).cuda()

    gt_mm = a @ b

    print("gt mat mul")
    print(gt_mm)
    print("triton mat mul")
    run_matmul_fp32_accum(a, b, c)
    print("A matrix")
    print(a)
    print("B matrix")
    print(b)
    print("C matrix")
    print(c)
    assert torch.allclose(a @ b, c, atol=1e-6), "did not match gt"

if __name__ == "__main__":
    strassens_test()