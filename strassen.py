import triton
import triton.language as tl
import torch




###################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_fp32_accum(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        A_stride_b, A_stride_m, A_stride_k,
        B_stride_b, B_stride_k, B_stride_n,
        BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    off_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        k_offs = k + tl.arange(0, BLOCK_SIZE)

        a_ptrs = A_ptr + A_stride_b * pid_b + off_m[:, None] * A_stride_m + k_offs[None, :]
        a_mask = (off_m[:, None] < M) & (k_offs[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + B_stride_b * pid_b + k_offs[:, None] * B_stride_k + off_n[None, :]
        b_mask = (k_offs[:, None] < K) & (off_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_block, b_block)

    c_ptrs = C_ptr + A_stride_b * pid_b + off_m[:, None] * A_stride_m + off_n[None, :] * A_stride_k
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    c_current = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(c_ptrs, acc + c_current, mask=c_mask)

def run_matmul_fp32_accum(A, B, C, BLOCK_SIZE=32):
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, _, N = B.shape
        grid = lambda meta: (batch_size, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

        matmul_kernel_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2))
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = lambda meta: (1, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    matmul_kernel_fp32_accum[grid](
        A, B, C,
        M, N, K,
        A.stride(0) * A.stride(1), A.stride(0), A.stride(1),
        B.stride(0) * B.stride(1), B.stride(0), B.stride(1))



###################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def strassen_kernel_fp32_accum(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        A_stride_b, A_stride_m, A_stride_k,
        BLOCK_SIZE: tl.constexpr):
    HALF_BLOCK: tl.constexpr = BLOCK_SIZE // 2
    # Regular grid handling like matmul
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

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

        # Load in FP16 for better memory bandwidth
        a_ptrs_11 = A_ptr + pid_b * A_stride_b + row_offs1[:, None] * A_stride_m + k_offs1[None, :]
        a_ptrs_12 = A_ptr + pid_b * A_stride_b + row_offs1[:, None] * A_stride_m + k_offs2[None, :]
        a_ptrs_21 = A_ptr + pid_b * A_stride_b + row_offs2[:, None] * A_stride_m + k_offs1[None, :]
        a_ptrs_22 = A_ptr + pid_b * A_stride_b + row_offs2[:, None] * A_stride_m + k_offs2[None, :]

        A_11 = tl.load(a_ptrs_11, mask=(row_offs1[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_12 = tl.load(a_ptrs_12, mask=(row_offs1[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)
        A_21 = tl.load(a_ptrs_21, mask=(row_offs2[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_22 = tl.load(a_ptrs_22, mask=(row_offs2[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)

        b_ptrs_11 = B_ptr + pid_b * A_stride_b + k_offs1[:, None] * A_stride_m + col_offs1[None, :]
        b_ptrs_12 = B_ptr + pid_b * A_stride_b + k_offs1[:, None] * A_stride_m + col_offs2[None, :]
        b_ptrs_21 = B_ptr + pid_b * A_stride_b + k_offs2[:, None] * A_stride_m + col_offs1[None, :]
        b_ptrs_22 = B_ptr + pid_b * A_stride_b + k_offs2[:, None] * A_stride_m + col_offs2[None, :]

        B_11 = tl.load(b_ptrs_11, mask=(k_offs1[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_12 = tl.load(b_ptrs_12, mask=(k_offs1[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)
        B_21 = tl.load(b_ptrs_21, mask=(k_offs2[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_22 = tl.load(b_ptrs_22, mask=(k_offs2[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)

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

    # Store results
    c_ptrs_11 = C_ptr + pid_b * A_stride_b + row_offs1[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_12 = C_ptr + pid_b * A_stride_b + row_offs1[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_21 = C_ptr + pid_b * A_stride_b + row_offs2[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_22 = C_ptr + pid_b * A_stride_b + row_offs2[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k

    C_11 = tl.load(c_ptrs_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_12 = tl.load(c_ptrs_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_21 = tl.load(c_ptrs_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_22 = tl.load(c_ptrs_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N), other=0.)

    tl.store(c_ptrs_11, acc_11 + C_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_12, acc_12 + C_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_21, acc_21 + C_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_22, acc_22 + C_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N))


def run_strassen(A, B, C, BLOCK_SIZE=32):
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, _, N = B.shape
        grid = lambda meta: (batch_size, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"])) # this might have been the reason for a block size of 32 being used
        strassen_kernel_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2))
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = lambda meta: (1, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    strassen_kernel_fp32_accum[grid](
        A, B, C, M, N, K,
        A.stride(0) * A.stride(1), A.stride(0), A.stride(1))


###################################
# can't use block size of <= 32
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def strassen2_fp32_accum(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        A_stride_b, A_stride_m, A_stride_k,
        BLOCK_SIZE: tl.constexpr):
    QUARTER_BLOCK: tl.constexpr = BLOCK_SIZE // 4

    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    base_m = pid_m * BLOCK_SIZE
    base_n = pid_n * BLOCK_SIZE

    # init 16 accumulators
    C_11 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_12 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_13 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_14 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)

    C_21 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_22 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_23 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_24 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)

    C_31 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_32 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_33 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_34 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)

    C_41 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_42 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_43 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    C_44 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)

    # quarter offsets
    row_offs1 = base_m + tl.arange(0, QUARTER_BLOCK)
    row_offs2 = base_m + QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
    row_offs3 = base_m + 2 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
    row_offs4 = base_m + 3 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)

    col_offs1 = base_n + tl.arange(0, QUARTER_BLOCK)
    col_offs2 = base_n + QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
    col_offs3 = base_n + 2 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
    col_offs4 = base_n + 3 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)

    for k in range(0, K, BLOCK_SIZE):
        k_offs1 = k + tl.arange(0, QUARTER_BLOCK)
        k_offs2 = k + QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
        k_offs3 = k + 2 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)
        k_offs4 = k + 3 * QUARTER_BLOCK + tl.arange(0, QUARTER_BLOCK)

        a_ptrs_11 = A_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + k_offs1[None, :] * A_stride_k
        a_ptrs_12 = A_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + k_offs2[None, :] * A_stride_k
        a_ptrs_13 = A_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + k_offs3[None, :] * A_stride_k
        a_ptrs_14 = A_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + k_offs4[None, :] * A_stride_k

        a_ptrs_21 = A_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + k_offs1[None, :] * A_stride_k
        a_ptrs_22 = A_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + k_offs2[None, :] * A_stride_k
        a_ptrs_23 = A_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + k_offs3[None, :] * A_stride_k
        a_ptrs_24 = A_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + k_offs4[None, :] * A_stride_k

        a_ptrs_31 = A_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + k_offs1[None, :] * A_stride_k
        a_ptrs_32 = A_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + k_offs2[None, :] * A_stride_k
        a_ptrs_33 = A_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + k_offs3[None, :] * A_stride_k
        a_ptrs_34 = A_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + k_offs4[None, :] * A_stride_k

        a_ptrs_41 = A_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + k_offs1[None, :] * A_stride_k
        a_ptrs_42 = A_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + k_offs2[None, :] * A_stride_k
        a_ptrs_43 = A_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + k_offs3[None, :] * A_stride_k
        a_ptrs_44 = A_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + k_offs4[None, :] * A_stride_k

        # load A blocks
        A_11 = tl.load(a_ptrs_11, mask=(row_offs1[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_12 = tl.load(a_ptrs_12, mask=(row_offs1[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)
        A_13 = tl.load(a_ptrs_13, mask=(row_offs1[:, None] < M) & (k_offs3[None, :] < K), other=0.).to(tl.float16)
        A_14 = tl.load(a_ptrs_14, mask=(row_offs1[:, None] < M) & (k_offs4[None, :] < K), other=0.).to(tl.float16)

        A_21 = tl.load(a_ptrs_21, mask=(row_offs2[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_22 = tl.load(a_ptrs_22, mask=(row_offs2[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)
        A_23 = tl.load(a_ptrs_23, mask=(row_offs2[:, None] < M) & (k_offs3[None, :] < K), other=0.).to(tl.float16)
        A_24 = tl.load(a_ptrs_24, mask=(row_offs2[:, None] < M) & (k_offs4[None, :] < K), other=0.).to(tl.float16)

        A_31 = tl.load(a_ptrs_31, mask=(row_offs3[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_32 = tl.load(a_ptrs_32, mask=(row_offs3[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)
        A_33 = tl.load(a_ptrs_33, mask=(row_offs3[:, None] < M) & (k_offs3[None, :] < K), other=0.).to(tl.float16)
        A_34 = tl.load(a_ptrs_34, mask=(row_offs3[:, None] < M) & (k_offs4[None, :] < K), other=0.).to(tl.float16)

        A_41 = tl.load(a_ptrs_41, mask=(row_offs4[:, None] < M) & (k_offs1[None, :] < K), other=0.).to(tl.float16)
        A_42 = tl.load(a_ptrs_42, mask=(row_offs4[:, None] < M) & (k_offs2[None, :] < K), other=0.).to(tl.float16)
        A_43 = tl.load(a_ptrs_43, mask=(row_offs4[:, None] < M) & (k_offs3[None, :] < K), other=0.).to(tl.float16)
        A_44 = tl.load(a_ptrs_44, mask=(row_offs4[:, None] < M) & (k_offs4[None, :] < K), other=0.).to(tl.float16)

        b_ptrs_11 = B_ptr + A_stride_b * pid_b + k_offs1[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
        b_ptrs_12 = B_ptr + A_stride_b * pid_b + k_offs1[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
        b_ptrs_13 = B_ptr + A_stride_b * pid_b + k_offs1[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
        b_ptrs_14 = B_ptr + A_stride_b * pid_b + k_offs1[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

        b_ptrs_21 = B_ptr + A_stride_b * pid_b + k_offs2[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
        b_ptrs_22 = B_ptr + A_stride_b * pid_b + k_offs2[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
        b_ptrs_23 = B_ptr + A_stride_b * pid_b + k_offs2[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
        b_ptrs_24 = B_ptr + A_stride_b * pid_b + k_offs2[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

        b_ptrs_31 = B_ptr + A_stride_b * pid_b + k_offs3[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
        b_ptrs_32 = B_ptr + A_stride_b * pid_b + k_offs3[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
        b_ptrs_33 = B_ptr + A_stride_b * pid_b + k_offs3[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
        b_ptrs_34 = B_ptr + A_stride_b * pid_b + k_offs3[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

        b_ptrs_41 = B_ptr + A_stride_b * pid_b + k_offs4[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
        b_ptrs_42 = B_ptr + A_stride_b * pid_b + k_offs4[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
        b_ptrs_43 = B_ptr + A_stride_b * pid_b + k_offs4[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
        b_ptrs_44 = B_ptr + A_stride_b * pid_b + k_offs4[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

        # load B blocks
        B_11 = tl.load(b_ptrs_11, mask=(k_offs1[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_12 = tl.load(b_ptrs_12, mask=(k_offs1[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)
        B_13 = tl.load(b_ptrs_13, mask=(k_offs1[:, None] < K) & (col_offs3[None, :] < N), other=0.).to(tl.float16)
        B_14 = tl.load(b_ptrs_14, mask=(k_offs1[:, None] < K) & (col_offs4[None, :] < N), other=0.).to(tl.float16)

        B_21 = tl.load(b_ptrs_21, mask=(k_offs2[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_22 = tl.load(b_ptrs_22, mask=(k_offs2[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)
        B_23 = tl.load(b_ptrs_23, mask=(k_offs2[:, None] < K) & (col_offs3[None, :] < N), other=0.).to(tl.float16)
        B_24 = tl.load(b_ptrs_24, mask=(k_offs2[:, None] < K) & (col_offs4[None, :] < N), other=0.).to(tl.float16)

        B_31 = tl.load(b_ptrs_31, mask=(k_offs3[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_32 = tl.load(b_ptrs_32, mask=(k_offs3[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)
        B_33 = tl.load(b_ptrs_33, mask=(k_offs3[:, None] < K) & (col_offs3[None, :] < N), other=0.).to(tl.float16)
        B_34 = tl.load(b_ptrs_34, mask=(k_offs3[:, None] < K) & (col_offs4[None, :] < N), other=0.).to(tl.float16)

        B_41 = tl.load(b_ptrs_41, mask=(k_offs4[:, None] < K) & (col_offs1[None, :] < N), other=0.).to(tl.float16)
        B_42 = tl.load(b_ptrs_42, mask=(k_offs4[:, None] < K) & (col_offs2[None, :] < N), other=0.).to(tl.float16)
        B_43 = tl.load(b_ptrs_43, mask=(k_offs4[:, None] < K) & (col_offs3[None, :] < N), other=0.).to(tl.float16)
        B_44 = tl.load(b_ptrs_44, mask=(k_offs4[:, None] < K) & (col_offs4[None, :] < N), other=0.).to(tl.float16)


        # TODO we will sub the a_11_22_* into the dots
        ##########################################
        #            global M1 comp              #
        ##########################################
        a_11_22_1 = A_11 + A_41
        a_11_22_2 = A_12 + A_42
        a_11_22_3 = A_13 + A_43
        a_11_22_4 = A_14 + A_44

        b_11_22_1 = B_11 + B_41
        b_11_22_2 = B_12 + B_42
        b_11_22_3 = B_13 + B_43
        b_11_22_4 = B_14 + B_44

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        M1_11 = M1 + M4 - M5 + M7
        M1_12 = M3 + M5
        M1_21 = M2 + M4
        M1_22 = M1 - M2 + M3 + M6

        C_11 += M1_11
        C_12 += M1_12
        C_13 += M1_21
        C_14 += M1_22

        C_41 += M1_11
        C_42 += M1_12
        C_43 += M1_21
        C_44 += M1_22


        ##########################################
        #            global M2 comp              #
        ##########################################
        a_11_22_1 = A_31 + A_41
        a_11_22_2 = A_32 + A_42
        a_11_22_3 = A_33 + A_43
        a_11_22_4 = A_34 + A_44

        # b_11_22_1 = B_11
        # b_11_22_2 = B_12
        # b_11_22_3 = B_13
        # b_11_22_4 = B_14

        M1 = tl.dot(a_11_22_1 + a_11_22_4, B_11 + B_14)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, B_11)
        M3 = tl.dot(a_11_22_1, B_12 - B_14)
        M4 = tl.dot(a_11_22_4, B_13 - B_11)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, B_14)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, B_11 + B_12)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, B_13 + B_14)

        M2_11 = M1 + M4 - M5 + M7
        M2_12 = M3 + M5
        M2_21 = M2 + M4
        M2_22 = M1 - M2 + M3 + M6

        C_31 += M2_11
        C_32 += M2_12
        C_33 += M2_21
        C_34 += M2_22

        C_41 -= M2_11
        C_42 -= M2_12
        C_43 -= M2_21
        C_44 -= M2_22


        ##########################################
        #            global M3 comp              #
        ##########################################
        # TODO do not store variables, replace the def in the dots
        a_11_22_1 = A_11
        a_11_22_2 = A_12
        a_11_22_3 = A_13
        a_11_22_4 = A_14

        b_11_22_1 = B_21 - B_41
        b_11_22_2 = B_22 - B_42
        b_11_22_3 = B_23 - B_43
        b_11_22_4 = B_24 - B_44

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        M3_11 = M1 + M4 - M5 + M7
        M3_12 = M3 + M5
        M3_21 = M2 + M4
        M3_22 = M1 - M2 + M3 + M6

        C_21 += M3_11
        C_22 += M3_12
        C_23 += M3_21
        C_24 += M3_22

        C_41 += M3_11
        C_42 += M3_12
        C_43 += M3_21
        C_44 += M3_22


        ##########################################
        #            global M4 comp              #
        ##########################################
        a_11_22_1 = A_41
        a_11_22_2 = A_42
        a_11_22_3 = A_43
        a_11_22_4 = A_44

        b_11_22_1 = B_31 - B_11
        b_11_22_2 = B_32 - B_12
        b_11_22_3 = B_33 - B_13
        b_11_22_4 = B_34 - B_14

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        M4_11 = M1 + M4 - M5 + M7
        M4_12 = M3 + M5
        M4_21 = M2 + M4
        M4_22 = M1 - M2 + M3 + M6

        C_11 += M4_11
        C_12 += M4_12
        C_13 += M4_21
        C_14 += M4_22

        C_31 += M4_11
        C_32 += M4_12
        C_33 += M4_21
        C_34 += M4_22


        ##########################################
        #            global M5 comp              #
        ##########################################
        a_11_22_1 = A_11 + A_21
        a_11_22_2 = A_12 + A_22
        a_11_22_3 = A_13 + A_23
        a_11_22_4 = A_14 + A_24

        b_11_22_1 = B_41
        b_11_22_2 = B_42
        b_11_22_3 = B_43
        b_11_22_4 = B_44

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        M5_11 = M1 + M4 - M5 + M7
        M5_12 = M3 + M5
        M5_21 = M2 + M4
        M5_22 = M1 - M2 + M3 + M6

        C_11 -= M5_11
        C_12 -= M5_12
        C_13 -= M5_21
        C_14 -= M5_22

        C_21 += M5_11
        C_22 += M5_12
        C_23 += M5_21
        C_24 += M5_22


        ##########################################
        #            global M6 comp              #
        ##########################################
        a_11_22_1 = A_31 - A_11
        a_11_22_2 = A_32 - A_12
        a_11_22_3 = A_33 - A_13
        a_11_22_4 = A_34 - A_14

        b_11_22_1 = B_11 + B_21
        b_11_22_2 = B_12 + B_22
        b_11_22_3 = B_13 + B_23
        b_11_22_4 = B_14 + B_24

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        C_41 += M1 + M4 - M5 + M7
        C_42 += M3 + M5
        C_43 += M2 + M4
        C_44 += M1 - M2 + M3 + M6


        ##########################################
        #            global M7 comp              #
        ##########################################
        a_11_22_1 = A_21 - A_41
        a_11_22_2 = A_22 - A_42
        a_11_22_3 = A_23 - A_43
        a_11_22_4 = A_24 - A_44

        b_11_22_1 = B_31 + B_41
        b_11_22_2 = B_32 + B_42
        b_11_22_3 = B_33 + B_43
        b_11_22_4 = B_34 + B_44

        M1 = tl.dot(a_11_22_1 + a_11_22_4, b_11_22_1 + b_11_22_4)
        M2 = tl.dot(a_11_22_3 + a_11_22_4, b_11_22_1)
        M3 = tl.dot(a_11_22_1, b_11_22_2 - b_11_22_4)
        M4 = tl.dot(a_11_22_4, b_11_22_3 - b_11_22_1)
        M5 = tl.dot(a_11_22_1 + a_11_22_2, b_11_22_4)
        M6 = tl.dot(a_11_22_3 - a_11_22_1, b_11_22_1 + b_11_22_2)
        M7 = tl.dot(a_11_22_2 - a_11_22_4, b_11_22_3 + b_11_22_4)

        C_11 += M1 + M4 - M5 + M7
        C_12 += M3 + M5
        C_13 += M2 + M4
        C_14 += M1 - M2 + M3 + M6

    c_ptrs_11 = C_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_12 = C_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_13 = C_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
    c_ptrs_14 = C_ptr + A_stride_b * pid_b + row_offs1[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

    c_ptrs_21 = C_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_22 = C_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_23 = C_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
    c_ptrs_24 = C_ptr + A_stride_b * pid_b + row_offs2[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

    c_ptrs_31 = C_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_32 = C_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_33 = C_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
    c_ptrs_34 = C_ptr + A_stride_b * pid_b + row_offs3[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

    c_ptrs_41 = C_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + col_offs1[None, :] * A_stride_k
    c_ptrs_42 = C_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + col_offs2[None, :] * A_stride_k
    c_ptrs_43 = C_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + col_offs3[None, :] * A_stride_k
    c_ptrs_44 = C_ptr + A_stride_b * pid_b + row_offs4[:, None] * A_stride_m + col_offs4[None, :] * A_stride_k

    # load C blocks
    C_11_p = tl.load(c_ptrs_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_12_p = tl.load(c_ptrs_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_13_p = tl.load(c_ptrs_13, mask=(row_offs1[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_14_p = tl.load(c_ptrs_14, mask=(row_offs1[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_21_p = tl.load(c_ptrs_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_22_p = tl.load(c_ptrs_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_23_p = tl.load(c_ptrs_23, mask=(row_offs2[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_24_p = tl.load(c_ptrs_24, mask=(row_offs2[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_31_p = tl.load(c_ptrs_31, mask=(row_offs3[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_32_p = tl.load(c_ptrs_32, mask=(row_offs3[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_33_p = tl.load(c_ptrs_33, mask=(row_offs3[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_34_p = tl.load(c_ptrs_34, mask=(row_offs3[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_41_p = tl.load(c_ptrs_41, mask=(row_offs4[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_42_p = tl.load(c_ptrs_42, mask=(row_offs4[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_43_p = tl.load(c_ptrs_43, mask=(row_offs4[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_44_p = tl.load(c_ptrs_44, mask=(row_offs4[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    tl.store(c_ptrs_11, C_11_p + C_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_12, C_12_p + C_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_13, C_13_p + C_13, mask=(row_offs1[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_14, C_14_p + C_14, mask=(row_offs1[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_21, C_21_p + C_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_22, C_22_p + C_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_23, C_23_p + C_23, mask=(row_offs2[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_24, C_24_p + C_24, mask=(row_offs2[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_31, C_31_p + C_31, mask=(row_offs3[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_32, C_32_p + C_32, mask=(row_offs3[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_33, C_33_p + C_33, mask=(row_offs3[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_34, C_34_p + C_34, mask=(row_offs3[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_41, C_41_p + C_41, mask=(row_offs4[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_42, C_42_p + C_42, mask=(row_offs4[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_43, C_43_p + C_43, mask=(row_offs4[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_44, C_44_p + C_44, mask=(row_offs4[:, None] < M) & (col_offs4[None, :] < N))

def run_strassen2(A, B, C, BLOCK_SIZE=64):
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, _, N = B.shape

        grid = lambda meta: (batch_size, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
        strassen2_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2))
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = lambda meta: (1, triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    strassen2_fp32_accum[grid](
        A, B, C, M, N, K,
        A.stride(0) * A.stride(1), A.stride(0), A.stride(1))


if __name__ == "__main__":
    # >> comment out autotuning if you want to test the kernels <<
    torch.manual_seed(1234)
    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a)
    run_strassen(a, b, c)
    print('matmul')
    print(c)
    print()
    print(a @ b)
