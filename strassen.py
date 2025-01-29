import triton
import triton.language as tl
import torch


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
#         triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def strassen_2_layer_fp32_accum(
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
    acc_11 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_12 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_13 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_14 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_21 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_22 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_23 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_24 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_31 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_32 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_33 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_34 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_41 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_42 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_43 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)
    acc_44 = tl.zeros((QUARTER_BLOCK, QUARTER_BLOCK), dtype=tl.float32)

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

        # C_11
        #####################################
        t1 = A_21 + A_22
        t2 = A_22 - A_12
        t3 = A_22 - A_11
        t4 = B_22 - B_11
        t5 = B_21 + B_22
        t6 = B_22 - B_12

        M1 = tl.dot(A_11, B_11)
        M2 = tl.dot(A_12, B_21)
        M3 = tl.dot(A_21, t4)
        M4 = tl.dot(A_22, B_22)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_12)

        acc_11 += M1 + M2
        acc_12 += M5 - M7
        acc_21 += M3 + M6
        acc_22 += M5 + M6 - M2 - M4

        t1 = A_23 + A_24
        t2 = A_24 - A_14
        t3 = A_24 - A_13
        t4 = B_42 - B_31
        t5 = B_41 + B_42
        t6 = B_42 - B_32

        M1 = tl.dot(A_13, B_31)
        M2 = tl.dot(A_14, B_41)
        M3 = tl.dot(A_23, t4)
        M4 = tl.dot(A_24, B_42)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_32)

        acc_11 += M1 + M2
        acc_12 += M5 - M7
        acc_21 += M3 + M6
        acc_22 += M5 + M6 - M2 - M4

        # C_12
        #####################################
        t1 = A_41 + A_42
        t2 = A_42 - A_32
        t3 = A_42 - A_31
        t4 = B_22 - B_11
        t5 = B_21 + B_22
        t6 = B_22 - B_12

        M1 = tl.dot(A_31, B_11)
        M2 = tl.dot(A_32, B_21)
        M3 = tl.dot(A_41, t4)
        M4 = tl.dot(A_42, B_22)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_12)

        acc_31 += M1 + M2
        acc_32 += M5 - M7
        acc_41 += M3 + M6
        acc_42 += M5 + M6 - M2 - M4

        t1 = A_43 + A_44
        t2 = A_44 - A_34
        t3 = A_44 - A_33
        t4 = B_42 - B_31
        t5 = B_41 + B_42
        t6 = B_42 - B_32

        M1 = tl.dot(A_33, B_31)
        M2 = tl.dot(A_34, B_41)
        M3 = tl.dot(A_43, t4)
        M4 = tl.dot(A_44, B_42)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_32)

        acc_33 += M1 + M2
        acc_34 += M5 - M7
        acc_43 += M3 + M6
        acc_44 += M5 + M6 - M2 - M4

        # C_21
        #####################################
        t1 = A_21 + A_22
        t2 = A_22 - A_12
        t3 = A_22 - A_11
        t4 = B_24 - B_13
        t5 = B_23 + B_24
        t6 = B_24 - B_14

        M1 = tl.dot(A_11, B_13)
        M2 = tl.dot(A_12, B_23)
        M3 = tl.dot(A_21, t4)
        M4 = tl.dot(A_22, B_24)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_14)

        acc_13 += M1 + M2
        acc_14 += M5 - M7
        acc_23 += M3 + M6
        acc_24 += M5 + M6 - M2 - M4

        t1 = A_23 + A_24
        t2 = A_24 - A_14
        t3 = A_24 - A_13
        t4 = B_44 - B_33
        t5 = B_43 + B_44
        t6 = B_44 - B_34

        M1 = tl.dot(A_13, B_33)
        M2 = tl.dot(A_14, B_43)
        M3 = tl.dot(A_23, t4)
        M4 = tl.dot(A_24, B_44)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_34)

        acc_13 += M1 + M2
        acc_14 += M5 - M7
        acc_23 += M3 + M6
        acc_24 += M5 + M6 - M2 - M4

        # C_22
        #####################################
        t1 = A_41 + A_42
        t2 = A_42 - A_32
        t3 = A_42 - A_31
        t4 = B_24 - B_13
        t5 = B_23 + B_24
        t6 = B_24 - B_14

        M1 = tl.dot(A_31, B_13)
        M2 = tl.dot(A_32, B_23)
        M3 = tl.dot(A_41, t4)
        M4 = tl.dot(A_42, B_24)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_14)

        acc_33 += M1 + M2
        acc_34 += M5 - M7
        acc_43 += M3 + M6
        acc_44 += M5 + M6 - M2 - M4

        t1 = A_43 + A_44
        t2 = A_44 - A_34
        t3 = A_44 - A_33
        t4 = B_44 - B_33
        t5 = B_43 + B_44
        t6 = B_44 - B_34

        M1 = tl.dot(A_33, B_33)
        M2 = tl.dot(A_34, B_43)
        M3 = tl.dot(A_43, t4)
        M4 = tl.dot(A_44, B_44)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_34)

        acc_33 += M1 + M2
        acc_34 += M5 - M7
        acc_43 += M3 + M6
        acc_44 += M5 + M6 - M2 - M4

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
    C_11 = tl.load(c_ptrs_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_12 = tl.load(c_ptrs_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_13 = tl.load(c_ptrs_13, mask=(row_offs1[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_14 = tl.load(c_ptrs_14, mask=(row_offs1[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_21 = tl.load(c_ptrs_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_22 = tl.load(c_ptrs_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_23 = tl.load(c_ptrs_23, mask=(row_offs2[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_24 = tl.load(c_ptrs_24, mask=(row_offs2[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_31 = tl.load(c_ptrs_31, mask=(row_offs3[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_32 = tl.load(c_ptrs_32, mask=(row_offs3[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_33 = tl.load(c_ptrs_33, mask=(row_offs3[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_34 = tl.load(c_ptrs_34, mask=(row_offs3[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    C_41 = tl.load(c_ptrs_41, mask=(row_offs4[:, None] < M) & (col_offs1[None, :] < N), other=0.)
    C_42 = tl.load(c_ptrs_42, mask=(row_offs4[:, None] < M) & (col_offs2[None, :] < N), other=0.)
    C_43 = tl.load(c_ptrs_43, mask=(row_offs4[:, None] < M) & (col_offs3[None, :] < N), other=0.)
    C_44 = tl.load(c_ptrs_44, mask=(row_offs4[:, None] < M) & (col_offs4[None, :] < N), other=0.)

    tl.store(c_ptrs_11, acc_11 + C_11, mask=(row_offs1[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_12, acc_12 + C_12, mask=(row_offs1[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_13, acc_13 + C_13, mask=(row_offs1[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_14, acc_14 + C_14, mask=(row_offs1[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_21, acc_21 + C_21, mask=(row_offs2[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_22, acc_22 + C_22, mask=(row_offs2[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_23, acc_23 + C_23, mask=(row_offs2[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_24, acc_24 + C_24, mask=(row_offs2[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_31, acc_31 + C_31, mask=(row_offs3[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_32, acc_32 + C_32, mask=(row_offs3[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_33, acc_33 + C_33, mask=(row_offs3[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_34, acc_34 + C_34, mask=(row_offs3[:, None] < M) & (col_offs4[None, :] < N))

    tl.store(c_ptrs_41, acc_41 + C_41, mask=(row_offs4[:, None] < M) & (col_offs1[None, :] < N))
    tl.store(c_ptrs_42, acc_42 + C_42, mask=(row_offs4[:, None] < M) & (col_offs2[None, :] < N))
    tl.store(c_ptrs_43, acc_43 + C_43, mask=(row_offs4[:, None] < M) & (col_offs3[None, :] < N))
    tl.store(c_ptrs_44, acc_44 + C_44, mask=(row_offs4[:, None] < M) & (col_offs4[None, :] < N))

def run_strassen_2_layer_fp32_accum(A, B, C, BLOCK_SIZE=64):
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, _, N = B.shape
        grid = (batch_size, triton.cdiv(M, BLOCK_SIZE),
                triton.cdiv(N, BLOCK_SIZE))
        strassen_2_layer_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2), BLOCK_SIZE)
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = (1, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    strassen_2_layer_fp32_accum[grid](A, B, C, M, N, K,
                                      1, A.stride(0), A.stride(1),
                                      BLOCK_SIZE)


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

        a_ptrs = A_ptr + A_stride_b * pid_b + off_m[:, None] * A_stride_m + k_offs[None, :] * A_stride_k
        a_mask = (off_m[:, None] < M) & (k_offs[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + B_stride_b * pid_b + k_offs[:, None] * B_stride_k + off_n[None, :] * B_stride_n
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
        grid = (batch_size, triton.cdiv(M, BLOCK_SIZE),
                triton.cdiv(N, BLOCK_SIZE))

        matmul_kernel_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2), BLOCK_SIZE)
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = (1, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    matmul_kernel_fp32_accum[grid](
        A, B, C,
        M, N, K,
        1, A.stride(0), A.stride(1),
        1, B.stride(0), B.stride(1), BLOCK_SIZE)


# @triton.autotune(
#     configs=[
#         # triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
#         triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def winograd_strassen_kernel_fp32_accum(
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

        t1 = A_21 + A_22
        t2 = A_22 - A_12
        t3 = A_22 - A_11
        t4 = B_22 - B_11
        t5 = B_21 + B_22
        t6 = B_22 - B_12

        M1 = tl.dot(A_11, B_11)
        M2 = tl.dot(A_12, B_21)
        M3 = tl.dot(A_21, t4)
        M4 = tl.dot(A_22, B_22)
        M5 = tl.dot(t1, t5)
        M6 = tl.dot(t2, t6)
        M7 = tl.dot(t3, B_12)

        acc_11 += M1 + M2
        acc_12 += M5 - M7
        acc_21 += M3 + M6
        acc_22 += M5 + M6 - M2 - M4

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

def run_winograd_strassen(A, B, C, BLOCK_SIZE=32):
    if len(A.shape) == 3:
        batch_size, M, K = A.shape
        _, _, N = B.shape
        grid = (batch_size, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        winograd_strassen_kernel_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_SIZE)
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = (1, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    winograd_strassen_kernel_fp32_accum[grid](
        A, B, C, M, N, K,
        1, A.stride(0), A.stride(1),
        BLOCK_SIZE)


@triton.jit
def old_winograd_strassen_kernel_fp32_accum(
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


def run_old_winograd_strassen(A, B, C, BLOCK_SIZE=32):
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, _, N = B.shape
        # Use the same reshape strategy as matmul

        grid = (batch_size, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        old_winograd_strassen_kernel_fp32_accum[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1), A.stride(2),  # Matrix strides
            BLOCK_SIZE)
        return

    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0] and A.shape[0] == M and B.shape[1] == N

    grid = (1, triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    old_winograd_strassen_kernel_fp32_accum[grid](
        A, B, C, M, N, K,
        1, A.stride(0), A.stride(1),
        BLOCK_SIZE)


if __name__ == "__main__":
    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a).cuda()

    run_winograd_strassen(a, b, c, 32)
    print(c)
    print(a @ b)
