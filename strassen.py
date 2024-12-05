import triton
import triton.language as tl
import torch
import torch.nn as nn

@triton.jit
def strassen_kernel_fp32_accum(
                    A_ptr,
                    B_ptr,
                    C_ptr,
                    A_m, A_n,
                    A_stride_m,
                    A_stride_n,
                    BLOCK_SIZE: tl.constexpr,
                    HALF_BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    row_offs = (pid // (A_m // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, HALF_BLOCK)
    col_offs = (pid % (A_n // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, HALF_BLOCK)
    
    # define block pntrs
    a_ptrs_1 = A_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    a_ptrs_2 = A_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    a_ptrs_3 = A_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    a_ptrs_4 = A_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    b_ptrs_1 = B_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    b_ptrs_2 = B_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    b_ptrs_3 = B_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    b_ptrs_4 = B_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    c_ptrs_1 = C_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    c_ptrs_2 = C_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    c_ptrs_3 = C_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    c_ptrs_4 = C_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    # load matrix sub blocks
    A_11 = tl.load(a_ptrs_1, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    A_12 = tl.load(a_ptrs_2, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
    A_21 = tl.load(a_ptrs_3, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    A_22 = tl.load(a_ptrs_4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)

    B_11 = tl.load(b_ptrs_1, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    B_12 = tl.load(b_ptrs_2, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
    B_21 = tl.load(b_ptrs_3, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    B_22 = tl.load(b_ptrs_4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
   
    # compute value blocks
    M1 = tl.dot(A_11 + A_22, B_11 + B_22)
    M2 = tl.dot(A_21 + A_22, B_11)
    M3 = tl.dot(A_11, B_12 - B_22)
    M4 = tl.dot(A_22, B_21 - B_11)
    M5 = tl.dot(A_11 + A_12, B_22)
    M6 = tl.dot(A_21 - A_11, B_11 + B_12)
    M7 = tl.dot(A_12 - A_22, B_21 + B_22)

    tl.store(c_ptrs_1, M1 + M4 - M5 + M7, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n))
    tl.store(c_ptrs_2, M3 + M5, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n))
    tl.store(c_ptrs_3, M2 + M4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n))
    tl.store(c_ptrs_4, M1 - M2 + M3 + M6, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n))

@triton.jit
def matmul_kernel_fp32_accum(
                    A_ptr,
                    B_ptr,
                    C_ptr,
                    A_m, A_n,
                    A_stride_m,
                    A_stride_n,
                    BLOCK_SIZE: tl.constexpr,
                    HALF_BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    row_offs = (pid // (A_m // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, HALF_BLOCK)
    col_offs = (pid % (A_n // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, HALF_BLOCK)
    
    # define block pntrs
    a_ptrs_1 = A_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    a_ptrs_2 = A_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    a_ptrs_3 = A_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    a_ptrs_4 = A_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    b_ptrs_1 = B_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    b_ptrs_2 = B_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    b_ptrs_3 = B_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    b_ptrs_4 = B_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    c_ptrs_1 = C_ptr + row_offs[:, None] * A_stride_m + col_offs[None, :]
    c_ptrs_2 = C_ptr + (row_offs[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])
    c_ptrs_3 = C_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + (col_offs[None, :])
    c_ptrs_4 = C_ptr + ((row_offs + HALF_BLOCK)[:, None]) * A_stride_m + ((col_offs + HALF_BLOCK)[None, :])

    # load matrix sub blocks
    A_11 = tl.load(a_ptrs_1, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    A_12 = tl.load(a_ptrs_2, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
    A_21 = tl.load(a_ptrs_3, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    A_22 = tl.load(a_ptrs_4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)

    B_11 = tl.load(b_ptrs_1, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    B_12 = tl.load(b_ptrs_2, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
    B_21 = tl.load(b_ptrs_3, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n), other = 0.)
    B_22 = tl.load(b_ptrs_4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n), other = 0.)
   
    M1 = tl.dot(A_11, B_11) + tl.dot(A_12, B_21)
    M2 = tl.dot(A_11, B_12) + tl.dot(A_12, B_22)
    M3 = tl.dot(A_21, B_11) + tl.dot(A_22, B_21)
    M4 = tl.dot(A_21, B_12) + tl.dot(A_22, B_22)

    tl.store(c_ptrs_1, M1, mask = (row_offs[:, None] < A_m) & (col_offs[None, :] < A_n))
    tl.store(c_ptrs_2, M2, mask = (row_offs[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n))
    tl.store(c_ptrs_3, M3, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & (col_offs[None, :] < A_n))
    tl.store(c_ptrs_4, M4, mask = ((row_offs + HALF_BLOCK)[:, None] < A_m) & ((col_offs + HALF_BLOCK)[None, :] < A_n))

    
# Launch strassen kernel
def run_strassen_fp32_accum(A, B, C, BLOCK_SIZE=64):
    m, n = A.shape
    *_, a_stride_m, a_stride_n = A.stride()
    assert n == A.shape[1] == B.shape[0] == B.shape[1] == C.shape[0] == C.shape[1]

    grid = ((m*n) // (BLOCK_SIZE * BLOCK_SIZE),)
    strassen_kernel_fp32_accum[grid](A, B, C, m, n, a_stride_m, a_stride_n, BLOCK_SIZE, BLOCK_SIZE // 2)

def run_matmul_fp32_accum(A, B, C, BLOCK_SIZE=64):
    m, n = A.shape
    *_, a_stride_m, a_stride_n = A.stride()
    assert n == A.shape[1] == B.shape[0] == B.shape[1] == C.shape[0] == C.shape[1]

    grid = ((m*n) // (BLOCK_SIZE * BLOCK_SIZE),)
    matmul_kernel_fp32_accum[grid](A, B, C, m, n, a_stride_m, a_stride_n, BLOCK_SIZE, BLOCK_SIZE // 2)

def strassens_test() -> None:
    torch.manual_seed(3331)

    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a).cuda()
    
    print("gt mat mul")
    print(a @ b)
    print("strassen's mat mul")
    run_strassen_fp32_accum(a, b, c)
    print(c)

    assert torch.allclose(a @ b, c, atol=1e-6), "did not match gt" 

def matmul_test() -> None:
    torch.manual_seed(3331)

    a = torch.randn((512, 512)).cuda()
    b = torch.randn((512, 512)).cuda()
    c = torch.zeros_like(a).cuda()
    
    print("gt mat mul")
    print(a @ b)
    print("mat mul")
    run_matmul_fp32_accum(a, b, c)
    print(c)

    assert torch.allclose(a @ b, c, atol=1e-6), "did not match gt" 

if __name__ == "__main__":
    strassens_test()
    matmul_test()
