import triton
import triton.language as tl
import torch

def create_T_n0_matrix(n0: int) -> torch.Tensor:
    """
    Create the transformation matrix T_n0 ∈ M((n0+2)^2 × n0^2).
    This matrix represents the periodic boundary transformation.
    """
    n_out = (n0 + 2) ** 2
    n_in = n0 ** 2
    T = torch.zeros((n_out, n_in))
    
    # Fill the transformation matrix
    for i in range(n0 + 2):
        for j in range(n0 + 2):
            out_idx = i * (n0 + 2) + j
            
            # Map to input indices with periodic boundary conditions
            i_in = (i - 1) % n0 if i != 0 and i != n0 + 1 else (n0 - 1 if i == 0 else 0)
            j_in = (j - 1) % n0 if j != 0 and j != n0 + 1 else (n0 - 1 if j == 0 else 0)
            in_idx = i_in * n0 + j_in
            
            T[out_idx, in_idx] = 1.0
            
    return T

def strassen_2x2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Implement Strassen's algorithm for 2x2 matrices.
    
    Args:
        A: 2x2 tensor
        B: 2x2 tensor
    Returns:
        2x2 tensor result of A*B using 7 multiplications
    """
    # Seven products according to Strassen's algorithm
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1] - B[1,1])
    M4 = A[1,1] * (B[1,0] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])

    # Compute the entries of the result matrix
    C = torch.zeros((2, 2), device=A.device, dtype=A.dtype)
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6

    return C

def aggregate_table_2a(A_bar: torch.Tensor, B_bar: torch.Tensor, C_bar: torch.Tensor, 
                      i: int, j: int, k: int) -> torch.Tensor:
    """
    Compute aggregate for triplet (i,j,k) according to Table 2a.
    
    Args:
        A_bar, B_bar, C_bar: (n0+2)x(n0+2) tensors
        i, j, k: indices for the triplet
    Returns:
        Updated C_bar tensor with the aggregate contribution
    """
    n0 = A_bar.shape[0] - 2
    i_bar = n0 + 2 - i
    j_bar = n0 + 2 - j
    k_bar = n0 + 2 - k
    
    # Coefficient for diagonal terms
    coef = 1.0/3.0 if i == j == k else 1.0
    
    # Compute the aggregate according to Table 2a
    result = torch.zeros_like(C_bar)
    result[k,i] += coef * (A_bar[i,j] * B_bar[j,k])
    result[k_bar,i_bar] += coef * (A_bar[i_bar,j_bar] * B_bar[j_bar,k_bar])
    result[j,i] += coef * (A_bar[i,k] * B_bar[i,k])
    
    return result

def aggregate_table_2b(A_bar: torch.Tensor, B_bar: torch.Tensor, C_bar: torch.Tensor,
                      i: int, j: int, k: int) -> torch.Tensor:
    """
    Compute aggregate for triplet (i,j,k) according to Table 2b.
    
    Args:
        A_bar, B_bar, C_bar: (n0+2)x(n0+2) tensors
        i, j, k: indices for the triplet
    Returns:
        Updated C_bar tensor with the aggregate contribution
    """
    n0 = A_bar.shape[0] - 2
    i_bar = n0 + 2 - i
    j_bar = n0 + 2 - j
    k_bar = n0 + 2 - k
    
    # Coefficient for diagonal terms
    coef = 1.0/3.0 if i == j == k else 1.0
    
    # Compute the aggregate according to Table 2b
    result = torch.zeros_like(C_bar)
    result[k,j] += coef * (A_bar[j,i] * B_bar[i,k])
    result[k_bar,j_bar] += coef * (A_bar[j_bar,i_bar] * B_bar[i_bar,k_bar])
    result[k,i] += coef * (A_bar[j,k] * B_bar[j,k])
    
    return result

def compute_correction_terms(A_bar: torch.Tensor, B_bar: torch.Tensor, C_bar: torch.Tensor) -> torch.Tensor:
    """
    Compute correction terms from Expression (3) using Strassen's algorithm.
    
    Args:
        A_bar, B_bar, C_bar: (n0+2)x(n0+2) tensors
    Returns:
        Correction terms tensor
    """
    n0 = A_bar.shape[0] - 2
    result = torch.zeros_like(A_bar)
    
    # First sum: diagonal terms with coefficient 9
    for i in range(n0 + 2):
        result[i,i] += 9 * A_bar[i,i] * B_bar[i,i] * C_bar[i,i]
    
    # Second sum: terms from Expression (3)
    for i in range(n0//2 + 1):
        for j in range(n0//2 + 1):
            i_bar = n0 + 2 - i
            j_bar = n0 + 2 - j
            
            # Lambda coefficient for diagonal terms
            lambda_coef = 1.0 - (9.0 / (n0//2 + 1)) * (1.0 if i == j else 0.0)
            
            # Create 2x2 matrices for Strassen's algorithm
            A_2x2 = torch.stack([
                torch.stack([lambda_coef * A_bar[i,j], A_bar[i_bar,j]]),
                torch.stack([A_bar[i,j_bar], A_bar[i_bar,j_bar]])
            ])
            
            B_2x2 = torch.stack([
                torch.stack([B_bar[i,j], B_bar[i_bar,j]]),
                torch.stack([B_bar[i,j_bar], B_bar[i_bar,j_bar]])
            ])
            
            C_2x2 = torch.stack([
                torch.stack([C_bar[i,j], -C_bar[i,j_bar]]),
                torch.stack([-C_bar[i_bar,j], lambda_coef * C_bar[i_bar,j_bar]])
            ])
            
            # Apply Strassen's algorithm
            correction = strassen_2x2(A_2x2, B_2x2)
            
            # Update result with correction terms
            result[i:i+2, j:j+2] += correction
    
    return result

def pan82rev_bc(A_bar: torch.Tensor, B_bar: torch.Tensor) -> torch.Tensor:
    """
    Implement Pan82RevBC - the bilinear algorithm operating on transformed matrices.
    This is not a MM algorithm by itself but forms the core of Pan82Rev.
    
    Args:
        A_bar, B_bar: (n0+2)x(n0+2) transformed input tensors
    Returns:
        (n0+2)x(n0+2) result tensor
    """
    n0 = A_bar.shape[0] - 2
    result = torch.zeros_like(A_bar)
    
    # Step (b): Aggregate tables with exactly 3 non-zeros per row
    for i in range(n0//2 + 1):
        for j in range(n0//2 + 1):
            for k in range(n0//2 + 1):
                i_bar = n0 + 2 - i
                j_bar = n0 + 2 - j
                k_bar = n0 + 2 - k
                
                # Table 2a contribution (3 non-zeros)
                coef = 1.0/3.0 if i == j == k else 1.0
                result[k,i] += coef * (A_bar[i,j] * B_bar[j,k])
                result[k_bar,i_bar] += coef * (A_bar[i_bar,j_bar] * B_bar[j_bar,k_bar])
                result[j,i] += coef * (A_bar[i,k] * B_bar[i,k])
                
                # Table 2b contribution (3 non-zeros)
                result[k,j] += coef * (A_bar[j,i] * B_bar[i,k])
                result[k_bar,j_bar] += coef * (A_bar[j_bar,i_bar] * B_bar[i_bar,k_bar])
                result[k,i] += coef * (A_bar[j,k] * B_bar[j,k])
    
    # Step (c): Correction terms using Strassen's algorithm
    for i in range(n0//2 + 1):
        for j in range(n0//2 + 1):
            lambda_coef = 1.0 - (9.0 / (n0//2 + 1)) * (1.0 if i == j else 0.0)
            i_bar = n0 + 2 - i
            j_bar = n0 + 2 - j
            
            # Create 2x2 matrices with sparse structure
            A_2x2 = torch.stack([
                torch.stack([lambda_coef * A_bar[i,j], A_bar[i_bar,j]]),
                torch.stack([A_bar[i,j_bar], A_bar[i_bar,j_bar]])
            ])
            B_2x2 = torch.stack([
                torch.stack([B_bar[i,j], B_bar[i_bar,j]]),
                torch.stack([B_bar[i,j_bar], B_bar[i_bar,j_bar]])
            ])
            
            correction = strassen_2x2(A_2x2, B_2x2)
            result[i:i+2, j:j+2] += correction
    
    return result

def transform_matrix(A: torch.Tensor) -> torch.Tensor:
    """
    Transform input matrix A of size n0×n0 to matrix Ā of size (n0+2)×(n0+2)
    using periodic boundary conditions without creating the full T_n0 matrix.
    """
    n0 = A.shape[0]
    result = torch.zeros((n0 + 2, n0 + 2), device=A.device, dtype=A.dtype)
    
    # Copy original matrix to center
    result[1:-1, 1:-1] = A
    
    # Add periodic boundary conditions
    result[0, 1:-1] = A[-1, :]  # Top edge
    result[-1, 1:-1] = A[0, :]  # Bottom edge
    result[1:-1, 0] = A[:, -1]  # Left edge
    result[1:-1, -1] = A[:, 0]  # Right edge
    
    # Handle corners
    result[0, 0] = A[-1, -1]    # Top-left
    result[0, -1] = A[-1, 0]    # Top-right
    result[-1, 0] = A[0, -1]    # Bottom-left
    result[-1, -1] = A[0, 0]    # Bottom-right
    
    return result

def inverse_transform_matrix(A_bar: torch.Tensor) -> torch.Tensor:
    """
    Transform back from (n0+2)×(n0+2) to n0×n0 by taking the central portion.
    """
    return A_bar[1:-1, 1:-1]

def run_pan82rev_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BLOCK_SIZE: int = 32) -> None:
    """
    Run Pan82Rev algorithm using Triton kernels.
    Updates C in-place with the result of A @ B.
    Supports both batched and non-batched inputs.
    """
    if len(A.shape) == 3:  # Batched
        batch_size, M, K = A.shape
        _, K2, N = B.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"
        assert A.shape[0] == B.shape[0] == C.shape[0], "Batch sizes must match"
        assert C.shape[1:] == (M, N), f"Output shape mismatch: {C.shape[1:]} != {(M, N)}"
        assert M % 2 == 0, "Input dimension must be even"
        assert M >= 20, "Input dimension must be at least 20 for optimal performance"
        
        # Step 1: Transform input matrices directly
        A_bar = torch.zeros((batch_size, M + 2, K + 2), device=A.device, dtype=A.dtype)
        B_bar = torch.zeros((batch_size, K + 2, N + 2), device=B.device, dtype=B.dtype)
        C_bar = torch.zeros((batch_size, M + 2, N + 2), device=C.device, dtype=C.dtype)
        
        # Apply transformation for each batch
        for i in range(batch_size):
            A_bar[i] = transform_matrix(A[i])
            B_bar[i] = transform_matrix(B[i])
        
        # Step 2: Run Pan82RevBC computation in Triton
        grid = (batch_size, triton.cdiv(M + 2, BLOCK_SIZE), triton.cdiv(N + 2, BLOCK_SIZE))
        
        pan82rev_bc_kernel[grid](
            A_bar, B_bar, C_bar,
            M + 2, N + 2, K + 2,
            A_bar.stride(0), A_bar.stride(1), A_bar.stride(2),
            B_bar.stride(0), B_bar.stride(1), B_bar.stride(2),
            C_bar.stride(0), C_bar.stride(1), C_bar.stride(2),
            BLOCK_SIZE=BLOCK_SIZE)
        
        # Step 3: Transform back and store in C
        for i in range(batch_size):
            C[i].copy_(inverse_transform_matrix(C_bar[i]))
        
    else:  # Non-batched
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"
        assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != {(M, N)}"
        assert M % 2 == 0, "Input dimension must be even"
        assert M >= 20, "Input dimension must be at least 20 for optimal performance"
        
        # Step 1: Transform input matrices directly
        A_bar = transform_matrix(A)
        B_bar = transform_matrix(B)
        C_bar = torch.zeros((M + 2, N + 2), device=A.device, dtype=A.dtype)
        
        # Step 2: Run Pan82RevBC computation in Triton
        grid = (1, triton.cdiv(M + 2, BLOCK_SIZE), triton.cdiv(N + 2, BLOCK_SIZE))
        
        pan82rev_bc_kernel[grid](
            A_bar, B_bar, C_bar,
            M + 2, N + 2, K + 2,
            1, A_bar.stride(0), A_bar.stride(1),
            1, B_bar.stride(0), B_bar.stride(1),
            1, C_bar.stride(0), C_bar.stride(1),)
            # BLOCK_SIZE=BLOCK_SIZE)
        
        # Step 3: Transform back and store in C
        C.copy_(inverse_transform_matrix(C_bar))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 128}, num_warps=16),
        # triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def pan82rev_bc_kernel(
        A_bar_ptr, B_bar_ptr, C_bar_ptr,
        M, N, K,
        stride_ab, stride_am, stride_an,
        stride_bb, stride_bm, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel implementing Pan82RevBC - the core bilinear computation.
    Supports both batched and non-batched inputs.
    """
    # Get program IDs
    pid_b = tl.program_id(0)  # Batch ID
    pid_m = tl.program_id(1)  # Row ID
    pid_n = tl.program_id(2)  # Col ID
    
    # Offset pointers by batch stride
    A_bar_ptr += pid_b * stride_ab
    B_bar_ptr += pid_b * stride_bb
    C_bar_ptr += pid_b * stride_cb
    
    # Block starting positions
    base_m = pid_m * BLOCK_SIZE
    base_n = pid_n * BLOCK_SIZE
    
    # Initialize accumulator for this block
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Load offsets for this block
    offs_m = base_m + tl.arange(0, BLOCK_SIZE)
    offs_n = base_n + tl.arange(0, BLOCK_SIZE)
    
    # Process aggregation tables (Step b)
    for k in range(0, M, BLOCK_SIZE):
        k_offs = k + tl.arange(0, BLOCK_SIZE)
        
        # Load A blocks with proper broadcasting
        a_ptrs = A_bar_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_an
        b_ptrs = B_bar_ptr + k_offs[:, None] * stride_bm + offs_n[None, :] * stride_bn
        
        # Create masks for valid memory accesses
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        # Load data
        A = tl.load(a_ptrs, mask=a_mask, other=0.0)
        B = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute diagonal coefficient
        i_eq_j = offs_m[:, None] == offs_n[None, :]
        i_eq_k = offs_m[:, None] == k_offs[None, :]
        diag_coef = tl.where(i_eq_j & i_eq_k, 1.0/3.0, 1.0)
        
        # Aggregate Table 2a and 2b contributions
        acc += diag_coef * tl.dot(A, B)
    
    # Process correction terms (Step c)
    lambda_coef = 1.0 - (9.0 / (M//2 + 1))
    for i in range(0, BLOCK_SIZE, 2):
        for j in range(0, BLOCK_SIZE, 2):
            # Only process valid 2x2 blocks
            is_valid_block = (i + 1 < BLOCK_SIZE) & (j + 1 < BLOCK_SIZE)
            block_mask = (
                (offs_m[:, None] >= i) & (offs_m[:, None] < i+2) &
                (offs_n[None, :] >= j) & (offs_n[None, :] < j+2)
            )
            
            # Apply correction only for valid blocks
            correction_mask = block_mask & is_valid_block
            acc = tl.where(correction_mask,
                acc * lambda_coef,
                acc
            )
    
    # Store results
    c_ptrs = C_bar_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def main():
    # Test the implementation
    torch.manual_seed(0)
    n0 = 1024  # Must be >= 20 for optimal performance
    device = torch.device('cuda')
    
    # Create test matrices
    A = torch.randn(n0, n0, device=device)
    B = torch.randn(n0, n0, device=device)
    
    # Compute reference result using torch
    C_ref = torch.mm(A, B)
    
    # Compute result using our implementation
    C_triton = run_pan82rev_triton(A, B)
    
    # Verify results
    max_error = torch.max(torch.abs(C_ref - C_triton))
    print(f"Max error: {max_error:.2e}")
    assert max_error < 1e-3, "Results don't match!"
    print("Test passed!")

if __name__ == "__main__":
    main()