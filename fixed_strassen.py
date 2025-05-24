import torch

# neva shoulda used triton :(
@torch.compile
def _strassen_base(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs one layer of Strassen's matrix multiplication.
    This will be the 'base case' for our two-layer unroll,
    using torch.mm for the N/4 x N/4 multiplications.
    """
    if A.shape[0] <= 16: # Or some other threshold where standard mm is faster
        return torch.mm(A, B)

    n = A.shape[0]
    mid = n // 2

    # Split A into quadrants
    A11 = A[..., :mid, :mid]
    A12 = A[..., :mid, mid:]
    A21 = A[..., mid:, :mid]
    A22 = A[..., mid:, mid:]

    # Split B into quadrants
    B11 = B[..., :mid, :mid]
    B12 = B[..., :mid, mid:]
    B21 = B[..., mid:, :mid]
    B22 = B[..., mid:, mid:]

    # Compute the 7 products using torch.mm (Base Case for the 2nd layer)
    M1 = torch.mm(A11 + A22, B11 + B22)
    M2 = torch.mm(A21 + A22, B11)
    M3 = torch.mm(A11, B12 - B22)
    M4 = torch.mm(A22, B21 - B11)
    M5 = torch.mm(A11 + A12, B22)
    M6 = torch.mm(A21 - A11, B11 + B12)
    M7 = torch.mm(A12 - A22, B21 + B22)

    # Compute the quadrants of the result
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine the quadrants
    return torch.vstack([
        torch.hstack([C11, C12]),
        torch.hstack([C21, C22])
    ])

@torch.compile
def strassen_matmul_two_layers(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs Strassen's matrix multiplication unrolled for two layers.

    Args:
        A: The first input matrix (N x N).
        B: The second input matrix (N x N).

    Returns:
        The result of A @ B using two layers of Strassen.
    """
    n = A.shape[0]

    # if n != B.shape[0] or A.shape[1] != B.shape[0] or A.shape[0] != A.shape[1]:
    #     raise ValueError("Input matrices must be square and have the same dimensions.")
    # if n % 4 != 0:
    #     raise ValueError("Matrix size must be divisible by 4 for two Strassen layers.")
    # if not n > 0:
    #      raise ValueError("Matrix size must be positive.")

    mid_l1 = n // 2
    A11 = A[:mid_l1, :mid_l1]
    A12 = A[:mid_l1, mid_l1:]
    A21 = A[mid_l1:, :mid_l1]
    A22 = A[mid_l1:, mid_l1:]

    B11 = B[:mid_l1, :mid_l1]
    B12 = B[:mid_l1, mid_l1:]
    B21 = B[mid_l1:, :mid_l1]
    B22 = B[mid_l1:, mid_l1:]

    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A21 - A11
    S10 = B11 + B12

    M1 = _strassen_base(S5, S6)  # M1 = (A11 + A22) * (B11 + B22)
    M2 = _strassen_base(S3, B11) # M2 = (A21 + A22) * B11
    M3 = _strassen_base(A11, S1) # M3 = A11 * (B12 - B22)
    M4 = _strassen_base(A22, S4) # M4 = A22 * (B21 - B11)
    M5 = _strassen_base(S2, B22) # M5 = (A11 + A12) * B22
    M6 = _strassen_base(S9, S10)# M6 = (A21 - A11) * (B11 + B12)
    M7 = _strassen_base(S7, S8)  # M7 = (A12 - A22) * (B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = torch.vstack([
        torch.hstack([C11, C12]),
        torch.hstack([C21, C22])
    ])

    return C
