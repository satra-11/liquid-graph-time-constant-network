import torch


def compute_laplacian(A: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Computes the Laplacian matrix from an adjacency matrix."""
    if A.ndim == 2:
        A = A.unsqueeze(0)  # Add a batch dimension

    # A: (B, N, N)
    D = torch.sum(A, dim=-1, keepdim=True)  # Degree matrix (B, N, 1)

    if normalize:
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # Handle zero degrees
        Laplacian = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(
            0
        ) - D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)
    else:
        Laplacian = torch.diag_embed(D) - A  # Unnormalized Laplacian

    return Laplacian.squeeze(0)  # Remove batch dimension if added


def compute_random_walk_matrix(A: torch.Tensor) -> torch.Tensor:
    """Computes the Random Walk matrix from an adjacency matrix."""
    if A.ndim == 2:
        A = A.unsqueeze(0)  # Add a batch dimension

    # A: (B, N, N)
    D = torch.sum(A, dim=-1, keepdim=True)  # Degree matrix (B, N, 1)
    D_inv = torch.pow(D, -1)
    D_inv[torch.isinf(D_inv)] = 0.0  # Handle zero degrees

    random_walk_matrix = D_inv * A

    return random_walk_matrix.squeeze(0)  # Remove batch dimension if added


def compute_s_powers(S: torch.Tensor, K: int):
    """
    S: (..., N, N)   何軸あっても最後の 2 軸が正方なら OK
    K: 最高次数
    """
    if S.ndim == 2:
        S = S.unsqueeze(0)  # Add a batch dimension if it's missing

    *batch, N, N2 = S.shape
    assert N == N2, "S must be square on the last two dims"

    eye = torch.eye(N, device=S.device, dtype=S.dtype).expand(*batch, N, N)
    powers = [eye]

    cur = S
    for _ in range(1, K + 1):
        powers.append(cur)
        cur = cur @ S
    return powers
