import torch

def compute_support_powers(S: torch.Tensor, K: int):
    """Return a list [S^0, S^1, ..., S^K].

    Args:
        S:   (N, N) dense or sparse support matrix (e.g. Laplacian or adjacency)
        K:   highest power.
    """
    powers = [torch.eye(S.size(0), device=S.device, dtype=S.dtype)]
    if K == 0:
        return powers
    cur = S
    for _ in range(1, K + 1):
        powers.append(cur)
        cur = cur @ S  # dense multiplication; replace with sparse ops if needed
    return powers