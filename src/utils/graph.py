import torch

def compute_support_powers(S: torch.Tensor, K: int):
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

    if K == 0:
        return powers

    cur = S
    for _ in range(1, K + 1):
        powers.append(cur)
        cur = cur @ S
    return powers

