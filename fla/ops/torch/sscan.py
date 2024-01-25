import torch
from einops import rearrange, einsum 

def selective_scan(v, delta, A, B, C, D, initial_state=None):
    """
    Args:
        v: B x L x d_v
        delta: B x L x d_v
        A: d_v x d_k
        B: B x L x d_k
        C: B x L x d_k
        D: B x L x d_v
        initial_state: B x d x d_k

    Intermediate variables:
        A_bar: B x L x d_v x d_k
        B_bar: B x L x d_v x d_k

    Returns:
        y: B x L x d_v
    """
    bs, L, d_v = v.shape
    _d_v, d_k = A.shape

    A_bar = torch.exp(einsum(delta, A, 'b l dv, dv dk -> b l dv dk'))
    B_bar = einsum(delta, B, 'b l dv, b l dk -> b l dv dk')

    if initial_state is None:
        s = torch.zeros((bs, d_v, d_k), device=v.device)
    else:
        s = initial_state

    ys = []    
    for i in range(L):
        s = A_bar[:, i] * s + B_bar[:, i] * v[:, i, :, None]
        y = einsum(s, C[:, i, :], 'b d n, b n -> b d')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_v)
    
    y = y + v * D[None, None, :]
    final_state = s.transpose(-1, -2) # B x d_k x d_v
    return y, final_state
