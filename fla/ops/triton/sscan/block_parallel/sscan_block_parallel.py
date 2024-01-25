import torch
from einops import rearrange, einsum
import torch.nn.functional as F

from .inter_chunk_scan import InterChunkScan
from .intra_chunk_reduce import IntraChunkReduce
from .intra_chunk_scan import IntraChunkScan


def selective_scan_block_parallel(v, delta, A, B, C, D, initial_state=None):
    bs, L, d_v = v.shape

    log_A_bar = einsum(delta, A, 'b l dv, dv dk -> b l dk dv')
    A_bar = torch.exp(log_A_bar)
    B_bar = einsum(delta, B, 'b l dv, b l dk -> b l dk dv')

    chunk_size = 2
    N_CHUNK = L // chunk_size
    assert L % chunk_size == 0
    v = rearrange(v, "b (n c) dv -> b n c dv", n=N_CHUNK, c=chunk_size)
    A_bar = rearrange(A_bar, "b (n c) dk dv -> b n c dk dv", n=N_CHUNK, c=chunk_size)
    log_A_bar = rearrange(log_A_bar, "b (n c) dk dv -> b n c dk dv", n=N_CHUNK, c=chunk_size)
    B_bar = rearrange(B_bar, "b (n c) dk dv -> b n c dk dv", n=N_CHUNK, c=chunk_size)
    C = rearrange(C, "b (n c) dk -> b n c dk", n=N_CHUNK, c=chunk_size)

    chunk_state = IntraChunkReduce.apply(v, A_bar, B_bar)
    chunk_gate = log_A_bar.sum(dim=2).exp()
    chunk_state = InterChunkScan.apply(chunk_state, chunk_gate)
    y = IntraChunkScan.apply(v, chunk_state, A_bar, B_bar, C)
    y = y + v * D[None, None, :]
    y = rearrange(y, "b n c dv -> b (n c) dv")
    return y, None
