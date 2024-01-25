# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

from fla.ops.triton.sscan import recurrent_fuse
from fla.ops.torch.sscan import selective_scan as naive_selective_scan
from fla.ops.triton.sscan.block_parallel.sscan_block_parallel import selective_scan_block_parallel

@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("d_v", [128])
@pytest.mark.parametrize("L", [32, 64])
def test_sscan(dtype, d_v, L, d_k=16, bs=4):
    v = torch.randn((bs, L, d_v), dtype=dtype, device='cuda').requires_grad_(True)
    A_prime = torch.randn((d_v, d_k), dtype=dtype, device='cuda').requires_grad_(True)
    A = - torch.exp(A_prime)
    B = torch.randn((bs, L, d_k), dtype=dtype, device='cuda').requires_grad_(True)
    C = torch.randn((bs, L, d_k), dtype=dtype, device='cuda').requires_grad_(True)
    D = torch.randn(d_v, dtype=dtype, device='cuda').requires_grad_(True)
    delta_prime = torch.randn((bs, L, d_v), dtype=dtype, device='cuda').requires_grad_(True)
    delta = F.softplus(delta_prime)
    initial_state = torch.randn((bs, d_v, d_k), dtype=dtype, device='cuda').requires_grad_(False)
    # initial_state = None

    do = torch.randn_like(v) 
    ref, ref_final_state = naive_selective_scan(v, delta, A, B, C, D, initial_state)
    ref.backward(do, retain_graph=True)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dA_prime, A_prime.grad = A_prime.grad.clone(), None
    ref_dB, B.grad = B.grad.clone(), None
    ref_dC, C.grad = C.grad.clone(), None
    ref_dD, D.grad = D.grad.clone(), None
    ref_ddelta_prime, delta_prime.grad = delta_prime.grad.clone(), None

    initial_state = initial_state.transpose(1, 2).contiguous() # b * dk * dv
    tri, tri_final_state = selective_scan_block_parallel(v, delta, A, B, C, D, initial_state)
    tri.backward(do, retain_graph=True)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dA_prime, A_prime.grad = A_prime.grad.clone(), None
    tri_dB, B.grad = B.grad.clone(), None
    tri_dC, C.grad = C.grad.clone(), None
    tri_dD, D.grad = D.grad.clone(), None
    tri_ddelta_prime, delta_prime.grad = delta_prime.grad.clone(), None

    # breakpoint()
    assert ref.allclose(tri, 0, 1e-1)
    assert ref_final_state.allclose(tri_final_state, 0, 1e-1)
    assert ref_dv.allclose(tri_dv, 0, 1e-1)
    assert ref_dA_prime.allclose(tri_dA_prime, 0, 1e-1)
    assert ref_dB.allclose(tri_dB, 0, 1e-1)
    assert ref_dC.allclose(tri_dC, 0, 1e-1)
    assert ref_dD.allclose(tri_dD, 0, 1e-1)
    assert ref_ddelta_prime.allclose(tri_ddelta_prime, 0, 1e-1)
    assert ref_final_state.allclose(tri_final_state, 0, 1e-1)

    print('Done!')

