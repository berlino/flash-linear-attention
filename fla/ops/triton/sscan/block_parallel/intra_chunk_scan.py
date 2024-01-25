import torch
import triton 
import triton.language as tl
from einops import rearrange, einsum

@triton.jit
def _fwd_recurrence(
    x, H, A_bar, B_bar, query,
    y, S,
    CHUNK_SIZE: tl.constexpr, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bn = tl.program_id(0)
    offset_v = tl.program_id(1)
    
    x = x + offset_bn * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)

    A_bar = A_bar + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    B_bar = B_bar + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    S = S + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL +  tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    query = query + offset_bn * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    y = y + offset_bn * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)

    H = H + offset_bn * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    acc = tl.load(H)

    for _ in range(CHUNK_SIZE):
        x_i = tl.load(x)
        A_bar_i = tl.load(A_bar)
        B_bar_i = tl.load(B_bar)
        x_bar_i = x_i[None, :] * B_bar_i
        acc = acc * A_bar_i + x_bar_i
        query_i = tl.load(query)
        y_i = tl.sum(query_i[:, None] * acc, axis=0)
        tl.store(y, y_i.to(x.dtype.element_ty))
        tl.store(S, acc.to(x.dtype.element_ty))

        x += D_MODEL_V
        query += D_MODEL_K
        A_bar += D_MODEL_K * D_MODEL_V
        B_bar += D_MODEL_K * D_MODEL_V     
        S +=  D_MODEL_K * D_MODEL_V
        y +=  D_MODEL_V


@triton.jit
def _bwd_recurrence(
    x, H, A_bar, B_bar, query, S,
    dy, 
    dx, dH, dA_bar, dB_bar, dquery,
    CHUNK_SIZE: tl.constexpr, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr, NUM_V_BLOCKS: tl.constexpr
 ):
    offset_bh = tl.program_id(0)
    offset_v = tl.program_id(1)    

    x = x + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V
    dx = dx + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V

    A_bar = A_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    dA_bar = dA_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V

    B_bar = B_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    dB_bar = dB_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V

    # current state
    S = S + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V

    H = H + offset_bh * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    dH = dH + offset_bh * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    query = query + offset_bh * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K
    dquery = dquery + offset_bh * CHUNK_SIZE * NUM_V_BLOCKS * D_MODEL_K + (CHUNK_SIZE - 1) * NUM_V_BLOCKS * D_MODEL_K + offset_v * D_MODEL_K + tl.arange(0, D_MODEL_K)

    dy = dy + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V

    D_acc = tl.zeros([D_MODEL_K, BLOCK_MODEL], dtype=tl.float32)

    # last item
    S_i = tl.load(S)
    dy_i = tl.load(dy)
    query_i = tl.load(query)

    dquery_i = tl.sum(dy_i[None, :] * S_i, axis=1)
    tl.store(dquery, dquery_i.to(x.dtype.element_ty))

    D_acc += query_i[:, None] * dy_i[None, :]

    x_i = tl.load(x)
    B_bar_i = tl.load(B_bar)
    dB_bar_i = x_i[None, :] * D_acc
    dx_i = tl.sum(B_bar_i * D_acc, axis=0)
    tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
    tl.store(dx, dx_i.to(x.dtype.element_ty))

    x -= D_MODEL_V
    dx -= D_MODEL_V
    dy -= D_MODEL_V
    query -= D_MODEL_K
    dquery -= D_MODEL_K * NUM_V_BLOCKS
    B_bar -= D_MODEL_K * D_MODEL_V
    dB_bar -= D_MODEL_K * D_MODEL_V
    S -= D_MODEL_K * D_MODEL_V


    # middle items
    for _ in range(CHUNK_SIZE - 1):
        S_i = tl.load(S)
        dy_i = tl.load(dy)
        query_i = tl.load(query)

        dquery_i = tl.sum(dy_i[None, :] * S_i, axis=1)
        tl.store(dquery, dquery_i.to(x.dtype.element_ty))
    
        dA_bar_i = S_i * D_acc
        tl.store(dA_bar, dA_bar_i.to(x.dtype.element_ty))
        dA_bar -= D_MODEL_K * D_MODEL_V

        # update \nabla_H L
        A_bar_i = tl.load(A_bar)
        A_bar -= D_MODEL_K * D_MODEL_V
        D_acc = D_acc * A_bar_i + query_i[:, None] * dy_i[None, :]

        x_i = tl.load(x)
        B_bar_i = tl.load(B_bar)
        dB_bar_i = x_i[None, :] * D_acc
        dx_i = tl.sum(B_bar_i * D_acc, axis=0)
        tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
        tl.store(dx, dx_i.to(x.dtype.element_ty))

        x -= D_MODEL_V
        dx -= D_MODEL_V
        dy -= D_MODEL_V
        query -= D_MODEL_K
        dquery -= D_MODEL_K * NUM_V_BLOCKS
        B_bar -= D_MODEL_K * D_MODEL_V
        dB_bar -= D_MODEL_K * D_MODEL_V
        S -= D_MODEL_K * D_MODEL_V

    # the first telement
    S_i = tl.load(H)
    dA_bar_i = S_i * D_acc
    A_bar_i = tl.load(A_bar)
    tl.store(dA_bar, dA_bar_i.to(x.dtype.element_ty))
    D_acc = A_bar_i * D_acc
    tl.store(dH, D_acc.to(x.dtype.element_ty))
    
    
class IntraChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, H, A_bar, B_bar, query):
        x = x.contiguous()
        H = H.contiguous()
        A_bar = A_bar.contiguous()
        B_bar = B_bar.contiguous()
        query = query.contiguous()
        
        B, N, L, D_k, D_v = A_bar.shape 
        y = torch.empty_like(x)
        S = torch.empty_like(A_bar)      

        assert D_k in {16, 32}
        BLOCK_MODEL = D_k
        assert D_v % BLOCK_MODEL == 0

        grid = (B * N, D_v//BLOCK_MODEL)
        ctx.grid = grid
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            x, H, A_bar, B_bar, query,
            y, S,
            CHUNK_SIZE=L,  
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL=BLOCK_MODEL
        )
        
        ctx.save_for_backward(x, H, A_bar, B_bar, query, S) 
        return y 

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()

        x, H, A_bar, B_bar, query, S = ctx.saved_tensors 
        B, N, L, D_k, D_v = S.shape 

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        dx = torch.empty_like(x)
        dH = torch.empty_like(H)
        dA_bar = torch.empty_like(A_bar)
        dB_bar = torch.empty_like(B_bar)

        num_v_blocks = D_v // BLOCK_MODEL
        assert D_v % BLOCK_MODEL == 0
        dquery = torch.empty(B, N, L, num_v_blocks, D_k, device=x.device, dtype=x.dtype)

        _bwd_recurrence[grid](
            x, H, A_bar, B_bar, query, S, 
            dy, 
            dx, dH, dA_bar, dB_bar, dquery,
            CHUNK_SIZE = L,
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL,
            NUM_V_BLOCKS = num_v_blocks
        )
        dquery = dquery.sum(dim=3)
        return dx, dH, dA_bar, dB_bar, dquery

def naive_chunk_scan(v, H, A_bar, B_bar, C):
    bs, num_chunk, chunk_size, d_k, d_v = A_bar.shape

    s = H
    res = []
    for idx in range(chunk_size):
        s = A_bar[:, :, idx] * s + B_bar[:, :, idx] * v[:, :, idx, None, :]
        o = einsum(s, C[:, :, idx],"b n dk dv, b n dk -> b n dv") 
        res.append(o)
    return torch.stack(res, dim=2)



if __name__ == "__main__":
    d_k = 16
    bs, num_chunk, chunk_size, d_v = 4, 8, 16, 128
    v = torch.randn((bs, num_chunk, chunk_size, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    A = torch.randn((bs, num_chunk, chunk_size, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    log_A_bar = - torch.exp(A)
    A_bar = torch.exp(log_A_bar)
    B_bar = torch.randn((bs, num_chunk, chunk_size, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    C = torch.randn((bs, num_chunk, chunk_size, d_k), dtype=torch.float32, device='cuda').requires_grad_(True)
    H = torch.randn((bs, num_chunk, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)

    do = torch.randn_like(v)

    ref = naive_chunk_scan(v, H, A_bar, B_bar, C)
    ref.backward(do, retain_graph=True)
    ref_dA, A.grad = A.grad.clone(), None
    ref_dB, B_bar.grad = B_bar.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_C, C.grad = C.grad.clone(), None
    ref_H, H.grad = H.grad.clone(), None

    tri = IntraChunkScan.apply(v, H, A_bar, B_bar, C)
    tri.backward(do)
    tri_dA, A.grad = A.grad.clone(), None
    tri_dB, B_bar.grad = B_bar.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_C, C.grad = C.grad.clone(), None
    tri_H, H.grad = H.grad.clone(), None

    breakpoint()
    assert ref.allclose(tri, 0, 1e1)
    assert ref_dA.allclose(tri_dA, 0, 1e1)
    assert ref_dB.allclose(tri_dB, 0, 1e-0)
    assert ref_dv.allclose(tri_dv, 0, 1e-0)
    assert ref_C.allclose(tri_C, 0, 1e-0)
    assert ref_H.allclose(tri_H, 0, 1e-0)
