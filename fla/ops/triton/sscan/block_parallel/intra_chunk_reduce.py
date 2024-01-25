import torch
import triton 
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    x, A_bar, B_bar,
    O,
    CHUNK_SIZE: tl.constexpr, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bn = tl.program_id(0)
    offset_v = tl.program_id(1)
    
    x = x + offset_bn * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)

    A_bar = A_bar + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    B_bar = B_bar + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    O = O + offset_bn * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL +  tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    acc = tl.zeros([D_MODEL_K, BLOCK_MODEL], dtype=tl.float32)
    for _ in range(CHUNK_SIZE):
        x_i = tl.load(x)
        A_bar_i = tl.load(A_bar)
        B_bar_i = tl.load(B_bar)
        x_bar_i = x_i[None, :] * B_bar_i
        acc = acc * A_bar_i + x_bar_i
        tl.store(O, acc.to(O.dtype.element_ty))

        x += D_MODEL_V
        A_bar +=  D_MODEL_K * D_MODEL_V
        B_bar +=  D_MODEL_K * D_MODEL_V       
        O +=  D_MODEL_K * D_MODEL_V


@triton.jit
def _bwd_recurrence(
    x, A_bar, B_bar, S,
    dC, dx, dA_bar, dB_bar,
    CHUNK_SIZE: tl.constexpr, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
    
 ):
    offset_bh = tl.program_id(0)
    offset_v = tl.program_id(1)    

    x = x + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V
    dx = dx + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V

    A_bar = A_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    dA_bar = dA_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    B_bar = B_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    dB_bar = dB_bar + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V

    # previous state
    S = S + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 2) * D_MODEL_K * D_MODEL_V

    dC = dC + offset_bh * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    Dacc = tl.load(dC)

    for _ in range(CHUNK_SIZE - 1):
        x_i = tl.load(x)
        A_bar_i = tl.load(A_bar)
        B_bar_i = tl.load(B_bar)
        S_i = tl.load(S)

        dA_bar_i = S_i * Dacc
        dB_bar_i = x_i[None, :] * Dacc
        dx_i = tl.sum(B_bar_i * Dacc, axis=0)
        tl.store(dA_bar, dA_bar_i.to(x.dtype.element_ty))
        tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
        tl.store(dx, dx_i.to(x.dtype.element_ty))
        
        Dacc = A_bar_i * Dacc

        x -= D_MODEL_V
        dx -= D_MODEL_V
        A_bar -= D_MODEL_K * D_MODEL_V
        B_bar -= D_MODEL_K * D_MODEL_V
        dA_bar -= D_MODEL_K * D_MODEL_V
        dB_bar -= D_MODEL_K * D_MODEL_V
        S -= D_MODEL_K * D_MODEL_V
    
    # handle first element, note that C_0 = 0
    x_i = tl.load(x)
    B_bar_i = tl.load(B_bar)
    dB_bar_i = x_i[None, :] * Dacc
    tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
    dx_i = tl.sum(B_bar_i * Dacc, axis=0)
    tl.store(dx, dx_i.to(x.dtype.element_ty))
    
class IntraChunkReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A_bar, B_bar):
        x = x.contiguous()
        A_bar = A_bar.contiguous()
        B_bar = B_bar.contiguous()
        
        B, N, L, D_k, D_v = A_bar.shape 
        S = torch.empty_like(A_bar)        

        assert D_k in {16, 32}
        BLOCK_MODEL = D_k
        assert D_v % BLOCK_MODEL == 0

        grid = (B * N, D_v//BLOCK_MODEL)
        ctx.grid = grid
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            x, A_bar, B_bar,
            S,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            CHUNK_SIZE=L,  
            BLOCK_MODEL=BLOCK_MODEL
        )
        
        ctx.save_for_backward(x, A_bar, B_bar, S) 
        return S[:, :, -1]  # B x N x D_k x D_v

    @staticmethod
    def backward(ctx, dC):
        dC = dC.contiguous()

        x, A_bar, B_bar, S = ctx.saved_tensors 
        B, N, L, D_k, D_v = S.shape 

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        dx = torch.empty_like(x)
        dA_bar = torch.empty_like(A_bar)
        dB_bar = torch.empty_like(B_bar)

        _bwd_recurrence[grid](
            x, A_bar, B_bar, S,
            dC, dx, dA_bar, dB_bar,
            CHUNK_SIZE = L,
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL
        )
        dA_bar[:, :, 0] = 0
        return dx, dA_bar, dB_bar

def naive_chunk_reduce(x, A_bar, B_bar, initial_state=None):
    bs, num_chunk, chunk_size, d_k, d_v = A_bar.shape
    if initial_state is None:
        initial_state = torch.zeros((bs, num_chunk, d_k, d_v), dtype=x.dtype, device=x.device)

    s = initial_state
    for idx in range(chunk_size):
        s = A_bar[:, :, idx] * s + B_bar[:, :, idx] * x[:, :, idx, None, :]
    return s
    

if __name__ == "__main__":
    d_k = 16
    bs, num_chunk, chunk_size, d_v = 4, 8, 16, 128
    v = torch.randn((bs, num_chunk, chunk_size, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    log_A_bar = torch.randn((bs, num_chunk, chunk_size, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    A_bar = torch.exp(- log_A_bar)
    B_bar = torch.randn((bs, num_chunk, chunk_size, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    # initial_state = torch.randn((bs, num_chunk, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)

    do = torch.randn_like(v)

    ref = naive_chunk_reduce(v, A_bar, B_bar)
    ref.backward(do, retain_graph=True)
    ref_dlogA, log_A_bar.grad = log_A_bar.grad.clone(), None
    ref_dB, B_bar.grad = B_bar.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = IntraChunkReduce.apply(v, A_bar, B_bar)
    tri.backward(do)
    tri_dlogA, log_A_bar.grad = log_A_bar.grad.clone(), None
    tri_dB, B_bar.grad = B_bar.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, 1e1)
    assert ref_dlogA.allclose(tri_dlogA, 0, 1e1)
    assert ref_dB.allclose(tri_dB, 0, 1e-0)
    assert ref_dv.allclose(tri_dv, 0, 1e-0)