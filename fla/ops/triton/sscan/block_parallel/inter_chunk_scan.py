import torch
import triton 
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    X, F, 
    H,
    init_H, final_H,
    NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bn = tl.program_id(0)
    offset_v = tl.program_id(1)    
    
    X = X + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    H = H + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] 

    F = F + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL +  tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] 

    init_H = init_H + offset_bn * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    final_H = final_H + offset_bn * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    # initial chunk
    acc = tl.load(init_H)
    tl.store(H, acc.to(H.dtype.element_ty))  # save to first chunk

    # for-loop
    H += D_MODEL_K * D_MODEL_V  # second chunk
    for _ in range(NUM_BLOCK-1):
        F_i = tl.load(F) 
        X_i = tl.load(X) 
        acc = acc * F_i + X_i  # for third chunk
        tl.store(H, acc.to(H.dtype.element_ty))
        X +=  D_MODEL_K * D_MODEL_V
        H +=  D_MODEL_K * D_MODEL_V       
        F +=  D_MODEL_K * D_MODEL_V

    # last chunk
    F_i = tl.load(F)
    X_i = tl.load(X)
    acc = acc * F_i + X_i
    tl.store(final_H, acc.to(H.dtype.element_ty))

@triton.jit
def _bwd_recurrence(
    H, F,
    dH, 
    dX, dF,
    NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
 ):
    """
    Note:
        the gradient of X is stored in X.
    """
    offset_bn = tl.program_id(0)
    offset_v = tl.program_id(1)    

    H = H + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    F = F + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    dH = dH + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[None, :]  + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    dX = dX + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    dF = dF + offset_bn * NUM_BLOCK * D_MODEL_K * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # load \nabla_{H_t+1} Loss
    Dacc = tl.load(dH)
    dH -= D_MODEL_K * D_MODEL_V

    # ignore the last chunk
    for _ in range(NUM_BLOCK - 1):
        F_i = tl.load(F)
        H_i = tl.load(H)
        dH_i = tl.load(dH)

        # Dacc means F_t \odot \nabla_{H_t+1} L in the note 
        dF_i = Dacc * H_i
        tl.store(dF, dF_i.to(F.dtype.element_ty))
        tl.store(dX, Dacc.to(F.dtype.element_ty))        
        
        # update \nabla_{H_t+1} L 
        Dacc = F_i * Dacc + dH_i

        H -= D_MODEL_K * D_MODEL_V 
        F -= D_MODEL_K * D_MODEL_V
        dH -= D_MODEL_K * D_MODEL_V 
        dX -= D_MODEL_K * D_MODEL_V 
        dF -= D_MODEL_K * D_MODEL_V
    
class InterChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, F, initial_state):
        F = F.contiguous()
        X = X.contiguous()

        if initial_state is None:
            initial_state = torch.zeros_like(X[:, 0])
        
        B, N, D_k, D_v = X.shape 
        H = torch.empty_like(X)        
        final_state = torch.zeros_like(initial_state)

        assert D_k in {16, 32}
        BLOCK_MODEL = D_k
        assert D_v % BLOCK_MODEL == 0

        grid = (B, D_v//BLOCK_MODEL)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            X, F,
            H, 
            initial_state, final_state,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N,  
            BLOCK_MODEL=D_k
        )
        
        ctx.save_for_backward(X, F, H)        
        return H, final_state

    @staticmethod
    def backward(ctx, dH, dfinal_state):
        dH = dH.contiguous()

        X, F, H = ctx.saved_tensors 
        B, N, D_k, D_v = H.shape 
        num_block = N

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        dX = torch.empty_like(X)
        dF = torch.empty_like(F)

        _bwd_recurrence[grid](
            H, F,
            dH, 
            dX, dF,
            NUM_BLOCK = num_block,
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL
        )

        dX[:,  -1] = 0
        dF[:, -1] = 0
        return dX, dF, None

def naive_inter_chunk_scan(X, F, initial_state=None):
    bs, num_chunk, d_k, d_v = X.shape
    if initial_state is None:
        initial_state = torch.zeros(bs, d_k, d_v, dtype=X.dtype, device=X.device)

    Hs = []
    for i in range(num_chunk + 1):
        if i == 0:
            H_i = initial_state
            Hs.append(H_i)
        elif i == num_chunk:
            final_state = F[:, i-1] * H_i + X[:, i-1]
        else:
            H_i = F[:, i-1] * H_i + X[:, i-1]
            Hs.append(H_i)
    H = torch.stack(Hs, 1)
    
    return H, final_state


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    bs, num_chunk, d_k, d_v = 4, 32, 16, 128
    X = torch.randn((bs, num_chunk, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    F = torch.rand((bs, num_chunk, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    initial_state = torch.randn((bs, d_k, d_v), dtype=torch.float32, device='cuda').requires_grad_(True)
    # initial_state = None

    do = torch.randn_like(X)

    ref, ref_final_state = naive_inter_chunk_scan(X, F, initial_state)
    ref.backward(do, retain_graph=True)
    ref_dX, X.grad = X.grad.clone(), None
    ref_dF, F.grad = F.grad.clone(), None

    tri, tri_final_state = InterChunkScan.apply(X, F, initial_state)
    tri.backward(do, None)
    tri_dX, X.grad = X.grad.clone(), None
    tri_dF, F.grad = F.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-3)
    assert ref_final_state.allclose(tri_final_state, 0, 1e-3)
    assert ref_dX.allclose(tri_dX, 0, 1e-3)
    assert ref_dF.allclose(tri_dF, 0, 1e-3)

    print(ref.sum().item(), tri.sum().item())
    print(ref_final_state.sum().item(), tri_final_state.sum().item())
    breakpoint()


