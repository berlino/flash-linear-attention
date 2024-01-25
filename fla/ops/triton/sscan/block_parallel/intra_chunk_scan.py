import torch
import triton 
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    x, C, A_bar, B_bar, query,
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

    C = C + offset_bn * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    acc = tl.load(C)

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
    x, C, A_bar, B_bar, query, S,
    dy, dx, dC, dA_bar, dB_bar, dquery,
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
    S1 = S + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V
    # previous state
    S2 = S + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :] + (CHUNK_SIZE - 2) * D_MODEL_K * D_MODEL_V

    # initial state
    C = C + offset_bh * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]
    dC = dC + offset_bh * D_MODEL_K * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, D_MODEL_K)[:, None] * D_MODEL_V + tl.arange(0, BLOCK_MODEL)[None, :]

    query = query + offset_bh * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K
    dquery = dquery + offset_bh * CHUNK_SIZE * NUM_V_BLOCKS * D_MODEL_K + (CHUNK_SIZE - 1) * NUM_V_BLOCKS * D_MODEL_K + offset_v * D_MODEL_K + tl.arange(0, D_MODEL_K)

    dy = dy + offset_bh * CHUNK_SIZE * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL) + (CHUNK_SIZE - 1) * D_MODEL_V

    Dacc = tl.zeros([D_MODEL_K, BLOCK_MODEL], dtype=tl.float32)
    for _ in range(CHUNK_SIZE - 1):
        dy_i = tl.load(dy)
        query_i = tl.load(query)
        S1_i = tl.load(S1)
        S2_i = tl.load(S2)
        Dacc += query_i[:, None] * dy_i[None, :]
        dquery_i = tl.sum(dy_i[None, :] * S1_i, axis=1)
        tl.store(dquery, dquery_i.to(x.dtype.element_ty))

        x_i = tl.load(x)
        A_bar_i = tl.load(A_bar)
        B_bar_i = tl.load(B_bar)

        dA_bar_i = S2_i * Dacc
        dB_bar_i = x_i[None, :] * Dacc
        dx_i = tl.sum(B_bar_i * Dacc, axis=0)
        tl.store(dA_bar, dA_bar_i.to(x.dtype.element_ty))
        tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
        tl.store(dx, dx_i.to(x.dtype.element_ty))
        
        Dacc = A_bar_i * Dacc

        x -= D_MODEL_V
        dx -= D_MODEL_V
        query -= D_MODEL_K
        dquery -= D_MODEL_K * NUM_V_BLOCKS
        A_bar -= D_MODEL_K * D_MODEL_V
        B_bar -= D_MODEL_K * D_MODEL_V
        dA_bar -= D_MODEL_K * D_MODEL_V
        dB_bar -= D_MODEL_K * D_MODEL_V
        S1 -= D_MODEL_K * D_MODEL_V
        S2 -= D_MODEL_K * D_MODEL_V
    
    # handle first element
    dy_i = tl.load(dy)
    query_i = tl.load(query)
    S1_i = tl.load(S1)
    S2_i = tl.load(C) # inital state
    Dacc += query_i[:, None] * dy_i[None, :]
    dquery_i = tl.sum(dy_i[None, :] * S1_i, axis=1)
    tl.store(dquery, dquery_i.to(x.dtype.element_ty))

    x_i = tl.load(x)
    A_bar_i = tl.load(A_bar)
    B_bar_i = tl.load(B_bar)

    dA_bar_i = S2_i * Dacc
    dB_bar_i = x_i[None, :] * Dacc
    dx_i = tl.sum(B_bar_i * Dacc, axis=0)
    tl.store(dA_bar, dA_bar_i.to(x.dtype.element_ty))
    tl.store(dB_bar, dB_bar_i.to(x.dtype.element_ty))
    tl.store(dx, dx_i.to(x.dtype.element_ty))
    
    Dacc = A_bar_i * Dacc
    tl.store(dC, Dacc.to(x.dtype.element_ty))
    
class IntraChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, C, A_bar, B_bar, query):
        x = x.contiguous()
        C = C.contiguous()
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
            x, C, A_bar, B_bar, query,
            y, S,
            CHUNK_SIZE=L,  
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL=BLOCK_MODEL
        )
        
        ctx.save_for_backward(x, C, A_bar, B_bar, query, S) 
        return y 

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()

        x, C, A_bar, B_bar, query, S = ctx.saved_tensors 
        B, N, L, D_k, D_v = S.shape 

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        dx = torch.empty_like(x)
        dC = torch.empty_like(C)
        dA_bar = torch.empty_like(A_bar)
        dB_bar = torch.empty_like(B_bar)

        num_v_blocks = D_v // BLOCK_MODEL
        assert D_v % BLOCK_MODEL == 0
        dquery = torch.empty(B, N, L, num_v_blocks, D_k, device=x.device, dtype=x.dtype)

        _bwd_recurrence[grid](
            x, C, A_bar, B_bar, query, S, 
            dy, dx, dC, dA_bar, dB_bar, dquery,
            CHUNK_SIZE = L,
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL,
            NUM_V_BLOCKS = num_v_blocks
        )
        dquery = dquery.sum(dim=3)
        return dx, dC, dA_bar, dB_bar, dquery