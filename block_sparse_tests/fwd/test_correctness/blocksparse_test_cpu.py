import pytest
import torch
from einops import repeat

from block_sparse_attn import block_sparse_attn_func
from utils import (
    attention_blocksparse_ref,
    generate_base_sparsity_mask,
    generate_qkv,
    tailor_mixedmask_for_test,
)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(32, 64), (64, 128)])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("sparsity", [0.25, 0.75])
def test_block_sparse_attn_cpu_mask_type_1_matches_reference(
    batch_size, nheads, seqlen_q, seqlen_k, d, sparsity
):
    torch.manual_seed(0)
    device = "cpu"
    block_size = 128

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.float32)

    q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, _, _, _, output_pad_fn, _, _ = generate_qkv(
        q, k, v, None, None, kvpacked=False
    )

    head_mask_type = torch.arange(1, nheads + 1, device=device, dtype=torch.int32)
    base_blockmask = generate_base_sparsity_mask(
        max_seqlen_q,
        max_seqlen_k,
        block_size,
        block_size,
        block_size,
        batch_size,
        nheads,
        [sparsity] * nheads,
        causal=False,
        device=device,
    )

    out_unpad = block_sparse_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info=None,
        base_blockmask=base_blockmask,
        max_seqlen_q_=max_seqlen_q,
        max_seqlen_k_=max_seqlen_k,
        p_dropout=0.0,
        deterministic=False,
        softmax_scale=1.0 / (d ** 0.5),
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = output_pad_fn(out_unpad).to(torch.float32)

    ref_mask = repeat(base_blockmask, "b h s_m s_n -> b h (s_m d_m) (s_n d_n)", d_m=block_size, d_n=block_size)
    ref_mask = tailor_mixedmask_for_test(ref_mask, seqlen_q, seqlen_k)
    ref_mask = ~ref_mask

    ref_out, _ = attention_blocksparse_ref(
        q,
        k,
        v,
        ref_mask,
        block_size,
        block_size,
        query_padding_mask=None,
        key_padding_mask=None,
        p_dropout=0.0,
        dropout_mask=None,
        causal=False,
        window_size=(-1, -1),
        upcast=True,
        reorder_ops=False,
    )

    assert out.shape == q.shape
    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-3)
