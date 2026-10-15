import json
import math
import os

import torch
import torch.nn.functional as F

from block_sparse_attn import block_sparse_attn_func
from utils import efficiency, flops, time_fwd, write_to_excel

BLOCK_SIZE = 128


def generate_base_sparsity_mask(
    max_seqlen_q,
    max_seqlen_k,
    round_base,
    m_block_dim,
    n_block_dim,
    sparsity,
    causal=False,
    device="cpu",
):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base

    nrow = round_to_multiple(max_seqlen_q, round_base) // m_block_dim
    ncol = round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)

    density = 1.0 - sparsity
    if 0.0 < density < 1.0:
        for i in range(nrow):
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                num_one = max(1, int(density * available_col_num))
                if available_col_num > 0:
                    base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])

    calculated_block_num = base_mask.sum().item()
    total_block_num = 0
    if causal:
        for i in range(nrow):
            total_block_num += max(0, ncol - i)
    else:
        total_block_num = nrow * ncol

    real_sparsity = 1.0 - (calculated_block_num / total_block_num) if total_block_num > 0 else 0.0
    return base_mask, real_sparsity


def dense_attention_reference(q, k, v, batch_size, seqlen, nheads, headdim):
    q = q.reshape(batch_size, seqlen, nheads, headdim).permute(0, 2, 1, 3)
    k = k.reshape(batch_size, seqlen, nheads, headdim).permute(0, 2, 1, 3)
    v = v.reshape(batch_size, seqlen, nheads, headdim).permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
    return out.permute(0, 2, 1, 3).reshape(batch_size * seqlen, nheads, headdim)


def profile_blocksparse_fwd_cpu():
    repeats = 10
    block_sparse_repeats = 5
    device = "cpu"
    dtype = torch.float32
    causal = False
    batch_size = 2
    seqlen_vals = [128, 256, 512, 1024, 2048]
    headdim = 64
    dim = 256
    dropout_p = 0.0
    nheads = dim // headdim

    excel_dir_path = "./excel/blocksparse_cpu/"
    excel_file_name = f"hdim{headdim}_nheads{nheads}_bts{batch_size}_fwd_cpu"
    excel_label = [
        "batch_size",
        "seqlen",
        "actual_sparsity",
        "speed",
        "latency",
        "speedup",
        "base_speed",
        "base_latency",
    ]
    excel_data = []
    all_results = {}

    print(f"\nCPU Configuration: headdim={headdim}, nheads={nheads}, batch_size={batch_size}, causal={causal}")
    print(f"{'SeqLen':<10} | {'Base Latency':<15} | {'Base TFLOPs':<15} | {'Avg Latency':<15} | {'Avg TFLOPs':<15} | {'Speedup':<10}")
    print("-" * 95)

    for seqlen in seqlen_vals:
        shape = (batch_size * seqlen, nheads, headdim)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)

        base_f = time_fwd(
            dense_attention_reference,
            q,
            k,
            v,
            batch_size,
            seqlen,
            nheads,
            headdim,
            repeats=repeats,
            verbose=False,
        )
        base_flops = flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd")
        base_speed = efficiency(base_flops, base_f)

        results = {}
        sparsity_list = [0.0, 0.25, 0.5, 0.75, 0.9]

        for sparsity in sparsity_list:
            sum_sparsity = 0.0
            sum_speed = 0.0
            sum_latency = 0.0

            for _ in range(block_sparse_repeats):
                base_blockmask, real_sparsity = generate_base_sparsity_mask(
                    seqlen,
                    seqlen,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                    sparsity,
                    causal=causal,
                    device=device,
                )
                base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
                head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)

                f = time_fwd(
                    block_sparse_attn_func,
                    q,
                    k,
                    v,
                    cu_seqlens,
                    cu_seqlens,
                    head_mask_type,
                    None,
                    base_blockmask,
                    seqlen,
                    seqlen,
                    dropout_p,
                    False,
                    1.0 / math.sqrt(headdim),
                    causal,
                    False,
                    False,
                    repeats=repeats,
                    verbose=False,
                )

                speed = efficiency(base_flops, f)
                sum_sparsity += real_sparsity
                sum_speed += speed
                sum_latency += f

            avg_sparsity = sum_sparsity / block_sparse_repeats
            avg_speed = sum_speed / block_sparse_repeats
            avg_latency = sum_latency / block_sparse_repeats
            speedup = base_f / avg_latency if avg_latency > 0 else 0.0

            results[avg_sparsity] = [avg_latency, avg_speed]
            excel_data.append([batch_size, seqlen, avg_sparsity, avg_speed, avg_latency, speedup, base_speed, base_f])
            print(f"{avg_sparsity:<10.4f} | {base_f*1000:>12.2f} ms | {base_speed:>12.2f} | {avg_latency*1000:>12.2f} ms | {avg_speed:>12.2f} | {speedup:>9.2f}x")

        all_results[seqlen] = {"base": [base_f, base_speed], "results": results}

    os.makedirs(excel_dir_path, exist_ok=True)
    with open(f"all_results_{excel_file_name}.json", "w") as f:
        json.dump(all_results, f, indent=4)

    write_to_excel(excel_label, excel_data, excel_dir_path, excel_file_name)
    print(f"\nResults saved to {excel_dir_path}{excel_file_name}.xlsx and all_results_{excel_file_name}.json")


if __name__ == "__main__":
    profile_blocksparse_fwd_cpu()
