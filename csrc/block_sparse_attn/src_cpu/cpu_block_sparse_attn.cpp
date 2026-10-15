#include <torch/extension.h>
#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

inline int64_t round_up(int64_t x, int64_t m) {
    return ((x + m - 1) / m) * m;
}

template <typename scalar_t>
void block_sparse_attention_cpu_impl(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& head_mask_type,
    const torch::Tensor& base_blockmask,
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const float softmax_scale,
    torch::Tensor& out) {
    const int64_t batch_size = cu_seqlens_q.size(0) - 1;
    const int64_t num_heads = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t block_m = 128;
    const int64_t block_n = 128;

    auto q_acc = q.accessor<scalar_t, 3>();
    auto k_acc = k.accessor<scalar_t, 3>();
    auto v_acc = v.accessor<scalar_t, 3>();
    auto out_acc = out.accessor<scalar_t, 3>();
    auto cuq_acc = cu_seqlens_q.accessor<int32_t, 1>();
    auto cuk_acc = cu_seqlens_k.accessor<int32_t, 1>();
    auto head_mask_acc = head_mask_type.accessor<int32_t, 1>();
    auto blockmask_bool = base_blockmask.to(torch::kBool).contiguous();
    auto blockmask_acc = blockmask_bool.accessor<bool, 4>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t q_start = cuq_acc[b];
        const int64_t q_end = cuq_acc[b + 1];
        const int64_t k_start = cuk_acc[b];
        const int64_t k_end = cuk_acc[b + 1];

        if (q_end <= q_start || k_end <= k_start) {
            continue;
        }

        for (int64_t h = 0; h < num_heads; ++h) {
            const int32_t mask_type = head_mask_acc[h];
            TORCH_CHECK(mask_type > 0,
                        "CPU block sparse attention only supports mask_type > 0; dense and streaming masks are not implemented.");
            const int64_t mask_index = mask_type - 1;
            TORCH_CHECK(mask_index < blockmask_bool.size(1),
                        "mask_type exceeds base_blockmask head count.");

            const int64_t nrow_blocks = blockmask_bool.size(2);
            const int64_t ncol_blocks = blockmask_bool.size(3);

            for (int64_t qi = q_start; qi < q_end; ++qi) {
                const int64_t q_relative = qi - q_start;
                const int64_t row_block = q_relative / block_m;
                TORCH_CHECK(row_block < nrow_blocks,
                            "Query row block index exceeds base_blockmask row dimension.");

                std::vector<int64_t> valid_k_idx;
                std::vector<float> scores;

                for (int64_t ki = k_start; ki < k_end; ++ki) {
                    const int64_t k_relative = ki - k_start;
                    const int64_t col_block = k_relative / block_n;
                    TORCH_CHECK(col_block < ncol_blocks,
                                "Key column block index exceeds base_blockmask column dimension.");

                    const bool allow = blockmask_acc[b][mask_index][row_block][col_block];
                    if (!allow) {
                        continue;
                    }

                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        score += static_cast<float>(q_acc[qi][h][d]) *
                                 static_cast<float>(k_acc[ki][h][d]);
                    }
                    score *= softmax_scale;
                    valid_k_idx.push_back(ki);
                    scores.push_back(score);
                }

                if (valid_k_idx.empty()) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        out_acc[qi][h][d] = static_cast<scalar_t>(0.0f);
                    }
                    continue;
                }

                const float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                std::vector<float> probs(scores.size());
                for (size_t i = 0; i < scores.size(); ++i) {
                    probs[i] = std::exp(scores[i] - max_score);
                    sum_exp += probs[i];
                }

                for (int64_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < valid_k_idx.size(); ++i) {
                        const int64_t ki = valid_k_idx[i];
                        const float weight = probs[i] / sum_exp;
                        acc += weight * static_cast<float>(v_acc[ki][h][d]);
                    }
                    out_acc[qi][h][d] = static_cast<scalar_t>(acc);
                }
            }
        }
    }
}

}  // namespace

torch::Tensor cpu_block_sparse_attn_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& head_mask_type,
    const torch::Tensor& base_blockmask,
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale) {
    TORCH_CHECK(q.device().is_cpu(), "CPU block sparse attention requires all inputs on CPU.");
    TORCH_CHECK(k.device().is_cpu(), "CPU block sparse attention requires all inputs on CPU.");
    TORCH_CHECK(v.device().is_cpu(), "CPU block sparse attention requires all inputs on CPU.");
    TORCH_CHECK(cu_seqlens_q.device().is_cpu(), "cu_seqlens_q must be on CPU.");
    TORCH_CHECK(cu_seqlens_k.device().is_cpu(), "cu_seqlens_k must be on CPU.");
    TORCH_CHECK(head_mask_type.device().is_cpu(), "head_mask_type must be on CPU.");
    TORCH_CHECK(base_blockmask.device().is_cpu(), "base_blockmask must be on CPU.");

    TORCH_CHECK(q.dim() == 3, "q must have shape (total_tokens, num_heads, head_dim)");
    TORCH_CHECK(k.dim() == 3, "k must have shape (total_tokens, num_heads, head_dim)");
    TORCH_CHECK(v.dim() == 3, "v must have shape (total_tokens, num_heads, head_dim)");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "num_heads mismatch");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "head_dim mismatch");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");
    TORCH_CHECK(head_mask_type.dtype() == torch::kInt32, "head_mask_type must be int32");
    TORCH_CHECK(base_blockmask.dim() == 4, "base_blockmask must have shape (batch, heads, nrow, ncol)");
    TORCH_CHECK(base_blockmask.size(0) == cu_seqlens_q.size(0) - 1,
                "base_blockmask batch dimension must match cu_seqlens_q length - 1");
    TORCH_CHECK(p_dropout == 0.0, "CPU block sparse attention only supports p_dropout == 0.0");

    auto out = torch::empty_like(q);
    auto base_blockmask_bool = base_blockmask.to(torch::kBool).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "cpu_block_sparse_attention",
        [&] {
            block_sparse_attention_cpu_impl<scalar_t>(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                base_blockmask_bool,
                max_seqlen_q,
                max_seqlen_k,
                static_cast<float>(softmax_scale),
                out);
        });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CPU block sparse attention mask_type=1 implementation";
    m.def("forward", &cpu_block_sparse_attn_forward, "CPU block sparse attention forward");
}
