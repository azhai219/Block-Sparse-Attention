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

torch::Tensor block_sparse_row_softmax(
    const torch::Tensor& logits,
    const float softmax_scale) {
    auto scaled_logits = logits * static_cast<double>(softmax_scale);
    return torch::softmax(scaled_logits, /*dim=*/1);
}

template <typename scalar_t>
void apply_single_head_block_sparse_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& base_blockmask,
    const int64_t sample_index,
    const int64_t head_index,
    const int64_t mask_index,
    const int64_t q_start,
    const int64_t q_end,
    const int64_t k_start,
    const int64_t k_end,
    const float softmax_scale,
    torch::Tensor& out) {
    const int64_t block_m = 128;
    const int64_t block_n = 128;
    const int64_t q_len = q_end - q_start;
    const int64_t k_len = k_end - k_start;

    const auto sample_mask = base_blockmask.select(0, sample_index).select(0, mask_index);
    const int64_t nrow_blocks = sample_mask.size(0);

    auto q_head = q.slice(0, q_start, q_end).select(1, head_index);
    auto k_head = k.slice(0, k_start, k_end).select(1, head_index);
    auto v_head = v.slice(0, k_start, k_end).select(1, head_index);

    for (int64_t row_block = 0; row_block < nrow_blocks; ++row_block) {
        const int64_t q_block_start = row_block * block_m;
        const int64_t q_block_end = std::min(q_block_start + block_m, q_len);
        if (q_block_start >= q_len) {
            continue;
        }

        const int64_t q_block_len = q_block_end - q_block_start;
        std::vector<int64_t> valid_col_blocks;
        valid_col_blocks.reserve(nrow_blocks);

        for (int64_t col_block = 0; col_block < sample_mask.size(1); ++col_block) {
            if (sample_mask[row_block][col_block].item<bool>()) {
                valid_col_blocks.push_back(col_block);
            }
        }

        if (valid_col_blocks.empty()) {
            out.slice(0, q_start + q_block_start, q_start + q_block_end)
                .select(1, head_index)
                .zero_();
            continue;
        }

        std::vector<torch::Tensor> valid_k_blocks;
        std::vector<torch::Tensor> valid_v_blocks;
        valid_k_blocks.reserve(valid_col_blocks.size());
        valid_v_blocks.reserve(valid_col_blocks.size());

        for (const int64_t col_block : valid_col_blocks) {
            const int64_t k_block_start = col_block * block_n;
            const int64_t k_block_end = std::min(k_block_start + block_n, k_len);
            if (k_block_start >= k_len) {
                continue;
            }

            valid_k_blocks.push_back(k_head.slice(0, k_block_start, k_block_end).contiguous());
            valid_v_blocks.push_back(v_head.slice(0, k_block_start, k_block_end).contiguous());
        }

        if (valid_k_blocks.empty()) {
            out.slice(0, q_start + q_block_start, q_start + q_block_end)
                .select(1, head_index)
                .zero_();
            continue;
        }

        auto q_block = q_head.slice(0, q_block_start, q_block_end);
        auto k_valid = torch::cat(valid_k_blocks, /*dim=*/0);
        auto v_valid = torch::cat(valid_v_blocks, /*dim=*/0);
        auto logits = torch::matmul(q_block, k_valid.transpose(0, 1));
        auto probs = block_sparse_row_softmax(logits, softmax_scale);
        auto out_block = torch::matmul(probs, v_valid);

        out.slice(0, q_start + q_block_start, q_start + q_block_end)
            .select(1, head_index)
            .copy_(out_block);
    }
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
    // A logical sample is a packed segment [q_start, q_end) / [k_start, k_end) in the flattened tensor.
    // Reading the implementation as a pipeline makes it easier to see the intent:
    //   1) choose the sample slice from cu_seqlens,
    //   2) select the head-specific block mask,
    //   3) build the valid-key mask,
    //   4) compute logits and softmax,
    //   5) aggregate V with the attention weights.
    const int64_t num_samples = cu_seqlens_q.size(0) - 1;
    const int64_t num_heads = q.size(1);

    auto cuq_acc = cu_seqlens_q.accessor<int32_t, 1>();
    auto cuk_acc = cu_seqlens_k.accessor<int32_t, 1>();
    auto head_mask_acc = head_mask_type.accessor<int32_t, 1>();

    for (int64_t b = 0; b < num_samples; ++b) {
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
            TORCH_CHECK(mask_index < base_blockmask.size(1),
                        "mask_type exceeds base_blockmask head count.");
            TORCH_CHECK(mask_index >= 0, "mask_type must be positive.");

            apply_single_head_block_sparse_attention<scalar_t>(
                q,
                k,
                v,
                base_blockmask,
                b,
                h,
                mask_index,
                q_start,
                q_end,
                k_start,
                k_end,
                softmax_scale,
                out);
        }
    }
}

}  // namespace

// CPU block sparse attention forward entry.
// q / k / v: packed Q/K/V tensors with shape [total_tokens, num_heads, head_dim].
// cu_seqlens_q / cu_seqlens_k: cumulative token counts for each sample in the batch, int32.
// head_mask_type: per-head mask selector, value > 0 means select mask_type = head_mask_type[i] from base_blockmask.
// base_blockmask: block mask tensor with shape [batch, num_mask_types, nrow_blocks, ncol_blocks];
//                the selected mask decides which K-blocks are allowed for each Q-block.
// max_seqlen_q / max_seqlen_k: padded maximum query/key length of the batch.
// p_dropout: dropout probability; CPU path only supports 0.0.
// softmax_scale: attention scaling factor, usually 1 / sqrt(head_dim).
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
