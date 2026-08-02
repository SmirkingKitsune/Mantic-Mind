#pragma once

// Soma — the three state-ownership tiers.
//
// Concurrency correctness in this engine reduces to knowing who owns what.
// There are exactly three tiers and the boundaries between them are the design:
//
//   model  immutable after load          no locking, shared by every thread
//   seq    per-sequence                  owned by one sequence
//   exec   per-step scratch              mutex-held for the step
//
// ONLY `exec` and the expert cache need locking. That is the whole story — and
// it depends entirely on `model` being genuinely immutable. The moment
// something mutates it at runtime (a lazily-materialized tensor, a cached
// transpose), the lock-free read collapses and the bug will present as
// nondeterministic garbage under concurrency. Admission is therefore
// responsible for every transformation ahead of time, including transposing
// fused 3D expert tensors. Never at runtime.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace soma {

struct ArchBackend;
class ExpertStore;
class CompiledTokenizer;
struct KernelTable;

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 — model. Immutable after load.
// ─────────────────────────────────────────────────────────────────────────────

/// A resident (never-evicted) tensor: attention projections, norms, embeddings,
/// router weights, shared experts. The static partition's dense half.
struct DenseTensor {
    CByteSpan data;
    DType dtype = DType::F32;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

struct DenseLayer {
    DenseTensor attn_norm;
    DenseTensor attn_qkv; ///< layout is the attention backend's business
    DenseTensor attn_out;
    DenseTensor ffn_norm;
    DenseTensor router;      ///< F32, always
    DenseTensor router_bias; ///< present iff RouterSpec::bias_correction
    DenseTensor shared_gate;
    DenseTensor shared_up;
    DenseTensor shared_down;
    DenseTensor dense_gate; ///< used when LayerKind::Dense
    DenseTensor dense_up;
    DenseTensor dense_down;
};

/// Read-only for the entire lifetime of a served model.
///
/// Deliberately holds no mutex, no atomic, and no mutable member. If one is ever
/// needed, that is a signal the design has drifted — not a reason to add it.
class ModelState {
public:
    ModelState() = default;
    ModelState(const ModelState&) = delete;
    ModelState& operator=(const ModelState&) = delete;
    ~ModelState();

    const ArchIr& arch() const noexcept;
    const ArchBackend& backend() const noexcept;
    const CompiledTokenizer& tokenizer() const noexcept;
    const KernelTable& kernels() const noexcept;
    ExpertStore& experts() const noexcept;

    const DenseLayer& layer(LayerIndex index) const noexcept;
    const DenseTensor& embedding() const noexcept;
    const DenseTensor& output_norm() const noexcept;
    const DenseTensor& output_head() const noexcept;

    /// Bytes the dense/resident half occupies. Feeds the plan document.
    std::uint64_t resident_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Load a converted model directory. Runs the backend's prepare_weights() hook
/// (MLA absorption; no-op for GQA) exactly once, here.
Status load_model(const std::string& model_dir, std::unique_ptr<ModelState>& out);

// ─────────────────────────────────────────────────────────────────────────────
// Tier 2 — seq. Per-sequence, owned by one sequence.
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque per-sequence KV bytes.
///
/// The core allocates and sizes this; only the attention backend knows the
/// layout inside. An interface that exposed `Tensor& k, Tensor& v` would be
/// GQA-shaped and MLA would have to lie to it, so nothing here names K or V.
struct KvRegion {
    ByteSpan bytes;
    std::uint32_t capacity_tokens = 0;
    std::uint32_t length_tokens = 0;
    std::uint32_t persist_format_id = 0; ///< from AttentionBackend
};

/// Reproducibility mode. See docs/architecture.md §10.
///
/// `Batched` tokens remain the argmax of a VALID forward, so quality holds —
/// but the exact stream can depend on who else is on the server, because
/// quantized integer kernels round differently at different shapes.
enum class Determinism : std::uint8_t {
    Batched = 0, ///< default: batch freely
    Strict,      ///< serialized single-row path, single-row kernel family
};

struct SamplerState {
    float temperature = 0.7f;
    float top_p = 0.9f;
    std::int32_t top_k = -1;
    float min_p = -1.0f;
    float presence_penalty = 0.0f;
    float repeat_penalty = -1.0f;
    std::uint64_t rng_state = 0;
};

class GrammarState;
class StopCondition;

struct SeqState {
    SeqId id = kInvalidSeq;
    KvRegion kv{};
    std::uint32_t position = 0;
    SamplerState sampler{};
    Determinism determinism = Determinism::Batched;

    std::vector<TokenId> draft_window;
    std::unique_ptr<GrammarState> grammar;
    std::unique_ptr<StopCondition> stop;

    bool prefill_complete = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tier 3 — exec. Per-step scratch, sized once for max_batch.
// ─────────────────────────────────────────────────────────────────────────────

/// One row of the ragged batch. Decode rows and prefill rows are just rows.
struct BatchRow {
    SeqState* seq = nullptr;
    TokenId token = 0;
    std::uint32_t position = 0;
    bool is_prefill = false;
};

struct SeqBatch {
    std::span<BatchRow> rows;

    std::uint32_t n_rows() const noexcept;
};

/// The batch-union: each unique expert appears once, with the list of rows that
/// routed to it.
///
/// Stored CSR because that is precisely the access pattern: read expert
/// `unique_experts[i]` once, then apply it to rows
/// `row_indices[row_offsets[i] .. row_offsets[i+1])`. Read cost is per-expert
/// and independent of how many rows consume it, which is the entire reason
/// aggregate throughput scales better than linearly in concurrency.
struct ExpertUnion {
    std::span<const ExpertId> unique_experts;
    std::span<const std::uint32_t> row_offsets; ///< size == unique_experts.size() + 1
    std::span<const std::uint32_t> row_indices;
    std::span<const float> row_weights;

    std::uint32_t n_unique() const noexcept;
};

/// Allocated once at engine start, sized for max_batch. The hot path performs no
/// allocation; anything that would need to grow here is a sizing bug, not an
/// occasion to call the allocator.
class ExecScratch {
public:
    ExecScratch();
    ExecScratch(const ExecScratch&) = delete;
    ExecScratch& operator=(const ExecScratch&) = delete;
    ~ExecScratch();

    Status reserve(const ArchIr& arch, std::uint32_t max_batch);

    std::span<float> hidden() noexcept; ///< [max_batch × d_model]
    std::span<float> residual() noexcept;
    std::span<float> ffn_workspace() noexcept;

    /// Router logits are F32 unconditionally — quantizing them changes which
    /// experts fire, which is a semantic change (schemas/arch-ir.md §5).
    std::span<float> router_logits() noexcept; ///< [max_batch × n_experts]

    /// Build the batch-union from this step's router output.
    Status build_expert_union(LayerIndex layer, std::uint32_t n_rows, ExpertUnion& out) noexcept;

    std::uint32_t max_batch() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace soma
