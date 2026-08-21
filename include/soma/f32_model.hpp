#pragma once

// Soma — the F32-activation model and forward paths.
//
// The `F32` prefix describes activations and workspace, not weight residency.
// This is the engine's sole execution path: weights may use quantized SIMD
// kernels, routed experts may stream through MemoryHierarchy, and Scheduler
// drives cached ragged batches through KvRows. The teacher-forced whole-sequence
// entry point remains as the transparent conformance reference.
//
// The seam is already real here. Anything family-specific lives behind
// F32Backend in src/soma/arch/, and tools/ci/check_seam.py enforces that this
// header never learns an architecture's name.

#include "soma/arch_ir.hpp"
#include "soma/kv_cache.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/quant_format.hpp"
#include "soma/safetensors.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace soma {

struct PromptCodec;
struct SpeculativeBackend;

/// Resolved weight pointers for one layer. Spans point into the SafeTensors
/// buffers, which must outlive the model.
/// Attention weights, owned by the backend and opaque to the core.
///
/// This is the shape the seam actually requires, and the first architecture
/// could not reveal it. `F32LayerWeights` previously held q/k/v/o_proj and
/// q_norm/k_norm — which is not "a layer's attention weights", it is "a GQA
/// layer's attention weights". MLA has none of those: it carries
/// kv_a_proj_with_mqa, kv_a_layernorm and kv_b_proj instead.
///
/// The easy fix is to widen the struct to the union of both families. That
/// works, and it means every future architecture edits a core type — which is
/// exactly the coupling the seam exists to prevent. So the core holds a void*
/// and a deleter, and never looks inside.
class ArchLayerPayload {
public:
    ArchLayerPayload() = default;

    ~ArchLayerPayload() { reset(); }

    ArchLayerPayload(ArchLayerPayload&& o) noexcept : p_(o.p_), del_(o.del_) {
        o.p_ = nullptr;
        o.del_ = nullptr;
    }

    ArchLayerPayload& operator=(ArchLayerPayload&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            del_ = o.del_;
            o.p_ = nullptr;
            o.del_ = nullptr;
        }
        return *this;
    }

    ArchLayerPayload(const ArchLayerPayload&) = delete;
    ArchLayerPayload& operator=(const ArchLayerPayload&) = delete;

    void reset() noexcept {
        if (p_ && del_) del_(p_);
        p_ = nullptr;
        del_ = nullptr;
    }

    void adopt(void* p, void (*del)(void*)) noexcept {
        reset();
        p_ = p;
        del_ = del;
    }

    /// Only the owning backend calls this, and only with the type it stored.
    template <class T>
    T* as() const noexcept {
        return static_cast<T*>(p_);
    }

    bool empty() const noexcept { return p_ == nullptr; }

private:
    void* p_ = nullptr;
    void (*del_)(void*) = nullptr;
};

struct F32LayerWeights {
    LayerKind kind = LayerKind::Moe;

    // Norms stay fp32 unconditionally (QuantMap::norms), so these remain plain
    // spans — a WeightRef would imply a choice that does not exist.
    //
    // These two are genuinely invariant: every transformer layer normalises
    // before attention and before the FFN, whatever the attention is.
    std::span<const float> input_norm;
    std::span<const float> post_attn_norm;

    /// Everything attention-specific. See ArchLayerPayload.
    ArchLayerPayload attn;

    /// F32 by schema constraint, never quantized (schemas/arch-ir.md §5).
    std::span<const float> router;

    std::vector<WeightRef> expert_gate, expert_up, expert_down;

    WeightRef shared_gate, shared_up, shared_down;
    WeightRef dense_gate, dense_up, dense_down;
};

struct F32Model {
    ArchIr arch;
    SafeTensors weights;
    std::vector<F32LayerWeights> layers;

    /// Architecture-owned model-level weights. Existing backends leave this
    /// empty; v2 block topologies use it for their final stream collapse.
    ArchLayerPayload arch_payload;
    ArchLayerPayload speculative_payload;
    const SpeculativeBackend* speculative_backend = nullptr;

    WeightRef embed; ///< gathered by row; may be resident-quantized
    std::span<const float> out_norm;
    WeightRef out_head; ///< aliases embed when tie_word_embeddings

    /// Owns every quantized tensor, so the WeightRefs above stay valid.
    std::vector<QTensor> quantized;
    QuantMap quant_map{};

    /// When set, routed experts are STREAMED from here instead of read from the
    /// resident table. Both are legitimate residency modes — `resident-only` is a
    /// verdict, not a fallback — so the forward keeps ONE expert loop and only
    /// the accessor differs.
    MemoryHierarchy* streamed_experts = nullptr;

    /// True when the load directory was a Soma container: the resident expert
    /// table is deliberately empty and `streamed_experts` MUST be set before any
    /// forward. Checked at forward entry rather than trusted.
    bool experts_are_streamed = false;

    /// Byte extents of the three sections inside one packed container expert
    /// (schemas/container.md: gate ++ up ++ down). Computed at load, because
    /// gate/up and down commonly differ in dtype.
    std::uint32_t expert_gate_bytes = 0;
    std::uint32_t expert_up_bytes = 0;
    std::uint32_t expert_down_bytes = 0;

    std::uint32_t d_model() const noexcept { return arch.topology.d_model; }

    std::uint32_t vocab() const noexcept { return arch.topology.vocab_size; }
};

/// The four buffers one GLU expert application needs, as spans one worker owns
/// exclusively for the duration of a call.
struct FfnScratch {
    std::span<float> gate, up, act, out;
};

/// Per-forward scratch. Allocated once and reused across layers and steps —
/// the hot path performs no allocation, which is a property worth establishing
/// before there is a hot path to break.
struct F32Workspace {
    std::vector<float> hidden; ///< [T, d_model]
    std::vector<float> residual;
    std::vector<float> normed;
    std::vector<float> attn_out;

    std::vector<float> q, k, v;    ///< [T, n_heads*head_dim] / [T, n_kv*head_dim]
    std::vector<float> attn_heads; ///< [T, n_heads*head_dim] — pre-o_proj
    std::vector<float> scores;     ///< [n_workers * T] — see worker_scores()

    /// Size `scores` for `n_workers` concurrent query positions.
    ///
    /// Attention parallelises over query positions, and each one needs its own
    /// score buffer: `scores` was a single shared [T] array, which is exactly the
    /// kind of scratch that turns a correct parallel loop into a silent race —
    /// the writes land, the reads are plausible, and the output is subtly wrong
    /// in a way no assertion catches.
    void ensure_score_scratch(std::uint32_t n_workers, std::uint32_t n_tokens);

    float* worker_scores(std::uint32_t worker, std::uint32_t n_tokens) noexcept {
        return scores.data() + static_cast<std::size_t>(worker) * n_tokens;
    }

    std::vector<float> router_logits; ///< [T, n_experts], always F32
    std::vector<std::uint32_t> expert_ids;
    std::vector<float> expert_weights;

    std::vector<float> gate_buf, up_buf, act_buf, ffn_out;

    /// Per-worker FFN scratch, laid out as `n_workers` copies of
    /// [gate | up | act | out], each `ffn_stride` wide.
    ///
    /// Exists because the rows that selected one expert are applied
    /// concurrently. The four buffers above are still used by the single-row
    /// paths (shared and dense FFN), where there is nothing to race.
    std::vector<float> ffn_scratch;
    std::uint32_t ffn_stride = 0;

    void ensure_ffn_scratch(std::uint32_t n_workers);
    FfnScratch worker_ffn(std::uint32_t worker) noexcept;

    /// Row-tile scratch for the expert loop: gathered inputs and the three
    /// intermediates, all [tile, width].
    ///
    /// Shared rather than per-worker, because with tiling the parallelism moves
    /// INSIDE the matmul (over weight rows) and the tile buffers are written at
    /// disjoint offsets by construction.
    std::vector<float> tile_x, tile_gate, tile_up, tile_act, tile_out;
    void ensure_tile_scratch(std::uint32_t tile, std::uint32_t d_model, std::uint32_t inter);

    // ── batch-union (CSR) ────────────────────────────────────────────────────
    //
    // The payoff the whole design rests on. Rather than acquiring an expert once
    // per (row, slot), the rows that selected each expert are grouped so it is
    // read ONCE and applied to all of them. Read cost is per-expert and
    // independent of how many rows consume it, which is why aggregate throughput
    // in the disk-bound regime scales better than linearly in batch size.
    //
    // CSR because that is exactly the access pattern: expert union_experts[i]
    // serves rows union_rows[union_offsets[i] .. union_offsets[i+1]).
    std::vector<std::uint32_t> union_experts;
    std::vector<std::uint32_t> union_offsets; ///< size n_unique + 1
    std::vector<std::uint32_t> union_rows;
    std::vector<float> union_weights;
    std::vector<std::uint32_t> union_counts; ///< scratch, size n_experts

    /// Made observable rather than assumed. A ratio near 1.0 means the union is
    /// buying nothing and something upstream is wrong.
    std::uint64_t naive_expert_reads = 0;
    std::uint64_t unique_expert_reads = 0;

    /// Optional per-layer activation tap. Null in every production path.
    ///
    /// Exists because whole-model logit comparison has a resolution limit, and
    /// G4 hit it: four rounds of hypothesise-and-eliminate localised an MLA
    /// defect to "positional" and no further, because the only observable was
    /// the final logits. A per-layer tap turns that into one run.
    ///
    /// Deliberately a C function pointer rather than std::function: this sits in
    /// the forward's hot path, and a null check that the branch predictor
    /// resolves to "never" is the entire cost when it is off.
    struct Sink {
        void (*emit)(void* ctx,
                     std::uint32_t layer,
                     const char* point,
                     const float* data,
                     std::size_t n) = nullptr;
        void* ctx = nullptr;

        void
        operator()(std::uint32_t layer, const char* point, const float* data, std::size_t n) const {
            if (emit) emit(ctx, layer, point, data, n);
        }

        explicit operator bool() const noexcept { return emit != nullptr; }
    };

    Sink sink{};

    /// The layer currently executing, published by the forward so a backend can
    /// label its own taps without the attention signature growing a parameter
    /// that only debug builds would use.
    std::uint32_t current_layer = 0;

    /// Backend-private state that outlives ONE layer but not one forward pass.
    ///
    /// The same opaque idiom `F32LayerWeights::attn` already uses, for the same
    /// reason: the core must not learn what is inside. `ArchLayerPayload` holds a
    /// void* and a deleter, and nothing here inspects it.
    ///
    /// DSA is what needs it. IndexShare means a `full` layer computes a top-k key
    /// selection and the following `shared` layers REUSE it — on GLM-5.2, 57 of 78
    /// layers own no indexer weights and cannot compute attention without a
    /// selection produced by a different layer. That is cross-layer state, and the
    /// workspace is where it belongs: it is per-forward, already threaded through
    /// every layer, and the selection is recomputed each pass rather than carried
    /// between steps, so nothing about it is per-sequence.
    ///
    /// A backend that needs no such state never touches it and pays a null
    /// pointer. Deliberately NOT reset per layer — outliving the layer is the
    /// entire point — but see `reset_arch_state()` for the per-forward boundary,
    /// which matters because a stale selection from a previous prompt would be
    /// silently wrong rather than an error.
    ArchLayerPayload arch_state;

    /// Called by the forward before layer 0. A selection computed for a different
    /// prompt has the wrong length and the wrong contents, and reusing it would
    /// produce plausible logits rather than a failure.
    void reset_arch_state() noexcept { arch_state.reset(); }

    void reserve(const ArchIr& arch, std::uint32_t max_tokens);
};

/// Optional production hidden-state taps requested by a speculative backend.
/// Values are packed [requested_layer][row][d_model] in the same order as
/// `layers`. The core compares numeric layer ids only; it never knows why a
/// backend selected them or which model family consumes them.
struct HiddenStateTaps {
    std::span<const LayerIndex> layers;
    std::vector<float> values;
    std::uint32_t n_rows = 0;
    std::uint32_t d_model = 0;

    std::span<const float> layer(std::size_t ordinal) const noexcept {
        const auto width = static_cast<std::size_t>(n_rows) * d_model;
        if (ordinal >= layers.size() || (ordinal + 1) * width > values.size()) return {};
        return std::span<const float>(values).subspan(ordinal * width, width);
    }
};

/// The architecture seam for the G0 path.
///
/// Only two operations differ by family at this gate: attention (cache shape,
/// qk-norm form, RoPE application) and routing (scoring function, ties,
/// normalization, bias correction). Expert application is invariant given
/// `ArchIr::ffn.activation`, so it stays in core.
/// What a backend needs from the loader in order to bind its OWN tensors.
///
/// The inversion that makes the opaque payload work: the loader knows how to
/// read and quantize a named tensor, the backend knows which names exist. Each
/// keeps its half. Without this the core would need a list of every family's
/// tensor names, which is the same coupling in a different file.
struct LayerBindCtx {
    const SafeTensors* weights = nullptr;
    const QuantMap* quant = nullptr;
    std::vector<QTensor>* owner = nullptr; ///< keeps quantized tensors alive
    LayerIndex layer = 0;
    std::string prefix; ///< optional exact layer prefix for auxiliary graphs

    /// "self_attn.q_proj.weight" -> model.layers.{layer}.self_attn.q_proj.weight
    std::string name(const char* suffix) const;
};

struct ModelBindCtx {
    const SafeTensors* weights = nullptr;
    const QuantMap* quant = nullptr;
    std::vector<QTensor>* owner = nullptr;
};

/// fp32 bind, for norms.
Status bind_layer_f32(const LayerBindCtx& ctx,
                      const char* suffix,
                      std::span<const float>& out,
                      bool optional = false);

/// Quantized-by-role bind, for projections.
Status bind_layer_weight(const LayerBindCtx& ctx,
                         const char* suffix,
                         TensorRole role,
                         WeightRef& out,
                         bool optional = false);

Status bind_model_f32(const ModelBindCtx& ctx,
                      const char* name,
                      std::span<const float>& out,
                      bool optional = false);

Status bind_model_weight(const ModelBindCtx& ctx,
                         const char* name,
                         TensorRole role,
                         WeightRef& out,
                         bool optional = false);

struct F32Backend {
    const char* name = nullptr;

    /// Optional model-owned prompt/completion protocol. Null preserves the
    /// generic role flattening used by all existing families.
    const PromptCodec* prompt_codec = nullptr;

    /// Bind this layer's attention tensors into a payload the backend owns.
    ///
    /// Called once per layer at load. The core stores the result and never
    /// inspects it; only this backend's own attention functions read it back.
    StatusCode (*bind_layer)(const ArchIr& arch,
                             const LayerBindCtx& ctx,
                             ArchLayerPayload& out) noexcept = nullptr;

    StatusCode (*bind_model)(const ArchIr& arch,
                             const ModelBindCtx& ctx,
                             ArchLayerPayload& out) noexcept = nullptr;

    /// Optional block lifecycle. Null preserves the ordinary single-stream
    /// residual path exactly; v2 architectures can own residual topology without
    /// adding family checks to the core loop.
    StatusCode (*begin_forward)(const ArchIr& arch,
                                const ArchLayerPayload& model_payload,
                                const TokenId* tokens,
                                std::uint32_t n_tokens,
                                F32Workspace& ws,
                                float* hidden) noexcept = nullptr;
    StatusCode (*pre_attention)(const ArchIr& arch,
                                const F32LayerWeights& w,
                                std::uint32_t n_tokens,
                                F32Workspace& ws,
                                float* hidden) noexcept = nullptr;
    StatusCode (*merge_attention)(const ArchIr& arch,
                                  const F32LayerWeights& w,
                                  const float* branch,
                                  std::uint32_t n_tokens,
                                  F32Workspace& ws,
                                  float* hidden) noexcept = nullptr;
    StatusCode (*pre_ffn)(const ArchIr& arch,
                          const F32LayerWeights& w,
                          std::uint32_t n_tokens,
                          F32Workspace& ws,
                          float* hidden) noexcept = nullptr;
    StatusCode (*merge_ffn)(const ArchIr& arch,
                            const F32LayerWeights& w,
                            const float* branch,
                            std::uint32_t n_tokens,
                            F32Workspace& ws,
                            float* hidden) noexcept = nullptr;

    /// Optional architecture-owned representation exported after a block.
    /// Null copies the core's ordinary hidden rows. Multi-stream topologies use
    /// this to expose the exact conditioning state consumed by auxiliary
    /// decoders without teaching the core how their residual streams are laid
    /// out or collapsed.
    StatusCode (*export_layer_hidden)(const ArchIr& arch,
                                      LayerIndex layer,
                                      std::uint32_t n_tokens,
                                      const F32Workspace& ws,
                                      const float* hidden,
                                      float* out) noexcept = nullptr;
    StatusCode (*end_forward)(const ArchIr& arch,
                              const ArchLayerPayload& model_payload,
                              std::uint32_t n_tokens,
                              F32Workspace& ws,
                              float* hidden) noexcept = nullptr;

    /// x:[T, d_model] -> out:[T, d_model]. Causal, no cache.
    StatusCode (*attention)(const ArchIr& arch,
                            const F32LayerWeights& w,
                            const float* x,
                            std::uint32_t n_tokens,
                            F32Workspace& ws,
                            float* out) noexcept = nullptr;

    /// The shape of both cache planes.
    ///
    /// The cache's SHAPE is the backend's knowledge, and core had it hardcoded as
    /// `n_kv_heads * head_dim` — GQA's formula, applied to every family. MLA does
    /// not store per-head K and V at all; it stores a `kv_lora_rank` latent plus
    /// one shared RoPE segment, which is the entire reason the architecture
    /// exists. That mismatch is why the cached-decode path was a stub: there was
    /// no width it could have been correct at.
    ///
    /// Same argument as `AttentionBackend::weight_bytes_per_layer` (roadmap D16),
    /// and the seam check would refuse the alternative — a family branch in core.
    ///
    /// Returns BOTH widths, so a family that stores no V plane can say so — one
    /// number could not express it, and the cache allocated a full second plane
    /// for MLA on the strength of that silence.
    ///
    /// Null means "use the GQA default" (both planes `n_kv_heads * head_dim`), so
    /// a backend need not implement it to keep working.
    KvGeometry (*kv_geometry)(const ArchIr& arch) noexcept = nullptr;

    /// The batched-decode form: one KvRow per row, each with its own cache,
    /// position and visible length.
    ///
    /// A SECOND entry point rather than a flag on the first, because the two
    /// differ in what they are allowed to assume. `attention` owns the whole
    /// sequence and may exploit that; this one owns exactly one position per row
    /// and must not. Collapsing them would put a "do I have a cache?" branch
    /// inside the hot loop of every future backend.
    StatusCode (*attention_kv)(const ArchIr& arch,
                               const F32LayerWeights& w,
                               const float* x,
                               std::uint32_t n_rows,
                               LayerIndex layer,
                               const KvRow* rows,
                               F32Workspace& ws,
                               float* out) noexcept = nullptr;

    /// logits:[T, n_experts] F32 -> ids/weights:[T, top_k].
    ///
    /// The function that decides WHICH EXPERTS FIRE. Its input is F32
    /// unconditionally; see schemas/arch-ir.md §5.
    ///
    /// Takes the layer's weights, not just the logits, because a router can have
    /// PARAMETERS BEYOND THE GATE MATRIX. DeepSeek-V3's `noaux_tc` scoring adds a
    /// per-expert correction bias that participates in selection; the original
    /// signature could not express it, and no amount of care in the backend would
    /// have worked around a missing argument. The layer weights carry the
    /// backend's own payload, so a router's extra tensors travel with it.
    StatusCode (*route)(const ArchIr& arch,
                        const F32LayerWeights& w,
                        const TokenId* input_tokens,
                        const float* logits,
                        std::uint32_t n_tokens,
                        std::uint32_t* out_ids,
                        float* out_weights) noexcept = nullptr;
};

struct SpeculativeProposal {
    std::vector<TokenId> tokens;
    std::vector<float> confidence;
    /// Optional draft-graph activation tap. Null in production; conformance
    /// tools use the same callback shape as the target model workspace.
    F32Workspace::Sink sink{};
};

/// Optional speculative model seam. The core owns scheduling and target KV
/// transactions; this descriptor owns draft weights, draft cache semantics, and
/// proposal construction. A null descriptor is ordinary autoregressive serving.
struct SpeculativeBackend {
    const char* name = nullptr;

    StatusCode (*bind_model)(F32Model& model,
                             const std::string& model_dir) noexcept = nullptr;
    StatusCode (*start_runtime)(F32Model& model,
                                const std::string& model_dir,
                                std::uint64_t expert_cache_bytes) noexcept = nullptr;
    StatusCode (*open_sequence)(const F32Model& model,
                                std::uint32_t max_context,
                                ArchLayerPayload& state) noexcept = nullptr;

    StatusCode (*observe_target)(const F32Model& model,
                                 const ArchLayerPayload& payload,
                                 ArchLayerPayload& state,
                                 const HiddenStateTaps& taps,
                                 std::uint32_t first_row,
                                 std::uint32_t row_count,
                                 std::uint32_t first_position) noexcept = nullptr;

    StatusCode (*propose)(const F32Model& model,
                          const ArchLayerPayload& payload,
                          ArchLayerPayload& state,
                          TokenId anchor,
                          std::uint32_t max_tokens,
                          float confidence_threshold,
                          SpeculativeProposal& out) noexcept = nullptr;

    Status (*serialize_state)(const F32Model& model,
                              const ArchLayerPayload& state,
                              std::vector<std::byte>& out) = nullptr;
    Status (*restore_state)(const F32Model& model,
                            std::span<const std::byte> payload,
                            ArchLayerPayload& state) = nullptr;
};

/// Resolves once, at load. A switch on family anywhere in a loop is a seam
/// violation; this is the only permitted one.
const F32Backend* resolve_f32_backend(const ArchIr& arch) noexcept;

/// Resolves from an optional architecture descriptor once at load. Core loops
/// call only the returned function table and contain no model-specific branch.
const SpeculativeBackend* resolve_speculative_backend(const ArchIr& arch) noexcept;

/// Load config.json + safetensors from a fixture or checkpoint directory.
///
/// `quant` selects precision PER TENSOR ROLE; the default map is all-F32, which
/// is the G0 path. Requantization happens here rather than offline so the fp32
/// and quantized runs share one loader and one forward.
Status load_f32_model(const std::string& dir, F32Model& out, const QuantMap& quant = {});

/// One routed expert, in whichever residency mode the model is in.
///
/// `pin` is empty for a resident expert and holds the cache borrow for a streamed
/// one — so the three WeightRefs stay valid exactly as long as the handle does.
/// Returning them together is what makes the lifetime impossible to get wrong.
struct ExpertHandle {
    WeightRef gate, up, down;
    MemoryHierarchy::ExpertRef pin;

    bool valid() const noexcept { return !gate.empty(); }
};

ExpertHandle acquire_expert(const F32Model& model, LayerIndex layer, ExpertId expert);

/// Group this step's (row, expert) selections by expert, into CSR form.
///
/// Experts are emitted in ASCENDING id order. That is a deterministic ordering,
/// which matters because the order experts are applied to a row changes the
/// float accumulation order — and therefore the low bits of the result. Any
/// stable order would do; an unstable one would make output depend on hash
/// iteration order.
void build_expert_union(std::uint32_t n_rows,
                        std::uint32_t top_k,
                        std::uint32_t n_experts,
                        const std::uint32_t* ids,
                        const float* weights,
                        F32Workspace& ws);

/// Teacher-forced forward over the whole sequence.
/// out_logits is resized to [n_tokens * vocab_size].
Status forward_f32(const F32Model& model,
                   std::span<const TokenId> tokens,
                   F32Workspace& ws,
                   std::vector<float>& out_logits);

/// One step of a RAGGED BATCH: `n_rows` rows, each from a possibly different
/// sequence, each carrying its own KV cache and absolute position.
///
/// This is the forward the scheduler drives, and the reason the batch union
/// finally earns its keep in production rather than only in a benchmark: the
/// rows in one call come from CONCURRENT SEQUENCES, so the expert sets being
/// unioned are genuinely independent draws rather than adjacent tokens of one
/// prompt.
///
/// out_logits is resized to [n_rows * vocab_size].
Status forward_step_f32(const F32Model& model,
                        std::span<const TokenId> tokens,
                        std::span<const KvRow> rows,
                        F32Workspace& ws,
                        std::vector<float>& out_logits,
                        HiddenStateTaps* taps = nullptr);

/// Greedy continuation from `prefix`, recomputing the full prefix each step.
Status generate_greedy_f32(const F32Model& model,
                           std::span<const TokenId> prefix,
                           std::uint32_t n_new,
                           F32Workspace& ws,
                           std::vector<TokenId>& out_tokens);

} // namespace soma
