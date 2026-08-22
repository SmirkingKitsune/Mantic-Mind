#pragma once

// Soma — the normalized in-memory architecture description.
//
// This is a DESCRIPTION of an architecture, not an implementation of one. It is
// plain data: the core reads it to size buffers and select a backend, and the
// backend reads it for its own parameters. Nothing here executes.
//
// Canonical spec, worked examples, and the verdict function: schemas/arch-ir.md

#include "soma/quant.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace soma {

// Version 1 remains the default for every architecture that predates the
// block-level execution contract below.  DeepSeek-V4 is the first v2 adapter.
// Keeping v1 as the default is intentional: merely compiling a newer Soma must
// not change the arch_hash (and therefore invalidate KV checkpoints) of an
// existing GQA/MLA container.
inline constexpr std::uint32_t kArchIrSchemaVersion = 1;
inline constexpr std::uint32_t kArchIrSchemaVersionV2 = 2;

/// Selects the attention backend. The core switches on this exactly once, at
/// load, to resolve a descriptor — never in a loop.
///
/// `Unknown` is the zero value and the default everywhere. There is deliberately
/// no default *family*: a default-constructed spec that claimed to be GQA would
/// be both wrong and a quiet bias toward the first architecture shipped, which
/// is the exact failure the two-family co-design exists to prevent.
enum class AttentionFamily : std::uint8_t {
    Unknown = 0,
    Mha,
    Gqa,
    Mla,
    MlaDsa,
    CompressedSparse,

    /// Hybrid stack: a minority of full-attention layers (MLA) interleaved with
    /// a majority of linear-attention layers carrying a fixed-size recurrent
    /// state instead of a per-token cache.
    ///
    /// Appended rather than inserted. The value is not what identifies the
    /// family in `arch_hash` — `to_string` is — but the enum is serialized
    /// nowhere it can be reordered safely, and appending costs nothing.
    MlaKda,
};

/// Whether a layer attends over a cache or carries a recurrent state.
///
/// Deliberately family-neutral: it says what the layer COSTS, which is the only
/// thing the planner asks. Length-n_layers and authoritative, for the same
/// reason `Topology::layer_kinds` and `DsaSpec::layer_kinds` are — the upstream
/// config states the layer sets as two explicit 1-BASED lists, and a stride
/// re-derived downstream is a disagreement waiting to happen.
enum class AttnLayerKind : std::uint8_t { Full = 0, Linear };

enum class LayerKind : std::uint8_t { Dense = 0, Moe };

/// Q/K normalization. NOT a boolean — the two forms differ in what they
/// normalize over, and a flag would silently conflate them.
///
///   PerHead    weight has head_dim elements, applied independently per head
///              (Qwen3-MoE: q_norm is [16] with head_dim 16)
///   FullWidth  weight spans n_heads * head_dim, applied to the whole
///              projection (OLMoE: q_norm is [64] with 4 heads x 16)
///
/// Both report `"qk_norm": true` upstream. Reading it as one bit produces a
/// model that runs, converges to plausible logits, and is wrong.
enum class QkNormKind : std::uint8_t { None = 0, PerHead, FullWidth };
enum class ScoreFn : std::uint8_t { Softmax = 0, Sigmoid, SqrtSoftplus };
/// Gated FFN nonlinearity.
///
/// `Situ` is Moonshot's: `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)`,
/// parameterized by `FfnSpec::situ_beta` and `situ_linear_beta`. It is a math
/// primitive selected by a family, not an architecture — same category as
/// SwiGlu, and core kernels may name it.
enum class Activation : std::uint8_t { SwiGlu = 0, GeGlu, Relu2, Situ };

/// What the checkpoint accepts as INPUT. Not a serving capability.
///
/// A vision-capable checkpoint served text-only is a real and useful thing; a
/// vision-capable checkpoint SILENTLY served text-only is a model answering
/// about an image it never received. Recording the declaration is what lets the
/// plan say which one is happening.
enum class Modality : std::uint8_t { Text = 0, VisionText };
enum class RopeScalingKind : std::uint8_t { None = 0, Linear, Ntk, Yarn };
enum class ExpertLayout : std::uint8_t { InterleavedGateUpDown = 0 };
enum class TokenizerKind : std::uint8_t { Bpe = 0, Unigram };

struct RopeScaling {
    RopeScalingKind kind = RopeScalingKind::None;
    float factor = 1.0f;
    std::uint32_t original_max_position = 0;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    float mscale = 1.0f;
    float mscale_all_dim = 1.0f;
};

struct RopeConfig {
    float theta = 10000.0f;
    std::uint32_t partial_dim = 0; ///< 0 = full head_dim
    bool interleaved = false;
    RopeScaling scaling{};
};

/// Present iff `AttentionSpec::family` is Mla or MlaDsa.
struct MlaSpec {
    std::uint32_t kv_lora_rank = 0;
    std::uint32_t q_lora_rank = 0; ///< 0 = no Q down-projection
    std::uint32_t qk_nope_head_dim = 0;
    std::uint32_t qk_rope_head_dim = 0;
    std::uint32_t v_head_dim = 0;

    /// Move the KV up-projection to the query side, so decode never materializes
    /// full K/V. There is no GQA analogue, which is why this lives here.
    ///
    /// PER STEP, not at load. This said "at load", which was the intent when the
    /// field was written and not what absorption wanted once measured: folding at
    /// load means keeping a transposed fp32 copy of the up-projection resident —
    /// 1.96 GB on GLM-5.2 — to save arithmetic that was never the bottleneck. See
    /// `arch::mla::f32_attention_kv`.
    ///
    /// False selects the expanded form, which is kept as the reference the
    /// absorbed one is checked against; `soma_decode_kv_g4` runs both.
    bool absorb_weights = true;

    /// No positional encoding on this family's full-attention layers.
    ///
    /// NOT the same as `qk_rope_head_dim == 0`. The rope-width slice still
    /// exists — it is projected, concatenated, cached and attended over — it is
    /// simply never rotated, because position enters the stack through the
    /// linear layers instead. So the SHAPES are the ordinary MLA shapes and only
    /// the rotation disappears.
    ///
    /// Reading `nope` as "drop the rope slice" would narrow every K by
    /// qk_rope_head_dim and produce a cache the weights do not fit. Reading
    /// `rope.theta` as meaningful would rotate a model that must not be
    /// rotated — finite logits, exact at position 0, wrong everywhere else,
    /// which is the D-class failure this codebase keeps re-finding.
    bool nope = false;

    /// Sigmoid output gate on the attention block: `o * sigmoid(g_proj(x))`.
    /// A full `d_model x (n_heads * v_head_dim)` matrix, so it is resident
    /// weight the planner must charge, not a flag.
    bool output_gate = false;
};

/// Present iff `AttentionSpec::family` is MlaKda.
///
/// Gated delta-rule linear attention. The property that matters to the planner
/// is that its per-sequence state is CONSTANT in context: a
/// `n_heads x head_dim x head_dim` recurrent matrix plus a short causal
/// convolution window, and nothing that grows with token count.
///
/// That is not a detail — it inverts the usual cache arithmetic. A stack where
/// most layers are linear pays a large fixed state and a small per-token cache,
/// so the crossover with an all-full-attention stack sits at a few thousand
/// tokens and the long-context end is where it wins.
struct KdaSpec {
    std::uint32_t n_heads = 0;
    std::uint32_t head_dim = 0;

    /// Short causal depthwise convolution applied to each of q/k/v. The carried
    /// state is `kernel - 1` positions wide, per projection.
    std::uint32_t conv_kernel = 0;

    /// Clamp on the log-space forget gate. Zero and "absent" are NOT the same
    /// thing here — an absent bound means no clamp — so `has_gate_bound`
    /// carries the distinction rather than overloading the value.
    bool has_gate_bound = false;
    float gate_lower_bound = 0.0f;

    /// Full-rank output gate (`d_model x n_heads*head_dim`) versus the low-rank
    /// pair through `head_dim`. A ~56x difference in that tensor's size, so it
    /// is a sizing input and not a style note.
    bool full_rank_gate = false;

    /// Length == n_layers when non-empty. Empty for every family but this one.
    std::vector<AttnLayerKind> layer_kinds;

    std::uint32_t n_linear_layers() const noexcept {
        std::uint32_t n = 0;
        for (const auto k : layer_kinds)
            if (k == AttnLayerKind::Linear) ++n;
        return n;
    }
    std::uint32_t n_full_layers() const noexcept {
        std::uint32_t n = 0;
        for (const auto k : layer_kinds)
            if (k == AttnLayerKind::Full) ++n;
        return n;
    }
};

/// Which layers own a sparse-attention indexer.
///
/// `Full` computes an index; `Shared` reuses the one the nearest preceding
/// `Full` layer produced. That reuse is IndexShare, and it is the reason this has
/// to be described per layer rather than as a single flag: on GLM-5.2, 57 of 78
/// layers carry no indexer weights at all and cannot compute attention without
/// state from a different layer.
enum class IndexerKind : std::uint8_t { None = 0, Full, Shared };

/// Present iff `AttentionSpec::family` is MlaDsa.
///
/// DSA is MLA plus a learned sparse key selector: instead of attending to every
/// cached token, each query attends to the `top_k` keys an indexer scores highest.
/// The KV cache is unaffected — every token is still stored, and `MlaSpec`
/// still describes it — so this adds no cache arithmetic, only selection.
struct DsaSpec {
    /// How many keys survive selection. **The number that decides whether a test
    /// means anything**: with fewer tokens in context than this, top-k selects
    /// everything and the sparse path is bit-identical to dense attention.
    /// Measured on the tiny fixture — positions below it match dense to 0.0 while
    /// positions above differ by 0.48 max|logit|.
    std::uint32_t index_topk = 0;

    std::uint32_t n_index_heads = 0;
    std::uint32_t index_head_dim = 0;

    /// How often a `Full` layer recurs. Informational — `layer_kinds` below is
    /// authoritative, for the same reason `Topology::layer_kinds` is: a stride
    /// plus an offset is a re-derivation waiting to disagree with the weights.
    std::uint32_t index_freq = 0;

    /// Length == n_layers when non-empty. Empty for every family but DSA.
    std::vector<IndexerKind> layer_kinds;

    /// Layers that own an indexer, i.e. `Full` count. Needed by the planner,
    /// which sizes a per-layer average and has no layer index to consult.
    std::uint32_t n_full_layers() const noexcept {
        std::uint32_t n = 0;
        for (const auto k : layer_kinds)
            if (k == IndexerKind::Full) ++n;
        return n;
    }
};

/// Hybrid compressed/sparse attention used by v2 Architecture IR models.
///
/// Unlike DSA, compression changes both the cache representation and the set of
/// key positions.  `compress_ratios` is therefore authoritative per layer and
/// is part of the architecture identity.
struct CompressedAttentionSpec {
    std::uint32_t q_lora_rank = 0;
    std::uint32_t rope_head_dim = 0;
    std::uint32_t o_groups = 0;
    std::uint32_t o_lora_rank = 0;
    std::uint32_t index_n_heads = 0;
    std::uint32_t index_head_dim = 0;
    std::uint32_t index_topk = 0;
    float compress_rope_theta = 10000.0f;
    /// Source checkpoint activation quant/dequant operations. These are model
    /// semantics for the production FP8/FP4 checkpoint, not a request for a
    /// native low-precision runtime kernel: the reference backend emulates them
    /// in fp32. Dense tiny fixtures can turn them off explicitly to match native
    /// Transformers' `from_config` execution.
    bool semantic_fp8_quant_dequant = true;
    bool semantic_fp4_quant_dequant = true;
    std::vector<std::uint32_t> compress_ratios;
};

/// Manifold-constrained residual mixing.  A multiplier of one is the ordinary
/// residual stream and requires no block hooks.
struct HyperConnectionSpec {
    std::uint32_t multiplier = 1;
    std::uint32_t sinkhorn_iters = 0;
    float eps = 1e-6f;
};

/// Optional DeepSeek speculative head carried by a converted container.
///
/// The upstream V4 config declares these values even when its `mtp.*` tensors
/// were deliberately omitted during conversion. `source_declared` records that
/// fact; `present` is the serving capability and becomes true only after
/// container metadata proves all DSpark payloads were converted. Keeping the
/// two separate prevents an old autoregressive-only container from advertising
/// a draft model it cannot load.
enum class SpeculativeMethod : std::uint8_t { None = 0, DSpark };

struct SpeculativeSpec {
    SpeculativeMethod method = SpeculativeMethod::None;
    bool source_declared = false;
    bool present = false;
    std::uint32_t n_layers = 0;
    std::vector<LayerIndex> target_layer_ids;
    std::uint32_t trained_block_size = 0;
    TokenId noise_token_id = 0;
    std::uint32_t markov_rank = 0;
    bool confidence_head = false;

    /// Exact converted payload accounting. Routed experts are streamed through
    /// the ordinary expert cache; resident tensors and persistent BF16 draft KV
    /// are admitted separately.
    std::uint64_t routed_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t expert_bytes = 0;
    std::uint64_t kv_bytes_per_sequence = 0;
    float profiled_speedup = 0.0f; ///< host measurement; excluded from arch_hash
};

struct Topology {
    std::uint32_t n_layers = 0;
    std::uint32_t d_model = 0;
    std::uint32_t vocab_size = 0;

    /// Length == n_layers, materialized at admission. Upstream configs express
    /// "which layers are MoE" three different ways (decoder_sparse_step,
    /// moe_layer_freq, mlp_only_layers); resolving that here means the core
    /// never has to.
    std::vector<LayerKind> layer_kinds;

    std::uint32_t first_k_dense = 0; ///< informational; layer_kinds is authoritative
    LayerIndex draft_layer = kInvalidLayer;
    bool tie_word_embeddings = false;
    std::uint32_t max_position_embeddings = 0;
    std::vector<std::uint32_t> eos_token_ids;
};

struct AttentionSpec {
    AttentionFamily family = AttentionFamily::Unknown;
    std::uint32_t n_heads = 0;
    std::uint32_t n_kv_heads = 0;
    std::uint32_t head_dim = 0;
    QkNormKind qk_norm = QkNormKind::None;
    std::uint32_t sliding_window = 0; ///< 0 = full attention
    bool bias = false;
    RopeConfig rope{};
    MlaSpec mla{}; ///< meaningful only for Mla/MlaDsa
    DsaSpec dsa{}; ///< meaningful only for MlaDsa
    KdaSpec kda{}; ///< meaningful only for MlaKda
    CompressedAttentionSpec compressed{}; ///< CompressedSparse only
};

struct RouterSpec {
    std::uint32_t n_experts = 0;
    std::uint32_t top_k = 0;
    ScoreFn score_fn = ScoreFn::Softmax;
    bool normalize_topk = false;
    float routed_scaling_factor = 1.0f;
    bool bias_correction = false;
    std::uint32_t n_groups = 1;
    std::uint32_t topk_group = 1;
    std::uint32_t n_shared_experts = 0;
    std::uint32_t n_hash_layers = 0;
};

struct FfnSpec {
    Activation activation = Activation::SwiGlu;
    bool has_gate = true;
    std::uint32_t expert_intermediate = 0;
    std::uint32_t dense_intermediate = 0;
    std::uint32_t shared_intermediate = 0;
    float swiglu_limit = 0.0f;
    ExpertLayout expert_layout = ExpertLayout::InterleavedGateUpDown;

    /// Width the ROUTED experts operate at, when it is not `d_model`.
    ///
    /// Zero means "the residual width", which is what every family before this
    /// one does. A latent MoE instead projects the residual DOWN into a narrower
    /// space, routes and runs the experts entirely inside it, and projects the
    /// combined result back up.
    ///
    /// **This is the single most consequential number in the whole IR for a
    /// streaming engine, because `bytes_per_token` is linear in it.** The
    /// planner sized an expert as `d_model x expert_intermediate` for five
    /// families where that was correct by coincidence — no earlier family had a
    /// latent MoE. Applied to one that does, it overcharges every expert by
    /// `d_model / routed_expert_hidden` and the error lands squarely on the
    /// quantity the verdict is computed from. Use `ArchIr::routed_expert_width()`
    /// rather than reading this field directly, so the "0 means d_model" rule
    /// exists in one place.
    ///
    /// The two projections it implies are RESIDENT per MoE layer and are charged
    /// as such: they are dense, they are read every token, and at
    /// `2 * d_model * routed_expert_hidden` per layer they are not a rounding
    /// error.
    std::uint32_t routed_expert_hidden = 0;

    /// RMSNorm on the combined expert output before the up-projection. Only
    /// meaningful when `routed_expert_hidden` is set.
    bool routed_expert_norm = false;

    /// `Activation::Situ` parameters. Ignored for every other activation.
    float situ_beta = 1.0f;

    /// Zero means the linear half is NOT transformed — absent, not "beta 0",
    /// which would collapse the term to zero and silence the up projection
    /// entirely.
    float situ_linear_beta = 0.0f;
};

/// Softmax-weighted mixing over periodically-snapshotted residuals.
///
/// Every `block_size`-th layer pushes a copy of the residual stream onto a
/// per-token stack; each layer then mixes over that stack with learned scores.
/// Zero — the default and every prior family — is the ordinary residual stream
/// and implies no stack, no per-layer projections, and no extra activation
/// memory.
///
/// Distinct from `HyperConnectionSpec`, which is Sinkhorn-normalized mixing over
/// a widened residual. Both "mix the residual", and folding them into one field
/// would mean a model executing the wrong one of two real mechanisms.
struct BlockResidualSpec {
    std::uint32_t block_size = 0;

    /// Snapshots a stack accumulates over `n_layers`. Derived, not read: it is
    /// what the activation footprint scales with.
    std::uint32_t n_blocks(std::uint32_t n_layers) const noexcept {
        if (block_size == 0) return 0;
        return (n_layers + block_size - 1) / block_size;
    }
};

/// What the checkpoint declares it accepts, and the tower that would serve it.
///
/// Sizes are recorded so a plan can state the cost of the half it is NOT
/// serving. Refusing to convert is a decision for admission; describing the
/// model honestly is this struct's only job.
struct ModalitySpec {
    Modality modality = Modality::Text;
    std::uint32_t media_placeholder_token_id = 0;
    std::uint32_t vision_layers = 0;
    std::uint32_t vision_hidden = 0;
    std::uint32_t vision_patch_size = 0;
};

/// How this family spells its tensors.
///
/// Part of the adapter's job, and genuinely IR data: Mixtral calls its MoE block
/// `block_sparse_moe` and its expert projections w1/w3/w2, where Qwen3 and OLMoE
/// use `mlp` and gate_proj/up_proj/down_proj. Hardcoding one spelling in the
/// loader means the second family fails with "tensor not found" and the fix
/// lands in core instead of in the adapter.
struct TensorNaming {
    std::string moe_block = "mlp";
    std::string dense_block = "mlp";
    std::string shared_block = "mlp.shared_experts";
    std::string router = "gate.weight";
    std::string expert_gate = "gate_proj.weight";
    std::string expert_up = "up_proj.weight";
    std::string expert_down = "down_proj.weight";
};

struct TokenizerSpec {
    TokenizerKind kind = TokenizerKind::Bpe;
    std::string compiled_path;
    bool byte_fallback = false;
    std::uint32_t n_special_tokens = 0;
    std::string roundtrip_sha;
};

/// Measured at admission. NOT covered by arch_hash — re-profiling on faster
/// disks must not invalidate KV checkpoints. Requantizing must, and does,
/// because QuantMap is inside the hash.
///
/// Every field here is a MEASUREMENT. None is inherited as a constant from prior
/// art: expert size, achievable bandwidth at that size, and whether streaming
/// pays at all are all per-model.
struct Economics {
    std::uint64_t expert_bytes = 0;
    std::uint32_t n_moe_layers = 0;
    std::uint64_t bytes_per_token = 0; ///< n_moe_layers × top_k × expert_bytes
    std::uint64_t total_routed_bytes = 0;
    std::uint64_t dense_resident_bytes = 0;
    float active_fraction = 0.0f; ///< top_k / n_experts

    /// Measured with reads the size of THIS model's experts. A 2.4 MB read and
    /// an 88 MB read do not achieve the same bandwidth on the same drive, and a
    /// single headline number is how a verdict ends up confidently wrong.
    std::uint64_t measured_disk_bw = 0;
    std::string measured_at_host;
};

struct ArchIr {
    std::uint32_t schema_version = kArchIrSchemaVersion;
    std::string arch_hash;

    std::string source_repo;
    std::string source_revision;
    std::string source_model_type;
    std::string adapter;

    /// RMSNorm epsilon. An architecture property, not a runtime knob — it
    /// applies to attention q/k norms, layer norms, and the output norm alike,
    /// so it lives here rather than being threaded through each of them.
    float rms_norm_eps = 1e-6f;

    Topology topology{};
    AttentionSpec attention{};
    RouterSpec router{};
    FfnSpec ffn{};
    ModalitySpec modality{};
    BlockResidualSpec block_residual{};
    HyperConnectionSpec hyper_connections{};
    SpeculativeSpec speculative{};
    QuantMap quantization{};
    TensorNaming naming{};
    TokenizerSpec tokenizer{};
    Economics economics{};

    std::uint32_t n_moe_layers() const noexcept;
    bool is_moe_layer(LayerIndex layer) const noexcept;

    /// The width a routed expert's gate/up read and its down writes.
    ///
    /// `d_model` for every family without a latent MoE, which is why callers
    /// that predate one still get the same answer. Expressed once, here, so a
    /// second site cannot forget the fallback and charge zero.
    std::uint32_t routed_expert_width() const noexcept {
        return ffn.routed_expert_hidden != 0 ? ffn.routed_expert_hidden : topology.d_model;
    }
};

/// Parse + validate the registry's canonical architecture JSON.
Status parse_arch_ir(std::string_view json, ArchIr& out);

/// Adapt an upstream HuggingFace config.json into the IR.
///
/// This is the per-family name/semantics adapter. It is the ONE place that knows
/// upstream expresses the same concept three ways (`num_experts` vs
/// `num_local_experts` vs `n_routed_experts`) and that some semantics are
/// implied by `model_type` rather than stated (which QkNormKind a family uses).
///
/// Returns Unsupported for a family with no adapter, naming it — better than
/// guessing defaults and producing a model that runs and is wrong.
Status adapt_hf_config(std::string_view json, ArchIr& out);

/// Overlay a converted container's `container_meta.json` onto an IR built from
/// config.json.
///
/// The quantization is part of the model's IDENTITY — arch_hash covers the quant
/// map precisely so that the same weights at two quantizations are two models
/// with two verdicts and two sets of KV checkpoints. config.json does not carry
/// it; container_meta.json is the record of the conversion and the only place it
/// exists.
///
/// A missing or unparseable field leaves the IR's default rather than failing:
/// this runs on the path that plans an UNCONVERTED checkpoint too, where there
/// is no conversion to describe.
Status apply_container_quant(std::string_view meta_json, ArchIr& io);

/// Structural validation. Rejects on any condition in schemas/arch-ir.md §9,
/// including a non-F32 router.
Status validate_arch_ir(const ArchIr& ir);

/// Canonicalize and hash §2–§6. Excludes `economics` and `arch_hash` itself.
Status compute_arch_hash(const ArchIr& ir, std::string& out_hash);

const char* to_string(AttentionFamily family) noexcept;
const char* to_string(Activation activation) noexcept;
const char* to_string(Modality modality) noexcept;
const char* to_string(ScoreFn score_fn) noexcept;

} // namespace soma
