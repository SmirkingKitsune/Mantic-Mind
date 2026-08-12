#pragma once

// Soma — the in-memory form of arch.json.
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

inline constexpr std::uint32_t kArchIrSchemaVersion = 1;

/// Selects the attention backend. The core switches on this exactly once, at
/// load, to resolve a descriptor — never in a loop.
///
/// `Unknown` is the zero value and the default everywhere. There is deliberately
/// no default *family*: a default-constructed spec that claimed to be GQA would
/// be both wrong and a quiet bias toward the first architecture shipped, which
/// is the exact failure the two-family co-design exists to prevent.
enum class AttentionFamily : std::uint8_t { Unknown = 0, Mha, Gqa, Mla, MlaDsa };

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
enum class ScoreFn : std::uint8_t { Softmax = 0, Sigmoid };
enum class Activation : std::uint8_t { SwiGlu = 0, GeGlu, Relu2 };
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
    /// `arch::mla::prepare_weights`.
    ///
    /// False selects the expanded form, which is kept as the reference the
    /// absorbed one is checked against; `soma_decode_kv_g4` runs both.
    bool absorb_weights = true;
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
};

struct FfnSpec {
    Activation activation = Activation::SwiGlu;
    bool has_gate = true;
    std::uint32_t expert_intermediate = 0;
    std::uint32_t dense_intermediate = 0;
    std::uint32_t shared_intermediate = 0;
    ExpertLayout expert_layout = ExpertLayout::InterleavedGateUpDown;
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
    QuantMap quantization{};
    TensorNaming naming{};
    TokenizerSpec tokenizer{};
    Economics economics{};

    std::uint32_t n_moe_layers() const noexcept;
    bool is_moe_layer(LayerIndex layer) const noexcept;
};

/// Parse + validate our own canonical arch.json.
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
const char* to_string(ScoreFn score_fn) noexcept;

} // namespace soma
