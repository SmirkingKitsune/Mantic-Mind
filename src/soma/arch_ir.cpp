#include "soma/arch_ir.hpp"

#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <sstream>

using json = nlohmann::json;

namespace soma {

namespace {

// Upstream names the same concept several ways. Resolving that here is the
// adapter's entire job — nothing downstream should ever see these spellings.
constexpr std::array kExpertCountKeys = {"num_experts", "num_local_experts", "n_routed_experts"};
constexpr std::array kMoeIntermediateKeys = {"moe_intermediate_size"};

// The DENSE layers' FFN width, when a family states it separately from
// `intermediate_size`.
//
// Every family before MiniMax-M3 had one FFN width per layer kind and spelled it
// `intermediate_size`, so "dense width" and "expert width" fell out of the same
// key. M3 splits them: `intermediate_size` 3072 is what a ROUTED expert runs at
// and `dense_intermediate_size` 12288 is what its three leading dense layers do.
// Reading only the shared key sizes those layers at a quarter of their real
// width -- the tensors then fail to bind, which is loud, but the PLANNER binds
// nothing and would have under-reported the resident half by ~1.1 GB per layer
// on the production checkpoint.
constexpr std::array kDenseIntermediateKeys = {"dense_intermediate_size"};

// Qwen2-MoE and Qwen3.5 say `shared_expert_intermediate_size`; MiniMax-M3 says
// `shared_intermediate_size`. Same quantity, and the DeepSeek families state
// neither and derive it from the count instead (below).
constexpr std::array kSharedIntermediateKeys = {"shared_expert_intermediate_size",
                                                "shared_intermediate_size"};

// Kimi spells the DeepSeek router's every field differently while meaning
// exactly the same thing — it says so itself, in a comment mapping
// `num_experts_per_token -> num_experts_per_tok` and three more. Reading only
// the DeepSeek spelling leaves top_k at 0, which validation catches, and
// `n_shared_experts` at 0, which it does NOT: the model then plans with no
// shared expert, under-counts the resident half, and serves fine right up until
// the tensors do not bind.
constexpr std::array kTopKKeys = {"num_experts_per_tok", "num_experts_per_token"};
constexpr std::array kSharedExpertKeys = {"n_shared_experts", "num_shared_experts"};
constexpr std::array kExpertGroupKeys = {"n_group", "num_expert_group"};
constexpr std::array kScoringKeys = {"scoring_func", "moe_router_activation_func"};
constexpr std::array kNormTopkKeys = {"norm_topk_prob", "moe_renormalize"};

template <typename T>
T get_or(const json& j, std::string_view key, T fallback) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    try {
        return it->get<T>();
    } catch (...) {
        return fallback;
    }
}

const json* find_first(const json& j, const auto& keys) {
    for (const char* k : keys) {
        const auto it = j.find(k);
        if (it != j.end() && !it->is_null()) return &(*it);
    }
    return nullptr;
}

/// Materialize dense-vs-MoE per layer.
///
/// Upstream expresses this three different ways and they compose: DeepSeek uses
/// first_k_dense_replace + moe_layer_freq, Qwen3 uses decoder_sparse_step +
/// mlp_only_layers, Mixtral uses nothing at all (every layer is MoE). Resolving
/// it once here means the core never re-derives it — and Topology::layer_kinds
/// is authoritative from that point on.
std::vector<LayerKind> resolve_layer_kinds(const json& j, std::uint32_t n_layers) {
    std::vector<LayerKind> kinds(n_layers, LayerKind::Moe);

    // A STATED per-layer list wins over any derivation, for the same reason
    // Topology::layer_kinds exists at all: a stride plus an offset is a
    // re-derivation waiting to disagree with the weights.
    //
    // GLM-5.2 ships `mlp_layer_types` and the derivation below happens to agree
    // with it — `first_k_dense_replace: 3` with `moe_layer_freq: 1` reproduces
    // its three-dense prefix exactly — so reading it changes no existing
    // container's arch_hash. That agreement is precisely why the explicit list
    // should win anyway: a family with an IRREGULAR pattern would be silently
    // mis-derived and nothing would notice.
    if (const auto it = j.find("mlp_layer_types"); it != j.end() && it->is_array()) {
        if (it->size() >= n_layers) {
            for (std::uint32_t i = 0; i < n_layers; ++i) {
                const auto& v = (*it)[i];
                const auto name = v.is_string() ? v.get<std::string>() : std::string{};
                kinds[i] = (name == "sparse" || name == "moe" || name == "hash_moe")
                               ? LayerKind::Moe
                               : LayerKind::Dense;
            }
            return kinds;
        }
    }

    const auto first_dense = get_or<std::uint32_t>(j, "first_k_dense_replace", 0);
    for (std::uint32_t i = 0; i < std::min(first_dense, n_layers); ++i) {
        kinds[i] = LayerKind::Dense;
    }

    const auto step =
        std::max<std::uint32_t>(1, get_or<std::uint32_t>(j, "decoder_sparse_step", 1));
    std::vector<std::uint32_t> mlp_only;
    if (const auto it = j.find("mlp_only_layers"); it != j.end() && it->is_array()) {
        for (const auto& v : *it)
            mlp_only.push_back(v.get<std::uint32_t>());
    }

    // `moe_layer_freq` is TWO different things under one key, and reading only
    // the scalar form is how a 57-MoE-layer model becomes a 60-MoE-layer one.
    //
    //   scalar  a STRIDE: layer i is MoE when `i % freq == 0`.
    //   list    a per-layer MASK, one 0/1 per layer, `i` naming layer `i`.
    //
    // DeepSeek's config class accepts both and MiniMax-M3 ships the list — 3
    // zeros then 57 ones. Read as a scalar the list simply fails the
    // `is_number_integer` test, `freq` stays 1, and the dense prefix vanishes:
    // every layer is called MoE, the three dense layers' `gate_up_proj` is never
    // bound, and the model asks for experts that are not there. That at least
    // fails loudly at bind time — but the PLANNER runs first and does not bind
    // anything, so `bytes_per_token` is over-reported by three layers' worth of
    // routed experts on the exact quantity the verdict is computed from.
    std::uint32_t freq = 1;
    std::vector<std::uint8_t> freq_mask;
    if (const auto it = j.find("moe_layer_freq"); it != j.end()) {
        if (it->is_number_integer()) {
            freq = std::max<std::uint32_t>(1, it->get<std::uint32_t>());
        } else if (it->is_array()) {
            freq_mask.reserve(it->size());
            for (const auto& v : *it) {
                freq_mask.push_back(v.is_number() && v.get<double>() != 0.0 ? 1u : 0u);
            }
        }
    }

    for (std::uint32_t i = 0; i < n_layers; ++i) {
        if (kinds[i] == LayerKind::Dense) continue;
        const bool only_mlp = std::find(mlp_only.begin(), mlp_only.end(), i) != mlp_only.end();
        // A mask SHORTER than the stack leaves the unnamed tail alone rather
        // than defaulting it dense: the tail is already Moe, and inventing an
        // answer for a layer the config did not describe is worse than
        // inheriting the default the rest of this function assumes.
        const bool masked_off = i < freq_mask.size() && freq_mask[i] == 0u;
        if (only_mlp || masked_off || (step > 1 && i % step != 0) ||
            (freq > 1 && i % freq != 0)) {
            kinds[i] = LayerKind::Dense;
        }
    }
    return kinds;
}

Status parse_rope(const json& j, RopeConfig& out) {
    out.theta = get_or<float>(j, "rope_theta", 10000.0f);

    // transformers migrated `rope_scaling` to `rope_parameters`; accept both.
    const json* scaling = nullptr;
    if (const auto it = j.find("rope_scaling"); it != j.end() && it->is_object()) scaling = &(*it);
    if (!scaling) {
        if (const auto it = j.find("rope_parameters"); it != j.end() && it->is_object()) {
            scaling = &(*it);
        }
    }
    if (!scaling) return {};

    // `rope_theta` MOVED INTO that block in the same migration, and reading only
    // the top level is how this silently used 10000 for a model that says
    // 8000000. GLM-5.2 states it nowhere else, so nothing was missing and nothing
    // was malformed — the default simply applied.
    //
    // The failure is quiet in the worst way: every projection, norm and shape
    // stays correct and only the rotation angles are wrong, so the model loads,
    // runs, produces finite logits, and matches its reference EXACTLY at position
    // 0 — where the rotation is the identity for any theta. It first appeared as
    // a 1.8 max|diff| on the `k_pe_rot` tap with everything feeding it clean.
    //
    // Nested wins when present: transformers' `standardize_rope_params` writes the
    // authoritative value there, and a config carrying both is a config mid-
    // migration rather than a config with two opinions.
    if (const auto it = scaling->find("rope_theta"); it != scaling->end() && it->is_number()) {
        out.theta = it->get<float>();
    }

    const auto kind = get_or<std::string>(
        *scaling, "rope_type", get_or<std::string>(*scaling, "type", "default"));
    if (kind == "yarn") {
        out.scaling.kind = RopeScalingKind::Yarn;
    } else if (kind == "linear") {
        out.scaling.kind = RopeScalingKind::Linear;
    } else if (kind == "ntk" || kind == "dynamic") {
        out.scaling.kind = RopeScalingKind::Ntk;
    } else if (kind == "default" || kind.empty()) {
        out.scaling.kind = RopeScalingKind::None;
        return {};
    } else {
        return {StatusCode::Unsupported, "unsupported rope scaling type '" + kind + "'"};
    }

    out.scaling.factor = get_or<float>(*scaling, "factor", 1.0f);
    out.scaling.original_max_position =
        get_or<std::uint32_t>(*scaling, "original_max_position_embeddings", 0);
    out.scaling.beta_fast = get_or<float>(*scaling, "beta_fast", 32.0f);
    out.scaling.beta_slow = get_or<float>(*scaling, "beta_slow", 1.0f);
    out.scaling.mscale = get_or<float>(*scaling, "mscale", 1.0f);
    out.scaling.mscale_all_dim = get_or<float>(*scaling, "mscale_all_dim", 1.0f);
    return {};
}

/// Per-family semantics that config.json does NOT state.
struct FamilyTraits {
    AttentionFamily attention = AttentionFamily::Gqa;
    QkNormKind qk_norm = QkNormKind::None;
    bool supported = false;

    /// Some families always renormalize top-k weights but carry no
    /// `norm_topk_prob` key to say so. Mixtral divides by the top-k sum
    /// unconditionally; reading the absent key as `false` would skip it and
    /// scale every expert contribution wrong.
    bool force_normalize_topk = false;

    /// Default for `rms_norm_eps` when config.json OMITS it.
    ///
    /// There is no universal fallback. OLMoE ships no rms_norm_eps at all and
    /// OlmoeConfig's class default is 1e-5; Qwen3 states 1e-6 explicitly. Using
    /// one hardcoded constant silently mis-normalizes every layer of any family
    /// whose default differs — small at position 0 and compounding with context
    /// length, which reads as a subtle attention bug rather than a config one.
    float default_rms_eps = 1e-6f;

    TensorNaming naming{};

    /// The shared expert is scaled by a learned scalar gate, and config.json
    /// never says so. `Qwen2MoeSparseMoeBlock` and `Qwen3_5MoeSparseMoeBlock`
    /// both build a `shared_expert_gate` Linear unconditionally alongside the
    /// shared expert; DeepSeek's adds its shared expert ungated. Same category
    /// as `force_normalize_topk` — family knowledge with no key to read.
    ///
    /// Trailing, so every existing aggregate initializer above stays valid.
    bool gated_shared_expert = false;

    /// `x_hat * w` or `x_hat * (1 + w)`. Family knowledge with no key to read,
    /// and the most quietly destructive of the three flags here: a one-plus
    /// checkpoint's norm weights are centred on ZERO, so reading them plainly
    /// multiplies every normalized activation by roughly nothing. Also trailing.
    RmsNormScale rms_norm_scale = RmsNormScale::Weight;
};

TensorNaming mixtral_naming() {
    TensorNaming n;
    n.moe_block = "block_sparse_moe";
    n.router = "gate.weight";
    n.expert_gate = "w1.weight"; // MixtralBlockSparseTop2MLP: w1 = gate
    n.expert_up = "w3.weight";   //                            w3 = up
    n.expert_down = "w2.weight"; //                            w2 = down
    return n;
}

TensorNaming compressed_sparse_naming() {
    TensorNaming n;
    n.moe_block = "ffn";
    n.dense_block = "ffn";
    n.shared_block = "ffn.shared_experts";
    n.router = "gate.weight";
    n.expert_gate = "w1.weight";
    n.expert_up = "w3.weight";
    n.expert_down = "w2.weight";
    return n;
}

TensorNaming kimi_naming() {
    TensorNaming n;
    n.moe_block = "block_sparse_moe";
    n.dense_block = "mlp";                            // KimiMLP: gate/up/down
    n.shared_block = "block_sparse_moe.shared_experts"; // KimiMLP too
    n.router = "gate.weight";
    n.expert_gate = "w1.weight"; // KimiBlockSparseMLP: w1 = gate
    n.expert_up = "w3.weight";   //                     w3 = up
    n.expert_down = "w2.weight"; //                     w2 = down
    return n;
}

TensorNaming qwen3_5_naming() {
    TensorNaming n;
    // SINGULAR. Qwen2-MoE and Qwen3.5 spell it `mlp.shared_expert`; DeepSeek and
    // the TensorNaming default spell it `mlp.shared_experts`. One character, and
    // the tensor simply does not bind — which at least fails loudly, unlike most
    // of the hazards in this file.
    n.shared_block = "mlp.shared_expert";
    return n;
}

FamilyTraits traits_for(const std::string& model_type) {
    // qk_norm kind is family knowledge, not config knowledge. Both OLMoE and
    // Qwen3-MoE simply have q_norm/k_norm tensors; only the shape distinguishes
    // them, and by then it is too late to be reading tensors to decide
    // semantics.
    if (model_type == "olmoe")
        return {AttentionFamily::Gqa, QkNormKind::FullWidth, true, false, 1e-5f, {}};
    if (model_type == "qwen3_moe")
        return {AttentionFamily::Gqa, QkNormKind::PerHead, true, false, 1e-6f, {}};
    if (model_type == "qwen2_moe")
        return {AttentionFamily::Gqa, QkNormKind::None, true, false, 1e-6f, {}};
    if (model_type == "mixtral")
        return {AttentionFamily::Gqa, QkNormKind::None, true, true, 1e-5f, mixtral_naming()};
    if (model_type == "deepseek_v2") {
        return {AttentionFamily::Mla, QkNormKind::None, true, false, 1e-6f, {}};
    }
    if (model_type == "deepseek_v3") {
        // Shares MLA attention with V2 and NOT its router: V3 adds `noaux_tc`
        // sigmoid scoring with a per-expert selection bias and group-limited
        // top-k. Grouping the two — as this table originally did — ran V3 through
        // V2's softmax router and produced identical logits at every position,
        // i.e. no signal at all. The MLA router now implements both, with V2 as
        // the degenerate case.
        return {AttentionFamily::Mla, QkNormKind::None, true, false, 1e-6f, {}};
    }
    if (model_type == "glm_moe_dsa") {
        // DESCRIBABLE, not servable — and those are different questions.
        //
        // GLM-5.2's expert half is ordinary: 256 routed experts, top-8, one
        // shared, sigmoid + noaux_tc routing, dense-then-sparse layers. Every one
        // of those keys is already read below. Its attention is MLA with a sparse
        // key indexer on top (DSA), and `resolve_f32_backend` returns nullptr for
        // MlaDsa on purpose — running it through the MLA backend would serve it
        // as DENSE attention, which is a different model that happens to produce
        // finite numbers.
        //
        // Adapting it anyway is the point. The verdict is a property of expert
        // ECONOMICS — bytes/token, active fraction, routed set against a host
        // budget — and none of that depends on how attention selects keys. So a
        // plan is honest and useful before a backend exists, and `arch_supported`
        // is what carries "cannot be run" to admission so it never spends hours
        // converting 1.4 TB it could not serve.
        return {AttentionFamily::MlaDsa, QkNormKind::None, true, false, 1e-5f, {}};
    }
    if (model_type == "kimi_k3" || model_type == "kimi_linear") {
        // Hybrid, and the hybrid is the architecture — not a variant of either
        // half. 24 of 93 layers are MLA (NoPE, output-gated); the other 69 are
        // gated delta-rule linear attention with a constant-size recurrent
        // state. Calling it Mla would size 93 layers of KV cache for a model
        // that has 24, which at its 1M context is a ~120 GB over-count in the
        // pessimistic direction; calling it a linear family would drop the KV
        // cache the other 24 genuinely need.
        //
        // `kimi_linear` is the same language model without the vision tower, and
        // shares every trait: `text_config.model_type` IS `kimi_linear`, so the
        // two names arriving here are the wrapper and the thing it wraps.
        //
        // Experts are Mixtral-spelled (w1/w3/w2 under `block_sparse_moe`) while
        // the DENSE and SHARED blocks are gate/up/down — one family using both
        // conventions in one layer, which is exactly why TensorNaming carries
        // the blocks separately instead of a single family switch.
        return {AttentionFamily::MlaKda, QkNormKind::None, true, false, 1e-6f, kimi_naming()};
    }
    if (model_type == "qwen3_5_moe_text" || model_type == "qwen3_5_moe") {
        // The second hybrid, and deliberately NOT routed through `MlaKda`.
        //
        // 69 of 92 layers are Gated DeltaNet — a fixed-size recurrent state; the
        // other 23 are ordinary GQA, 64 heads over 4 kv-heads, output-gated and
        // rotated over only the first quarter of each head. Nothing about the
        // full half is MLA, so the two hybrids share a SHAPE and no arithmetic.
        //
        // `force_normalize_topk` because `Qwen3_5MoeTopKRouter` divides the
        // top-k probabilities by their sum unconditionally and the config
        // carries no `norm_topk_prob` to say so — the same trap Mixtral sets,
        // and reading the absent key as false mis-scales every expert.
        //
        // `qwen3_5_moe` is accepted alongside the text-only spelling because the
        // multimodal wrapper's `model_type` is that, with the language model
        // under `text_config`; the hoist above lands here either way.
        return {AttentionFamily::GqaGdn,
                QkNormKind::PerHead,
                true,
                true,
                1e-6f,
                qwen3_5_naming(),
                /*gated_shared_expert=*/true,
                // `Qwen3_5MoeRMSNorm.forward` is `x_hat * (1.0 + weight)` with
                // `weight` a zeros-initialized Parameter — Gemma's convention,
                // and the first family here to use it. Its GATED norm is a
                // different class using the plain form, which is why this is a
                // model-level property that `arch::gdn::gated_rmsnorm`
                // deliberately does not consult.
                RmsNormScale::OnePlusWeight};
    }
    if (model_type == "minimax_m3_vl" || model_type == "minimax_m3_vl_text") {
        // GQA whose key SELECTION is block-sparse. NOT a spelling of MlaDsa, and
        // not plain Gqa either.
        //
        // Routing it to `Gqa` is the tempting shortcut — the projections, the
        // head ratio, the qk-norm and the partial rotation are all ordinary, and
        // it would run. It would run DENSE: 57 of 60 layers attending over the
        // whole prefix instead of 16 blocks of it. Correct-looking at short
        // context, because top-k over fewer than 2048 tokens selects everything,
        // and progressively a different model beyond that. Exactly the failure
        // `MlaDsa` was held back for.
        //
        // Three traits below have no key in config.json to read, and each was
        // taken from `modeling_minimax_m3_vl.py` rather than guessed:
        //
        //   force_normalize_topk  `MiniMaxM3VLTopKRouter.forward` ends
        //                         `top_k_weights /= top_k_weights.sum(...)`
        //                         unconditionally, and the config carries no
        //                         `norm_topk_prob`. Reading the absent key as
        //                         false leaves four sigmoid probabilities
        //                         summing to anything, then multiplies the lot
        //                         by `routed_scaling_factor` 2.0 — the same trap
        //                         Mixtral sets, with a factor of two on top.
        //   rms_norm_scale        `MiniMaxM3VLRMSNorm` is Gemma's `x_hat *
        //                         (1 + w)` with a zeros-initialized weight. The
        //                         config DOES say so (`use_gemma_norm: true`),
        //                         which makes this the first family here whose
        //                         convention is stated — but it is family
        //                         knowledge for the text-only spelling too, and
        //                         one source is better than two.
        //   qk_norm               per head: `q_norm` is `RMSNorm(head_dim)`
        //                         applied to a `[..., n_heads, head_dim]` view.
        //
        // Naming is the DEFAULT, deliberately: `mlp`, `mlp.shared_experts`,
        // `gate.weight`. What is not default is that the dense, shared and
        // routed FFNs all carry a FUSED `gate_up_proj` rather than separate
        // gate/up projections — a layout fact the loader probes for rather than
        // a name TensorNaming could hold.
        return {AttentionFamily::GqaBsa,
                QkNormKind::PerHead,
                true,
                /*force_normalize_topk=*/true,
                1e-6f,
                {},
                /*gated_shared_expert=*/false,
                RmsNormScale::OnePlusWeight};
    }
    if (model_type == "deepseek_v4") {
        return {AttentionFamily::CompressedSparse,
                QkNormKind::FullWidth,
                true,
                false,
                1e-6f,
                compressed_sparse_naming()};
    }
    return {};
}

} // namespace

std::uint32_t ArchIr::n_moe_layers() const noexcept {
    return static_cast<std::uint32_t>(
        std::count(topology.layer_kinds.begin(), topology.layer_kinds.end(), LayerKind::Moe));
}

bool ArchIr::is_moe_layer(LayerIndex layer) const noexcept {
    return layer < topology.layer_kinds.size() && topology.layer_kinds[layer] == LayerKind::Moe;
}

const char* to_string(AttentionFamily family) noexcept {
    switch (family) {
    case AttentionFamily::Unknown:
        return "unknown";
    case AttentionFamily::Mha:
        return "mha";
    case AttentionFamily::Gqa:
        return "gqa";
    case AttentionFamily::Mla:
        return "mla";
    case AttentionFamily::MlaDsa:
        return "mla+dsa";
    case AttentionFamily::CompressedSparse:
        return "compressed+sparse";
    case AttentionFamily::MlaKda:
        return "mla+kda";
    case AttentionFamily::GqaGdn:
        return "gqa+gdn";
    case AttentionFamily::GqaBsa:
        return "gqa+bsa";
    }
    return "unknown";
}

const char* to_string(Activation activation) noexcept {
    switch (activation) {
    case Activation::SwiGlu:
        return "swiglu";
    case Activation::GeGlu:
        return "geglu";
    case Activation::Relu2:
        return "relu2";
    case Activation::Situ:
        return "situ";
    case Activation::SwiGluOai:
        return "swiglu-oai";
    }
    return "unknown";
}

const char* to_string(Modality modality) noexcept {
    switch (modality) {
    case Modality::Text:
        return "text";
    case Modality::VisionText:
        return "vision+text";
    }
    return "unknown";
}

const char* to_string(ScoreFn score_fn) noexcept {
    switch (score_fn) {
    case ScoreFn::Softmax:
        return "softmax";
    case ScoreFn::Sigmoid:
        return "sigmoid";
    case ScoreFn::SqrtSoftplus:
        return "sqrtsoftplus";
    }
    return "unknown";
}

Status adapt_hf_config(std::string_view text, ArchIr& out) {
    json j; // reassigned below when a multimodal wrapper nests the language model
    try {
        j = json::parse(text);
    } catch (const std::exception& e) {
        return {StatusCode::InvalidArgument, std::string("config.json parse failed: ") + e.what()};
    }
    if (!j.is_object()) {
        return {StatusCode::InvalidArgument, "config.json is not an object"};
    }

    const auto model_type = get_or<std::string>(j, "model_type", "");
    const auto traits = traits_for(model_type);
    if (!traits.supported) {
        return {StatusCode::Unsupported,
                "no adapter for model_type '" + model_type +
                    "'; supported: olmoe, qwen3_moe, qwen2_moe, qwen3_5_moe, "
                    "qwen3_5_moe_text, mixtral, deepseek_v2, deepseek_v3, deepseek_v4, "
                    "glm_moe_dsa, kimi_k3, kimi_linear, minimax_m3_vl, "
                    "minimax_m3_vl_text"};
    }

    out = ArchIr{};
    if (model_type == "deepseek_v4") out.schema_version = kArchIrSchemaVersionV2;
    out.source_model_type = model_type;
    out.adapter = model_type;
    out.source_repo = get_or<std::string>(j, "_name_or_path", "");
    out.source_revision = get_or<std::string>(j, "_commit_hash", "");

    // ── the multimodal wrapper, hoisted ──────────────────────────────────────
    //
    // Every family before this one ships a FLAT config.json: the language model
    // IS the top level. A multimodal checkpoint instead makes the top level a
    // container — `text_config` holds the language model, `vision_config` the
    // tower — and the top level then carries almost none of the keys read below.
    //
    // Read flat, this does not fail. It SUCCEEDS with n_layers 0, d_model 0 and
    // no experts, and validation rejects it with "topology has a zero
    // dimension", which is true and tells the operator nothing about why.
    //
    // The wrapper's own facts are captured BEFORE the hoist, because after it
    // they are unreachable. Which modality the checkpoint accepts is one of
    // them, and it is not cosmetic: a vision checkpoint served as text is a
    // model being asked about images it was never given.
    if (const auto it = j.find("vision_config"); it != j.end() && it->is_object()) {
        auto& mm = out.modality;
        mm.modality = Modality::VisionText;
        // Two spellings, and the `vt_`-prefixed one is tried FIRST so that no
        // checkpoint already described here changes its arch_hash.
        //
        // Kimi-K3 prefixes the tower's dimensions inside `vision_config`;
        // MiniMax-M3 states them plainly, because its `vision_config` is a whole
        // nested config object of its own rather than a flat block. Reading only
        // the prefixed form left the second family reporting a 0-layer,
        // 0-wide tower — which validates, plans, and tells the operator the
        // model is text-only when it is not. That is the one thing
        // `ModalitySpec` exists to prevent.
        mm.media_placeholder_token_id = get_or<std::uint32_t>(
            j, "media_placeholder_token_id", get_or<std::uint32_t>(j, "image_token_index", 0));
        mm.vision_layers = get_or<std::uint32_t>(
            *it, "vt_num_hidden_layers", get_or<std::uint32_t>(*it, "num_hidden_layers", 0));
        mm.vision_hidden = get_or<std::uint32_t>(
            *it, "vt_hidden_size", get_or<std::uint32_t>(*it, "hidden_size", 0));
        mm.vision_patch_size = get_or<std::uint32_t>(*it, "patch_size", 0);
    }
    if (const auto it = j.find("text_config"); it != j.end() && it->is_object()) {
        // The nested block wins wholesale rather than being merged key-by-key.
        // Both levels declare bos/eos/pad and tie_word_embeddings, and the
        // language model's copy is the one its tensors were trained with; a
        // merge would have to pick a winner per key and would pick wrong the
        // first time the two disagree.
        json nested = *it;
        j = std::move(nested);
    }

    // ── topology ─────────────────────────────────────────────────────────────
    // Family default when the key is absent — NOT a global constant.
    out.rms_norm_eps = get_or<float>(
        j, "rms_norm_eps", get_or<float>(j, "layer_norm_eps", traits.default_rms_eps));
    out.rms_norm_scale = traits.rms_norm_scale;
    out.naming = traits.naming;

    out.topology.n_layers = get_or<std::uint32_t>(j, "num_hidden_layers", 0);
    out.topology.d_model = get_or<std::uint32_t>(j, "hidden_size", 0);
    out.topology.vocab_size = get_or<std::uint32_t>(j, "vocab_size", 0);
    out.topology.first_k_dense = get_or<std::uint32_t>(j, "first_k_dense_replace", 0);
    out.topology.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", false);
    out.topology.max_position_embeddings =
        get_or<std::uint32_t>(j, "max_position_embeddings", 0);
    if (const auto it = j.find("eos_token_id"); it != j.end() && !it->is_null()) {
        if (it->is_array()) {
            for (const auto& token : *it) {
                if (token.is_number_unsigned() || token.is_number_integer()) {
                    out.topology.eos_token_ids.push_back(token.get<std::uint32_t>());
                }
            }
        } else if (it->is_number_unsigned() || it->is_number_integer()) {
            out.topology.eos_token_ids.push_back(it->get<std::uint32_t>());
        }
    }
    out.topology.layer_kinds = resolve_layer_kinds(j, out.topology.n_layers);

    // ── attention ────────────────────────────────────────────────────────────
    auto& attn = out.attention;
    attn.n_heads = get_or<std::uint32_t>(j, "num_attention_heads", 0);
    attn.n_kv_heads = get_or<std::uint32_t>(j, "num_key_value_heads", attn.n_heads);
    attn.head_dim = get_or<std::uint32_t>(j, "head_dim", 0);
    if (attn.head_dim == 0 && attn.n_heads > 0) {
        attn.head_dim = out.topology.d_model / attn.n_heads;
    }
    attn.qk_norm = traits.qk_norm;
    attn.bias = get_or<bool>(j, "attention_bias", false);

    // MHA is the n_kv_heads == n_heads case OF THE GQA BACKEND. Recorded
    // distinctly because kv_bytes_per_token differs by a factor of
    // n_heads/n_kv_heads, and the planner needs the true number.
    //
    // The collapse is guarded on the family, which the first architecture could
    // not reveal: DeepSeek-V2-Lite has num_key_value_heads == num_attention_heads
    // == 4, so the unguarded rule classified an MLA model as MHA and handed it to
    // the GQA backend. Every tensor name would then have been missing, which is a
    // loud failure — but a config where the shapes happened to line up would have
    // run and produced wrong numbers.
    if (traits.attention == AttentionFamily::Gqa) {
        attn.family =
            (attn.n_kv_heads == attn.n_heads) ? AttentionFamily::Mha : AttentionFamily::Gqa;
    } else {
        attn.family = traits.attention;
    }

    // ── MLA ──────────────────────────────────────────────────────────────────
    //
    // Compressed KV: the cache holds a `kv_lora_rank`-wide latent plus one shared
    // RoPE segment, not per-head K and V. `head_dim` above is the GQA reading of
    // the config and is meaningless here, so it is recomputed from the two halves
    // MLA actually splits a query into.
    if (attn.family == AttentionFamily::Mla || attn.family == AttentionFamily::MlaDsa ||
        attn.family == AttentionFamily::MlaKda) {
        auto& m = attn.mla;
        m.kv_lora_rank = get_or<std::uint32_t>(j, "kv_lora_rank", 0);
        // null in config.json for V2-Lite: Q is NOT down-projected there, while
        // the full V2 compresses it. Absent and zero mean the same thing and the
        // backend branches on it.
        m.q_lora_rank = get_or<std::uint32_t>(j, "q_lora_rank", 0);
        m.qk_nope_head_dim = get_or<std::uint32_t>(j, "qk_nope_head_dim", 0);
        m.qk_rope_head_dim = get_or<std::uint32_t>(j, "qk_rope_head_dim", 0);
        m.v_head_dim = get_or<std::uint32_t>(j, "v_head_dim", 0);

        // A query head is nope ++ rope; the value head is a different width
        // entirely. Carrying one `head_dim` for both is a GQA assumption.
        attn.head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim;
        // Only the rope half is rotated.
        attn.rope.partial_dim = m.qk_rope_head_dim;

        m.nope = get_or<bool>(j, "mla_use_nope", false);
        m.output_gate = get_or<bool>(j, "mla_use_output_gate", false);
    }

    // ── DSA ──────────────────────────────────────────────────────────────────
    //
    // Read from `indexer_types` rather than derived from `index_topk_freq`, and
    // that is the whole point. The stride would give the right answer for
    // GLM-5.2 — one `full` every four after the first three — and the weights are
    // what actually decide: 57 of its 78 layers ship no indexer tensors, and a
    // layer the IR called `Full` without the weights to back it would fail at
    // bind time or, worse, silently borrow nothing.
    //
    // Same argument as `Topology::layer_kinds`, which resolves three different
    // upstream spellings of "which layers are MoE" once, here, so the core never
    // re-derives it.
    if (attn.family == AttentionFamily::MlaDsa) {
        auto& dsa = attn.dsa;
        dsa.index_topk = get_or<std::uint32_t>(j, "index_topk", 0);
        dsa.n_index_heads = get_or<std::uint32_t>(j, "index_n_heads", 0);
        dsa.index_head_dim = get_or<std::uint32_t>(j, "index_head_dim", 0);
        dsa.index_freq = get_or<std::uint32_t>(j, "index_topk_freq", 0);

        if (const auto it = j.find("indexer_types"); it != j.end() && it->is_array()) {
            dsa.layer_kinds.reserve(it->size());
            for (const auto& v : *it) {
                const auto name = v.is_string() ? v.get<std::string>() : std::string{};
                dsa.layer_kinds.push_back(name == "full"     ? IndexerKind::Full
                                          : name == "shared" ? IndexerKind::Shared
                                                             : IndexerKind::None);
            }
        }
        // Length must match, or every downstream layer lookup is off. A config
        // that disagrees with itself is not something to paper over.
        if (!dsa.layer_kinds.empty() && dsa.layer_kinds.size() != out.topology.layer_kinds.size()) {
            return {StatusCode::InvalidArgument,
                    "indexer_types has " + std::to_string(dsa.layer_kinds.size()) +
                        " entries for " + std::to_string(out.topology.layer_kinds.size()) +
                        " layers"};
        }
        // An all-`Shared` stack has nothing to share FROM. Caught here rather
        // than at bind time, where it would present as a missing tensor.
        if (!dsa.layer_kinds.empty() && dsa.n_full_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "indexer_types contains no 'full' layer, so no layer can compute an index"};
        }
    }

    // ── KDA ──────────────────────────────────────────────────────────────────
    //
    // The layer sets are stated, not strided, and they are stated ONE-BASED.
    //
    // That is the whole hazard of this family. `full_attn_layers` reads
    // [4, 8, ... 88, 92, 93] against 93 layers, which is a perfectly plausible
    // zero-based list right up to the 93 — and even that reads as a harmless
    // off-the-end entry rather than the proof it is. Upstream resolves it in one
    // line, `(layer_idx + 1) in kda_layers`, and every layer of the resulting
    // model depends on it.
    //
    // Read zero-based, the assignment shifts by one: layer 3 becomes linear and
    // layer 4 becomes full, the KV cache is sized for the wrong 24 layers, and
    // the tensors bind to the wrong blocks. The stride LOOKS regular enough
    // (every 4th) that a re-derivation would agree with the wrong answer
    // everywhere except the final pair 92/93 — which is exactly where the list
    // stops being a stride.
    //
    // So it is converted here, once, and `layer_kinds` is authoritative after.
    if (attn.family == AttentionFamily::MlaKda) {
        auto& kda = attn.kda;
        const auto it = j.find("linear_attn_config");
        if (it == j.end() || !it->is_object()) {
            return {StatusCode::InvalidArgument,
                    "hybrid linear attention declared with no linear_attn_config"};
        }
        const json& lc = *it;
        kda.n_heads = get_or<std::uint32_t>(lc, "num_heads", 0);
        kda.head_dim = get_or<std::uint32_t>(lc, "head_dim", 0);
        kda.conv_kernel = get_or<std::uint32_t>(lc, "short_conv_kernel_size", 0);
        kda.full_rank_gate = get_or<bool>(lc, "use_full_rank_gate", false);
        if (const auto g = lc.find("gate_lower_bound"); g != lc.end() && g->is_number()) {
            kda.has_gate_bound = true;
            kda.gate_lower_bound = g->get<float>();
        }
        if (kda.n_heads == 0 || kda.head_dim == 0 || kda.conv_kernel == 0) {
            return {StatusCode::InvalidArgument, "linear_attn_config has a zero dimension"};
        }

        const auto n_layers = out.topology.n_layers;
        // Default Full, then mark the stated linear layers. A layer named by
        // NEITHER list would silently stay Full, so both lists are read and
        // their total is checked against n_layers below.
        kda.layer_kinds.assign(n_layers, AttnLayerKind::Full);
        std::uint32_t named = 0;
        const auto mark = [&](const char* key, AttnLayerKind kind) -> Status {
            const auto list = lc.find(key);
            if (list == lc.end() || !list->is_array()) {
                return {StatusCode::InvalidArgument,
                        std::string("linear_attn_config is missing '") + key + "'"};
            }
            for (const auto& v : *list) {
                if (!v.is_number_integer() && !v.is_number_unsigned()) continue;
                const auto one_based = v.get<std::int64_t>();
                if (one_based < 1 || one_based > static_cast<std::int64_t>(n_layers)) {
                    return {StatusCode::InvalidArgument,
                            std::string(key) + " names layer " + std::to_string(one_based) +
                                ", outside 1.." + std::to_string(n_layers)};
                }
                kda.layer_kinds[static_cast<std::size_t>(one_based - 1)] = kind;
                ++named;
            }
            return {};
        };
        if (auto st = mark("kda_layers", AttnLayerKind::Linear); !st.ok()) return st;
        if (auto st = mark("full_attn_layers", AttnLayerKind::Full); !st.ok()) return st;

        // Every layer named exactly once. A config that names one twice or skips
        // one is a config that disagrees with itself, and the resulting stack
        // would bind the wrong tensors in a way nothing downstream can detect.
        if (named != n_layers) {
            return {StatusCode::InvalidArgument,
                    "linear_attn_config names " + std::to_string(named) + " layers for " +
                        std::to_string(n_layers) + " layers"};
        }
        if (kda.n_full_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid stack has no full-attention layer, so nothing carries a KV cache"};
        }
    }

    // ── GDN ──────────────────────────────────────────────────────────────────
    //
    // The layer split is a per-layer LIST and it is ZERO-based, which is the
    // exact opposite of KDA's hazard above: `layer_types[i]` names layer `i`, so
    // reading it plainly is right and there is nothing to convert.
    //
    // The trap is the FALLBACK. With `layer_types` absent, upstream synthesizes
    // `"linear_attention" if (i + 1) % interval else "full_attention"` — a
    // one-based stride producing full layers at zero-based 3, 7, 11 … The
    // obvious `i % interval == 0` reading places them at 0, 4, 8 … instead,
    // which shifts every layer's kind by three and binds `linear_attn` tensors
    // into `self_attn` blocks. Transcribed rather than reinvented, and the
    // stated list wins whenever it exists.
    if (attn.family == AttentionFamily::GqaGdn) {
        auto& gdn = attn.gdn;
        gdn.n_k_heads = get_or<std::uint32_t>(j, "linear_num_key_heads", 0);
        gdn.n_v_heads = get_or<std::uint32_t>(j, "linear_num_value_heads", 0);
        gdn.head_k_dim = get_or<std::uint32_t>(j, "linear_key_head_dim", 0);
        gdn.head_v_dim = get_or<std::uint32_t>(j, "linear_value_head_dim", 0);
        gdn.conv_kernel = get_or<std::uint32_t>(j, "linear_conv_kernel_dim", 0);

        if (gdn.n_k_heads == 0 || gdn.n_v_heads == 0 || gdn.head_k_dim == 0 ||
            gdn.head_v_dim == 0 || gdn.conv_kernel == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid linear attention has a zero dimension (linear_num_key_heads, "
                    "linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, "
                    "linear_conv_kernel_dim)"};
        }
        // q and k are broadcast to the value head count by `repeat_interleave`,
        // which is only defined for an exact multiple. A config that is not one
        // would run the recurrence over a head count that matches neither
        // projection.
        if (gdn.n_v_heads % gdn.n_k_heads != 0) {
            return {StatusCode::InvalidArgument,
                    "linear_num_value_heads " + std::to_string(gdn.n_v_heads) +
                        " is not a multiple of linear_num_key_heads " +
                        std::to_string(gdn.n_k_heads)};
        }

        const auto n_layers = out.topology.n_layers;
        gdn.layer_kinds.clear();
        if (const auto it = j.find("layer_types"); it != j.end() && it->is_array()) {
            if (it->size() != n_layers) {
                return {StatusCode::InvalidArgument,
                        "layer_types has " + std::to_string(it->size()) + " entries for " +
                            std::to_string(n_layers) + " layers"};
            }
            gdn.layer_kinds.reserve(n_layers);
            for (const auto& v : *it) {
                const auto name = v.is_string() ? v.get<std::string>() : std::string{};
                if (name == "linear_attention") {
                    gdn.layer_kinds.push_back(AttnLayerKind::Linear);
                } else if (name == "full_attention") {
                    gdn.layer_kinds.push_back(AttnLayerKind::Full);
                } else {
                    // Not defaulted. `sliding_attention` is a real value in this
                    // slot for other Qwen configs, and quietly calling it Full
                    // would size a full cache for a windowed layer.
                    return {StatusCode::Unsupported,
                            "layer_types contains unsupported kind '" + name + "'"};
                }
            }
        } else {
            const auto interval = get_or<std::uint32_t>(j, "full_attention_interval", 4);
            if (interval == 0) {
                return {StatusCode::InvalidArgument, "full_attention_interval is zero"};
            }
            gdn.layer_kinds.reserve(n_layers);
            for (std::uint32_t i = 0; i < n_layers; ++i) {
                gdn.layer_kinds.push_back((i + 1) % interval == 0 ? AttnLayerKind::Full
                                                                  : AttnLayerKind::Linear);
            }
        }
        if (gdn.n_full_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid stack has no full-attention layer, so nothing carries a KV cache"};
        }

        // The gate rides inside `q_proj` at double width; see
        // AttentionSpec::fused_output_gate for why that is a sizing fact and not
        // a style note.
        attn.fused_output_gate = get_or<bool>(j, "attn_output_gate", false);

        // Only the leading `head_dim * partial_rotary_factor` channels rotate;
        // the rest pass through. Truncated, not rounded — upstream computes
        // `int(head_dim * partial_rotary_factor)`, and at 256 x 0.25 the two
        // agree, which is exactly why a rounding difference here would survive
        // this checkpoint and surface on the next one.
        //
        // Stated in two places by this config; the nested copy is the one
        // `compute_default_rope_parameters` reads, so it wins.
        float rotary_factor = get_or<float>(j, "partial_rotary_factor", 1.0f);
        if (const auto it = j.find("rope_parameters"); it != j.end() && it->is_object()) {
            rotary_factor = get_or<float>(*it, "partial_rotary_factor", rotary_factor);
        }
        if (rotary_factor <= 0.0f || rotary_factor > 1.0f) {
            return {StatusCode::InvalidArgument,
                    "partial_rotary_factor " + std::to_string(rotary_factor) +
                        " is outside (0, 1]"};
        }
        const auto rotary_dim =
            static_cast<std::uint32_t>(static_cast<float>(attn.head_dim) * rotary_factor);
        if (rotary_dim == 0 || rotary_dim % 2 != 0) {
            return {StatusCode::InvalidArgument,
                    "partial rotary width " + std::to_string(rotary_dim) +
                        " is not a positive even number of channels"};
        }
        // `partial_dim == head_dim` and "full rope" are the same rotation, and
        // recording the width either way keeps one meaning for the field.
        attn.rope.partial_dim = rotary_dim;
    }

    // -- BSA ------------------------------------------------------------------
    //
    // Block-sparse key selection over ordinary GQA. Two spellings reach here and
    // both are read, because the checkpoint on the Hub and the config class that
    // loads it disagree about which one is canonical.
    //
    //   nested   `sparse_attention_config: {sparse_num_index_heads,
    //            sparse_index_dim, sparse_block_size, sparse_topk_blocks,
    //            sparse_local_block, sparse_attention_freq}` -- what MiniMax-M3
    //            ships.
    //   flat     `index_n_heads, index_head_dim, index_block_size,
    //            index_topk_blocks, index_local_blocks, layer_types` -- what
    //            `MiniMaxM3VLTextConfig.__post_init__` rewrites it into, and
    //            therefore what a config re-saved by transformers carries.
    //
    // The flat form WINS where both exist, for the same reason the nested
    // `rope_parameters` wins over a top-level `rope_theta`: it is the value the
    // reference implementation actually executes against. The tiny fixture is
    // saved by transformers and carries only the flat form; the production
    // checkpoint carries only the nested one. Reading one spelling would have
    // meant the fixture and the model it stands for were not the same family.
    if (attn.family == AttentionFamily::GqaBsa) {
        auto& b = attn.bsa;
        const json* sparse = nullptr;
        if (const auto it = j.find("sparse_attention_config");
            it != j.end() && it->is_object()) {
            sparse = &(*it);
        }
        const auto nested = [&](const char* key, std::uint32_t fallback) {
            return sparse != nullptr ? get_or<std::uint32_t>(*sparse, key, fallback) : fallback;
        };

        b.n_index_heads =
            get_or<std::uint32_t>(j, "index_n_heads", nested("sparse_num_index_heads", 0));
        b.index_head_dim =
            get_or<std::uint32_t>(j, "index_head_dim", nested("sparse_index_dim", 0));
        b.block_size =
            get_or<std::uint32_t>(j, "index_block_size", nested("sparse_block_size", 0));
        b.topk_blocks =
            get_or<std::uint32_t>(j, "index_topk_blocks", nested("sparse_topk_blocks", 0));
        // Zero is a REAL value here -- no forced-local block at all -- so it is
        // not folded into the "absent means broken" check the four above get.
        b.local_blocks =
            get_or<std::uint32_t>(j, "index_local_blocks", nested("sparse_local_block", 0));

        if (b.n_index_heads == 0 || b.index_head_dim == 0 || b.block_size == 0 ||
            b.topk_blocks == 0) {
            return {StatusCode::InvalidArgument,
                    "block-sparse attention has a zero dimension (index_n_heads, "
                    "index_head_dim, index_block_size, index_topk_blocks)"};
        }

        // Which layers own an indexer. Stated per layer, ZERO-based, and stated
        // twice -- `layer_types` naming the kind, `sparse_attention_freq` as a
        // 0/1 mask. Same argument as `Topology::layer_kinds`: a stride
        // re-derived downstream is a disagreement waiting to happen, and here
        // there is no stride to re-derive from anyway -- the pattern is "the
        // first three are dense, the rest are sparse", which no interval
        // expresses.
        const auto n_layers = out.topology.n_layers;
        b.layer_kinds.clear();
        if (const auto it = j.find("layer_types"); it != j.end() && it->is_array()) {
            if (it->size() != n_layers) {
                return {StatusCode::InvalidArgument,
                        "layer_types has " + std::to_string(it->size()) + " entries for " +
                            std::to_string(n_layers) + " layers"};
            }
            b.layer_kinds.reserve(n_layers);
            for (const auto& v : *it) {
                const auto name = v.is_string() ? v.get<std::string>() : std::string{};
                if (name == "minimax_m3_sparse") {
                    b.layer_kinds.push_back(IndexerKind::Full);
                } else if (name == "full_attention") {
                    b.layer_kinds.push_back(IndexerKind::None);
                } else {
                    // Not defaulted, for the reason the GDN adapter gives:
                    // `sliding_attention` is a real value in this slot for other
                    // configs, and quietly calling it dense would attend over a
                    // prefix a windowed layer cannot see.
                    return {StatusCode::Unsupported,
                            "layer_types contains unsupported kind '" + name + "'"};
                }
            }
        } else if (sparse != nullptr) {
            if (const auto freq = sparse->find("sparse_attention_freq");
                freq != sparse->end() && freq->is_array()) {
                if (freq->size() != n_layers) {
                    return {StatusCode::InvalidArgument,
                            "sparse_attention_freq has " + std::to_string(freq->size()) +
                                " entries for " + std::to_string(n_layers) + " layers"};
                }
                b.layer_kinds.reserve(n_layers);
                for (const auto& v : *freq) {
                    b.layer_kinds.push_back(v.is_number() && v.get<double>() != 0.0
                                                ? IndexerKind::Full
                                                : IndexerKind::None);
                }
            }
        }
        if (b.layer_kinds.empty()) {
            // Refused rather than defaulted either way. "All dense" is a model
            // that runs and is not this one; "all sparse" binds indexer tensors
            // the leading layers do not have. Both are decisions the config is
            // supposed to make, and neither is one to make on its behalf.
            return {StatusCode::InvalidArgument,
                    "block-sparse attention declared with neither layer_types nor "
                    "sparse_attention_config.sparse_attention_freq"};
        }
        if (b.n_indexed_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "no layer owns a block-sparse indexer, so the family is plain gqa"};
        }
        // One selection per GQA group: query head `h` obeys indexer head
        // `h / (n_heads / n_index_heads)`. A count that does not divide the query
        // heads has no such mapping, and the plausible repair -- reusing head 0
        // for the remainder -- would silently give some heads another group's
        // blocks.
        if (attn.n_heads % b.n_index_heads != 0) {
            return {StatusCode::InvalidArgument,
                    "n_heads " + std::to_string(attn.n_heads) +
                        " is not a multiple of index_n_heads " +
                        std::to_string(b.n_index_heads)};
        }

        // The rotation, which this family states TWICE and whose two statements
        // agree -- verified against transformers 5.15.1, where `rope_parameters`
        // resolves to `{rope_theta: 5e6, partial_rotary_factor: 0.5}` and
        // `compute_default_rope_parameters` therefore rotates
        // `int(128 * 0.5) == 64` of each 128-wide head.
        //
        // `partial_rotary_factor` is the one the rope reads; `rotary_dim` is a
        // declared field nothing in the rope path consults. So the factor wins
        // and `rotary_dim` is a CROSS-CHECK rather than a second source: a config
        // where they disagree has two opinions about the single quantity that
        // decides whether this model's long-context behaviour is right, and
        // picking one would be answering that by coin flip.
        float rotary_factor = get_or<float>(j, "partial_rotary_factor", 1.0f);
        if (const auto it = j.find("rope_parameters"); it != j.end() && it->is_object()) {
            rotary_factor = get_or<float>(*it, "partial_rotary_factor", rotary_factor);
        }
        if (rotary_factor <= 0.0f || rotary_factor > 1.0f) {
            return {StatusCode::InvalidArgument,
                    "partial_rotary_factor " + std::to_string(rotary_factor) +
                        " is outside (0, 1]"};
        }
        const auto rotary =
            static_cast<std::uint32_t>(static_cast<float>(attn.head_dim) * rotary_factor);
        if (rotary == 0 || rotary % 2 != 0) {
            return {StatusCode::InvalidArgument,
                    "partial rotary width " + std::to_string(rotary) +
                        " is not a positive even number of channels"};
        }
        if (const auto stated = get_or<std::uint32_t>(j, "rotary_dim", 0);
            stated != 0 && stated != rotary) {
            return {StatusCode::InvalidArgument,
                    "rotary_dim " + std::to_string(stated) + " disagrees with head_dim " +
                        std::to_string(attn.head_dim) + " x partial_rotary_factor " +
                        std::to_string(rotary_factor)};
        }
        attn.rope.partial_dim = rotary;
    }

    out.block_residual.block_size = get_or<std::uint32_t>(j, "attn_res_block_size", 0);

    if (attn.family == AttentionFamily::CompressedSparse) {
        auto& c = attn.compressed;
        c.q_lora_rank = get_or<std::uint32_t>(j, "q_lora_rank", 0);
        c.rope_head_dim = get_or<std::uint32_t>(j, "qk_rope_head_dim", 0);
        c.o_groups = get_or<std::uint32_t>(j, "o_groups", 0);
        c.o_lora_rank = get_or<std::uint32_t>(j, "o_lora_rank", 0);
        c.index_n_heads = get_or<std::uint32_t>(j, "index_n_heads", 0);
        c.index_head_dim = get_or<std::uint32_t>(j, "index_head_dim", 0);
        c.index_topk = get_or<std::uint32_t>(j, "index_topk", 0);
        c.compress_rope_theta = get_or<float>(j, "compress_rope_theta", 10000.0f);
        c.semantic_fp8_quant_dequant =
            get_or<bool>(j, "semantic_fp8_quant_dequant", true);
        c.semantic_fp4_quant_dequant =
            get_or<bool>(j, "semantic_fp4_quant_dequant", true);
        if (const auto it = j.find("compress_ratios"); it != j.end() && it->is_array()) {
            c.compress_ratios.reserve(out.topology.n_layers);
            for (std::size_t i = 0; i < it->size() && i < out.topology.n_layers; ++i) {
                c.compress_ratios.push_back((*it)[i].get<std::uint32_t>());
            }
        }
        // Every V4 attention layer owns a single shared KV head.  The upstream
        // config states this too, but recording the architecture rule here makes
        // a malformed config fail validation rather than select a different
        // cache layout.
        attn.rope.partial_dim = c.rope_head_dim;
        attn.sliding_window = get_or<std::uint32_t>(j, "sliding_window", 0);
    }

    if (get_or<bool>(j, "use_sliding_window", false)) {
        attn.sliding_window = get_or<std::uint32_t>(j, "sliding_window", 0);
    }
    if (auto st = parse_rope(j, attn.rope); !st.ok()) return st;

    // NoPE, applied AFTER parse_rope and not before it.
    //
    // The slice stays; the ROTATION goes. `qk_rope_head_dim` is still 64 and
    // still concatenated into every query and cached key, so `head_dim` and the
    // cache width above are unchanged — see MlaSpec::nope.
    //
    // What goes is any claim about an angle. `RopeConfig{}` does NOT express
    // that: its default theta is 10000, so a default-constructed config asserts
    // "rotate at 10000" as confidently as a stated one. theta is inside
    // arch_hash, so leaving either the default or the parsed value would make
    // "rotated" and "not rotated at all" hash identically for two models that
    // are not the same model — and would hand a backend a plausible angle to
    // rotate by.
    //
    // Ordering is the whole of the fix: written before parse_rope, this was
    // overwritten by it and had no effect at all.
    if (attn.mla.nope) {
        attn.rope = RopeConfig{};
        attn.rope.theta = 0.0f;
        attn.rope.partial_dim = 0;
    }

    // ── router ───────────────────────────────────────────────────────────────
    auto& router = out.router;
    if (const json* n = find_first(j, kExpertCountKeys)) {
        router.n_experts = n->get<std::uint32_t>();
    }
    if (const json* k = find_first(j, kTopKKeys)) router.top_k = k->get<std::uint32_t>();
    const json* score = find_first(j, kScoringKeys);
    const auto scoring = (score != nullptr && score->is_string()) ? score->get<std::string>()
                                                                 : std::string("softmax");
    router.score_fn = scoring == "sigmoid"       ? ScoreFn::Sigmoid
                      : scoring == "sqrtsoftplus" ? ScoreFn::SqrtSoftplus
                                                   : ScoreFn::Softmax;
    const json* norm = find_first(j, kNormTopkKeys);
    router.normalize_topk =
        traits.force_normalize_topk || (norm != nullptr && norm->is_boolean() && norm->get<bool>());
    router.routed_scaling_factor = get_or<float>(j, "routed_scaling_factor", 1.0f);
    // Two ways of declaring the same mechanism: a per-expert bias that steers
    // SELECTION and never reaches the retained weights.
    //
    // DeepSeek names the algorithm (`topk_method: noaux_tc`); MiniMax-M3 states
    // the fact (`use_routing_bias: true`) and names no method at all. Reading
    // only the method leaves `bias_correction` false on a model whose router
    // adds `e_score_correction_bias` before its top-k -- the tensor is present
    // and bound, so nothing fails; a different set of experts simply fires.
    router.bias_correction = (get_or<std::string>(j, "topk_method", "greedy") == "noaux_tc") ||
                             get_or<bool>(j, "use_routing_bias", false);
    router.n_groups = 1;
    if (const json* g = find_first(j, kExpertGroupKeys))
        router.n_groups = std::max<std::uint32_t>(1, g->get<std::uint32_t>());
    router.topk_group = std::max<std::uint32_t>(1, get_or<std::uint32_t>(j, "topk_group", 1));
    router.n_shared_experts = 0;
    if (const json* sh = find_first(j, kSharedExpertKeys))
        router.n_shared_experts = sh->get<std::uint32_t>();
    router.n_hash_layers = get_or<std::uint32_t>(j, "num_hash_layers", 0);

    // ── ffn ──────────────────────────────────────────────────────────────────
    auto& ffn = out.ffn;
    const auto act = get_or<std::string>(j, "hidden_act", "silu");
    if (act == "silu" || act == "swish") {
        ffn.activation = Activation::SwiGlu;
    } else if (act == "gelu" || act == "gelu_new" || act == "gelu_pytorch_tanh") {
        ffn.activation = Activation::GeGlu;
    } else if (act == "relu2") {
        ffn.activation = Activation::Relu2;
    } else if (act == "swigluoai") {
        ffn.activation = Activation::SwiGluOai;
    } else if (act == "situ") {
        ffn.activation = Activation::Situ;
        // `beta or 1.0` upstream: a stated 0 is treated as absent, because
        // dividing by it is the only other reading and it is not one.
        const auto beta = get_or<float>(j, "activation_situ_beta", 0.0f);
        ffn.situ_beta = beta != 0.0f ? beta : 1.0f;
        // The linear half is transformed ONLY when its beta is stated. Absent
        // means the up projection passes through untouched — not "beta 0", which
        // would multiply the whole FFN output by zero.
        ffn.situ_linear_beta = get_or<float>(j, "activation_situ_linear_beta", 0.0f);
    } else {
        return {StatusCode::Unsupported, "unsupported hidden_act '" + act + "'"};
    }
    ffn.has_gate = true;
    // The shared spelling, and the base both widths fall back to.
    const auto intermediate = get_or<std::uint32_t>(j, "intermediate_size", 0);
    // A family that states the dense width separately wins; every family that
    // does not gets `intermediate_size`, exactly as before this key existed --
    // which is why no existing container's arch_hash moves.
    ffn.dense_intermediate = intermediate;
    if (const json* dn = find_first(j, kDenseIntermediateKeys)) {
        ffn.dense_intermediate = dn->get<std::uint32_t>();
    }
    if (const json* m = find_first(j, kMoeIntermediateKeys)) {
        ffn.expert_intermediate = m->get<std::uint32_t>();
    } else {
        // Mixtral and OLMoE have no moe_intermediate_size: intermediate_size IS
        // the expert width. Read from `intermediate` rather than from
        // `dense_intermediate`, which is the same number for them and is NOT for
        // a family that states a separate dense width.
        ffn.expert_intermediate = intermediate;
    }
    // Two families, two ways of saying the same thing.
    //
    // Qwen2-MoE STATES the shared expert's width in `shared_expert_intermediate_size`.
    // DeepSeek DERIVES it as `moe_intermediate_size * n_shared_experts` and ships
    // no such key. Reading only the stated form gave DeepSeek a shared width of
    // zero — the tensors bind, the width is wrong, and the logits come out
    // finite and off by ~0.1 mean, which reads as a subtle attention bug rather
    // than a config one.
    ffn.shared_intermediate = 0;
    if (const json* sw = find_first(j, kSharedIntermediateKeys)) {
        ffn.shared_intermediate = sw->get<std::uint32_t>();
    }
    if (ffn.shared_intermediate == 0 && router.n_shared_experts > 0) {
        ffn.shared_intermediate = ffn.expert_intermediate * router.n_shared_experts;
    }
    // ...and the CONVERSE, which was missing and cost the Qwen families their
    // shared expert entirely.
    //
    // DeepSeek states the count and derives the width; Qwen2-MoE and Qwen3.5
    // state the WIDTH and never state a count at all. Both `f32_bind_layer` and
    // the planner gate the shared expert on `n_shared_experts > 0`, so a
    // width-only config bound no shared tensors and charged no shared bytes —
    // the branch simply did not exist, on a model that has one on every MoE
    // layer. Nothing failed to load; the FFN was just missing a term.
    //
    // One, not more: `shared_intermediate` is the FUSED width of however many
    // shared experts there are (see the planner's note on why multiplying by the
    // count again squares it), and a config that states only a width describes
    // exactly one such fused block.
    //
    // This CHANGES arch_hash for any qwen2_moe container admitted before now,
    // which is correct — those containers describe a model missing an FFN
    // branch, and re-admitting them is the point.
    if (router.n_shared_experts == 0 && ffn.shared_intermediate > 0) {
        router.n_shared_experts = 1;
    }
    ffn.shared_expert_gate = traits.gated_shared_expert && router.n_shared_experts > 0;
    ffn.expert_layout = ExpertLayout::InterleavedGateUpDown;
    ffn.swiglu_limit = get_or<float>(j, "swiglu_limit", 0.0f);
    // Defaults to 1 -- the value at which the gate half degenerates to SiLU --
    // rather than to any family's constant. See FfnSpec::swiglu_alpha.
    ffn.swiglu_alpha = get_or<float>(j, "swiglu_alpha", 1.0f);

    // The activation is FAMILY knowledge for this one, and `hidden_act` is not
    // merely unhelpful here -- it is actively misleading.
    //
    // MiniMax-M3's checkpoint declares `"hidden_act": "swigluoai"`, which the
    // clause above reads correctly. But `MiniMaxM3VLTextConfig.__post_init__`
    // OVERWRITES it with `"silu"` and says why: the gate is computed inline from
    // `swiglu_alpha`/`swiglu_limit`, and `hidden_act` has to stay a real ACT2FN
    // key. So a config re-saved by transformers -- which is exactly what the
    // tiny fixture is -- carries "silu" and would select plain SwiGLU: no alpha
    // inside the sigmoid, no `+1` on the linear half, a different function on
    // every token of every layer, and a fixture that is not the model it stands
    // for. Stated here so that both spellings land on the same architecture.
    if (attn.family == AttentionFamily::GqaBsa) {
        ffn.activation = Activation::SwiGluOai;
    }

    // Latent MoE: the routed experts live in a NARROWER space than the residual
    // stream. Recorded only when it actually differs, so that a config stating
    // `routed_expert_hidden_size == hidden_size` is identical to one omitting
    // it — including in arch_hash, which is conditional on this being non-zero.
    if (const auto rh = get_or<std::uint32_t>(j, "routed_expert_hidden_size", 0);
        rh != 0 && rh != out.topology.d_model) {
        ffn.routed_expert_hidden = rh;
        ffn.routed_expert_norm = get_or<bool>(j, "latent_moe_use_norm", false);
    }

    out.hyper_connections.multiplier = get_or<std::uint32_t>(j, "hc_mult", 1);
    out.hyper_connections.sinkhorn_iters = get_or<std::uint32_t>(j, "hc_sinkhorn_iters", 0);
    out.hyper_connections.eps = get_or<float>(j, "hc_eps", 1e-6f);

    // V4 ships the DSpark descriptor in config.json even when a Soma
    // conversion intentionally omitted every mtp.* tensor.  Parse the source
    // declaration here, but do not turn it into a runtime capability: only the
    // atomic container metadata overlay below may set `present`.
    if (model_type == "deepseek_v4") {
        auto& d = out.speculative;
        d.method = SpeculativeMethod::DSpark;
        d.trained_block_size = get_or<std::uint32_t>(j, "dspark_block_size", 0);
        d.noise_token_id = get_or<TokenId>(j, "dspark_noise_token_id", 0);
        d.markov_rank = get_or<std::uint32_t>(j, "dspark_markov_rank", 0);
        if (const auto it = j.find("dspark_target_layer_ids");
            it != j.end() && it->is_array()) {
            for (const auto& layer : *it) d.target_layer_ids.push_back(layer.get<LayerIndex>());
        }
        d.n_layers = static_cast<std::uint32_t>(d.target_layer_ids.size());
        d.source_declared = d.n_layers > 0 || d.trained_block_size > 0 || d.markov_rank > 0;
    }

    // Qwen3.5 carries a one-layer multi-token-prediction head under a `mtp.*`
    // prefix — `mtp.fc`, one ordinary decoder layer, and two pre-projection
    // norms. transformers 5.15.1 does not implement it, and neither does Soma.
    //
    // Recorded anyway, because the alternative is silence. `mtp.*` is a
    // TOP-LEVEL prefix, not a `model.layers.<N>` name, so it is invisible to the
    // layer-index rule that excludes GLM-5.2's MTP layer and the converter has
    // to be told about it either way. An operator can see the head in the
    // checkpoint index; a plan that never mentions it is the plan being quiet
    // about a real part of the model. `present` stays false — see
    // SpeculativeSpec — so nothing advertises a draft it cannot run.
    if (attn.family == AttentionFamily::GqaGdn) {
        if (const auto n = get_or<std::uint32_t>(j, "mtp_num_hidden_layers", 0); n > 0) {
            auto& d = out.speculative;
            d.method = SpeculativeMethod::Mtp;
            d.n_layers = n;
            d.source_declared = true;
        }
    }

    // MiniMax-M3 carries one too, and states its size with two keys that mean
    // different things: `num_mtp_modules` 7 is how many draft heads were
    // TRAINED, `num_nextn_predict_layers` 1 is how many decoder layers each one
    // is. The layer count is the one comparable to the field above and to
    // GLM-5.2's, so it is what `n_layers` records; the module count would make
    // this head look seven times the size it is.
    //
    // Transformers ignores the tensors outright
    // (`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`) and so does
    // Soma. Recorded anyway rather than silently dropped: an operator can see
    // `mtp.*` in the checkpoint index, and `present` stays false so nothing
    // advertises a draft model it cannot run.
    if (attn.family == AttentionFamily::GqaBsa) {
        const auto modules = get_or<std::uint32_t>(j, "num_mtp_modules", 0);
        const auto per_module = get_or<std::uint32_t>(j, "num_nextn_predict_layers", 0);
        if (modules > 0 || per_module > 0) {
            auto& d = out.speculative;
            d.method = SpeculativeMethod::Mtp;
            d.n_layers = per_module;
            d.source_declared = true;
        }
    }

    // G0/G1 read fp32 HF checkpoints directly; requantization happens at
    // admission, not here.
    out.quantization = QuantMap{};

    return validate_arch_ir(out);
}

Status validate_arch_ir(const ArchIr& ir) {
    if (ir.schema_version != kArchIrSchemaVersion &&
        ir.schema_version != kArchIrSchemaVersionV2) {
        return {StatusCode::VersionMismatch,
                "arch IR schema " + std::to_string(ir.schema_version) +
                    " is not supported"};
    }
    if (ir.topology.n_layers == 0 || ir.topology.d_model == 0 || ir.topology.vocab_size == 0) {
        return {StatusCode::InvalidArgument, "topology has a zero dimension"};
    }
    if (ir.topology.layer_kinds.size() != ir.topology.n_layers) {
        return {StatusCode::InvalidArgument,
                "layer_kinds has " + std::to_string(ir.topology.layer_kinds.size()) +
                    " entries for " + std::to_string(ir.topology.n_layers) + " layers"};
    }
    if (ir.attention.family == AttentionFamily::Unknown) {
        return {StatusCode::InvalidArgument, "attention family unresolved"};
    }
    if (ir.attention.n_heads == 0 || ir.attention.head_dim == 0) {
        return {StatusCode::InvalidArgument, "attention has a zero dimension"};
    }
    if (ir.attention.n_kv_heads == 0 || ir.attention.n_heads % ir.attention.n_kv_heads != 0) {
        return {StatusCode::InvalidArgument,
                "n_heads " + std::to_string(ir.attention.n_heads) +
                    " is not a multiple of n_kv_heads " + std::to_string(ir.attention.n_kv_heads)};
    }
    if (ir.attention.family == AttentionFamily::MlaKda) {
        const auto& k = ir.attention.kda;
        if (k.layer_kinds.size() != ir.topology.n_layers) {
            return {StatusCode::InvalidArgument,
                    "kda layer_kinds has " + std::to_string(k.layer_kinds.size()) +
                        " entries for " + std::to_string(ir.topology.n_layers) + " layers"};
        }
        if (k.n_heads == 0 || k.head_dim == 0 || k.conv_kernel < 2) {
            return {StatusCode::InvalidArgument, "hybrid linear attention is incomplete"};
        }
        // No full-attention layer means no KV cache anywhere, which the planner
        // would report as a free context and the checkpoint reader as an empty
        // cache. Both are wrong in the optimistic direction.
        if (k.n_full_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid stack has no full-attention layer"};
        }
        // The full-attention layers are MLA, so their dimensions must be real.
        // A hybrid that reached here with a zero kv_lora_rank would size a
        // zero-byte cache for 24 layers that need one.
        const auto& m = ir.attention.mla;
        if (m.kv_lora_rank == 0 || m.qk_nope_head_dim == 0 || m.v_head_dim == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid full-attention layers have no MLA dimensions"};
        }
    }
    if (ir.attention.family == AttentionFamily::GqaGdn) {
        const auto& g = ir.attention.gdn;
        if (g.layer_kinds.size() != ir.topology.n_layers) {
            return {StatusCode::InvalidArgument,
                    "gdn layer_kinds has " + std::to_string(g.layer_kinds.size()) +
                        " entries for " + std::to_string(ir.topology.n_layers) + " layers"};
        }
        if (g.n_k_heads == 0 || g.n_v_heads == 0 || g.head_k_dim == 0 || g.head_v_dim == 0 ||
            g.conv_kernel < 2) {
            return {StatusCode::InvalidArgument, "hybrid linear attention is incomplete"};
        }
        if (g.n_v_heads % g.n_k_heads != 0) {
            return {StatusCode::InvalidArgument,
                    "gdn value heads " + std::to_string(g.n_v_heads) +
                        " is not a multiple of key heads " + std::to_string(g.n_k_heads)};
        }
        // Same reason as the hybrid above: no full-attention layer means no KV
        // cache anywhere, and the planner would report the context as free.
        if (g.n_full_layers() == 0) {
            return {StatusCode::InvalidArgument,
                    "hybrid stack has no full-attention layer"};
        }
        // The full-attention layers are GQA, so the rotation must be a real
        // even-width slice of a real head. A zero here would rotate nothing on
        // a model whose long-context behaviour is entirely positional.
        const auto rot = ir.attention.rope.partial_dim;
        if (rot == 0 || rot > ir.attention.head_dim || rot % 2 != 0) {
            return {StatusCode::InvalidArgument,
                    "gdn full-attention rotary width " + std::to_string(rot) +
                        " is not an even slice of head_dim " +
                        std::to_string(ir.attention.head_dim)};
        }
    }
    if (ir.attention.family == AttentionFamily::GqaBsa) {
        const auto& b = ir.attention.bsa;
        if (b.layer_kinds.size() != ir.topology.n_layers) {
            return {StatusCode::InvalidArgument,
                    "bsa layer_kinds has " + std::to_string(b.layer_kinds.size()) +
                        " entries for " + std::to_string(ir.topology.n_layers) + " layers"};
        }
        if (b.n_index_heads == 0 || b.index_head_dim == 0 || b.block_size == 0 ||
            b.topk_blocks == 0) {
            return {StatusCode::InvalidArgument, "block-sparse attention is incomplete"};
        }
        if (ir.attention.n_heads % b.n_index_heads != 0) {
            return {StatusCode::InvalidArgument,
                    "n_heads " + std::to_string(ir.attention.n_heads) +
                        " is not a multiple of index_n_heads " +
                        std::to_string(b.n_index_heads)};
        }
        // No indexed layer means the model is plain GQA and this family is a
        // mislabel -- which matters because the mislabel is in the OPTIMISTIC
        // direction for arithmetic and the pessimistic one for cache: the
        // planner would charge an indexer key plane nothing writes.
        if (b.n_indexed_layers() == 0) {
            return {StatusCode::InvalidArgument, "no layer owns a block-sparse indexer"};
        }
        // A local guarantee wider than the selection cannot be satisfied:
        // upstream forces the local blocks by writing `+inf` into their scores
        // BEFORE a top-k of `topk_blocks`, so more local blocks than slots would
        // silently drop the furthest of them rather than growing the selection.
        if (b.local_blocks > b.topk_blocks) {
            return {StatusCode::InvalidArgument,
                    "index_local_blocks " + std::to_string(b.local_blocks) +
                        " exceeds index_topk_blocks " + std::to_string(b.topk_blocks)};
        }
        // The indexer rotates the same slice the main attention does, so the
        // slice must be real and must fit inside the indexer's own head.
        const auto rot = ir.attention.rope.partial_dim;
        if (rot == 0 || rot > ir.attention.head_dim || rot % 2 != 0) {
            return {StatusCode::InvalidArgument,
                    "bsa rotary width " + std::to_string(rot) +
                        " is not an even slice of head_dim " +
                        std::to_string(ir.attention.head_dim)};
        }
        if (rot > b.index_head_dim) {
            // Upstream slices the rope table as `cos[..., :index_head_dim]`, so a
            // narrower indexer head would truncate the table mid-frequency and
            // pair channels with rotations that are not theirs. It runs. It is
            // not a rotation. Refused rather than transcribed, because no
            // checkpoint in the wild does this and a fixture that did would be
            // grading Soma against a reference bug.
            return {StatusCode::Unsupported,
                    "index_head_dim " + std::to_string(b.index_head_dim) +
                        " is narrower than the rotary width " + std::to_string(rot)};
        }
    }
    if (ir.ffn.routed_expert_hidden != 0 && ir.n_moe_layers() > 0 &&
        ir.ffn.routed_expert_hidden > ir.topology.d_model) {
        // A latent MoE projects DOWN. Wider than the residual stream is not a
        // latent space, and taking it at face value would inflate every
        // expert — the one number the verdict is computed from.
        return {StatusCode::InvalidArgument,
                "routed_expert_hidden " + std::to_string(ir.ffn.routed_expert_hidden) +
                    " exceeds d_model " + std::to_string(ir.topology.d_model)};
    }
    if (ir.attention.family == AttentionFamily::CompressedSparse) {
        if (ir.schema_version != kArchIrSchemaVersionV2) {
            return {StatusCode::VersionMismatch, "compressed sparse attention requires arch IR v2"};
        }
        const auto& c = ir.attention.compressed;
        if (c.compress_ratios.size() != ir.topology.n_layers) {
            return {StatusCode::InvalidArgument,
                    "compress_ratios has " + std::to_string(c.compress_ratios.size()) +
                        " entries for " + std::to_string(ir.topology.n_layers) + " layers"};
        }
        if (c.q_lora_rank == 0 || c.rope_head_dim == 0 || c.rope_head_dim > ir.attention.head_dim ||
            c.o_groups == 0 || ir.attention.n_heads % c.o_groups != 0 ||
            c.o_lora_rank == 0 || c.index_n_heads == 0 || c.index_head_dim == 0 ||
            c.index_topk == 0 || ir.attention.sliding_window == 0) {
            return {StatusCode::InvalidArgument, "compressed sparse attention is incomplete"};
        }
        for (const auto ratio : c.compress_ratios) {
            if (ratio != 4 && ratio != 128) {
                return {StatusCode::InvalidArgument,
                        "base-model compression ratio must be 4 or 128"};
            }
        }
        if (ir.hyper_connections.multiplier < 2 || ir.hyper_connections.sinkhorn_iters == 0) {
            return {StatusCode::InvalidArgument,
                    "compressed sparse attention requires configured hyper-connections"};
        }
        if (ir.router.n_hash_layers > ir.topology.n_layers) {
            return {StatusCode::InvalidArgument, "n_hash_layers exceeds n_layers"};
        }
    }
    if (ir.n_moe_layers() > 0) {
        if (ir.router.n_experts == 0 || ir.router.top_k == 0) {
            return {StatusCode::InvalidArgument, "MoE layers present but router is unconfigured"};
        }
        if (ir.router.top_k > ir.router.n_experts) {
            return {StatusCode::InvalidArgument,
                    "top_k " + std::to_string(ir.router.top_k) + " > n_experts " +
                        std::to_string(ir.router.n_experts)};
        }
        if (ir.router.n_groups > 1 && ir.router.n_experts % ir.router.n_groups != 0) {
            return {StatusCode::InvalidArgument,
                    "n_groups " + std::to_string(ir.router.n_groups) +
                        " does not divide n_experts"};
        }
        if (ir.router.topk_group > ir.router.n_groups) {
            return {StatusCode::InvalidArgument, "topk_group > n_groups"};
        }
        if (ir.ffn.expert_intermediate == 0) {
            return {StatusCode::InvalidArgument, "expert_intermediate is zero"};
        }
    }
    if (ir.router.n_shared_experts > 0 && ir.ffn.shared_intermediate == 0 &&
        ir.ffn.expert_intermediate == 0) {
        return {StatusCode::InvalidArgument, "shared experts declared with no width"};
    }
    if (ir.speculative.present) {
        const auto& d = ir.speculative;
        if (ir.schema_version != kArchIrSchemaVersionV2 ||
            ir.attention.family != AttentionFamily::CompressedSparse) {
            return {StatusCode::InvalidArgument,
                    "DSpark requires a compressed-sparse Architecture IR v2 target"};
        }
        if (!d.source_declared || d.n_layers != 3 || d.target_layer_ids.size() != d.n_layers ||
            d.trained_block_size == 0 || d.markov_rank == 0 || !d.confidence_head ||
            d.routed_bytes == 0 || d.resident_bytes == 0 || d.expert_bytes == 0 ||
            d.kv_bytes_per_sequence == 0) {
            return {StatusCode::InvalidArgument, "DSpark descriptor is incomplete"};
        }
        for (const auto layer : d.target_layer_ids) {
            if (layer >= ir.topology.n_layers) {
                return {StatusCode::InvalidArgument, "DSpark target layer is out of range"};
            }
        }
        if (d.noise_token_id >= ir.topology.vocab_size) {
            return {StatusCode::InvalidArgument, "DSpark noise token is out of range"};
        }
    }
    return validate_quant_map(ir.quantization);
}

Status parse_arch_ir(std::string_view text, ArchIr& out) {
    // The registry persists a canonical JSON form, but the model loader adapts
    // config.json directly. Keep that as the one tested load path until a
    // registry consumer needs this parser.
    (void)text;
    (void)out;
    return {StatusCode::Unsupported,
            "canonical registry architecture JSON parsing is not implemented; "
            "use adapt_hf_config() to read an upstream config.json"};
}

Status apply_container_quant(std::string_view meta_json, ArchIr& io) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(meta_json);
    } catch (const std::exception& e) {
        return {StatusCode::InvalidArgument,
                std::string("container_meta.json is not valid JSON: ") + e.what()};
    }

    const auto group = j.value("group", 0u);
    // Conversion identity is metadata rather than architecture, but carrying it
    // in the IR lets the plan name the exact artifact it describes. Overlay
    // JSON omits these keys and therefore leaves the source unchanged.
    if (const auto repo = j.value("source_repo", std::string{}); !repo.empty())
        io.source_repo = repo;
    if (const auto revision = j.value("source_revision", std::string{}); !revision.empty())
        io.source_revision = revision;
    // An UNNAMED role is left entirely alone — dtype and group.
    //
    // The group used to be applied unconditionally, which was harmless while only
    // the three expert roles were settable: a real container_meta always carries
    // `group` alongside `dtype_gate_up`, so the two arrived together. The moment
    // `dtype_dense` made embed/attn_proj/shared_expert settable, the same line
    // began stamping `group` onto them even when no dense dtype was asked for.
    //
    // That is not cosmetic. arch_hash covers dtype AND group for EVERY role, so
    // it silently changed the hash of every already-admitted container — the
    // registry would read them as StaleRecord and route to the fallback, and KV
    // checkpoints keyed on the hash would stop loading. Caught by comparing a
    // real admitted model's stored hash against a freshly computed one, which is
    // the only check that could have caught it.
    const auto set = [&](QuantSpec& spec, const std::string& name) {
        if (name.empty()) return;
        DType d{};
        if (parse_dtype(name, d)) spec.dtype = d;
        if (group > 0) spec.group = group;
    };

    // gate and up share a dtype because they are converted together, interleaved
    // into one expert range. Splitting them here would describe a container the
    // converter cannot produce.
    const auto gate_up = j.value("dtype_gate_up", std::string{});
    set(io.quantization.expert_gate, gate_up);
    set(io.quantization.expert_up, gate_up);
    set(io.quantization.expert_down, j.value("dtype_down", std::string{}));

    // The DENSE half, which nothing could previously ask about.
    //
    // The loader has always been able to quantize these — bind_weight reads the
    // role's spec and quantizes at load — but only the three expert roles were
    // ever settable, so embeddings, attention projections and shared experts
    // stayed F32 by omission rather than by decision. For a 744B model with 78
    // MLA layers and a 155k-token vocabulary that omission is tens of gigabytes
    // of resident memory (roadmap D17).
    //
    // DISK stays F32 either way: dense.safetensors holds full precision and the
    // loader quantizes into RAM. That is a feature rather than a compromise —
    // the resident precision can be changed without reconverting a single byte,
    // which is exactly what the expert half cannot do.
    //
    // Router is deliberately absent. `TensorRole::Router` "MUST be F32.
    // Enforced at admission, not by convention" — one f32 matrix per layer is
    // negligible next to the embeddings, and a quantized router changes which
    // experts fire.
    const auto dense = j.value("dtype_dense", std::string{});
    set(io.quantization.embed, dense);
    set(io.quantization.attn_proj, dense);
    set(io.quantization.shared_expert, dense);
    if (j.value("dspark", std::string{}) == "present") {
        auto& d = io.speculative;
        d.present = true;
        d.confidence_head = j.value("dspark_confidence_head", false);
        d.routed_bytes = j.value("dspark_total_expert_bytes", std::uint64_t{0});
        d.resident_bytes = j.value("dspark_resident_bytes", std::uint64_t{0});
        d.expert_bytes = j.value("dspark_expert_bytes", std::uint64_t{0});
        d.kv_bytes_per_sequence =
            j.value("dspark_kv_bytes_per_sequence", std::uint64_t{0});
        d.profiled_speedup = j.value("dspark_profiled_speedup", 0.0f);
        set(io.quantization.draft_head,
            j.value("dtype_dspark", j.value("dtype_dense", std::string{})));
    }
    return {};
}

Status compute_arch_hash(const ArchIr& ir, std::string& out_hash) {
    // Covers §2–§6 (what the model IS) and deliberately NOT `economics`:
    // re-profiling on faster disks must not invalidate KV checkpoints, while
    // requantization must — and QuantMap is inside this.
    std::ostringstream canon;
    canon << "v" << ir.schema_version << "|arch=" << to_string(ir.attention.family)
          << "|L=" << ir.topology.n_layers << "|d=" << ir.topology.d_model
          << "|V=" << ir.topology.vocab_size << "|kinds=";
    for (const auto k : ir.topology.layer_kinds)
        canon << (k == LayerKind::Moe ? 'm' : 'd');
    canon << "|h=" << ir.attention.n_heads << "|kv=" << ir.attention.n_kv_heads
          << "|hd=" << ir.attention.head_dim << "|qkn=" << static_cast<int>(ir.attention.qk_norm)
          << "|sw=" << ir.attention.sliding_window << "|rope=" << ir.attention.rope.theta << ':'
          << static_cast<int>(ir.attention.rope.scaling.kind) << ':'
          << ir.attention.rope.scaling.factor << "|E=" << ir.router.n_experts
          << "|k=" << ir.router.top_k << "|score=" << to_string(ir.router.score_fn)
          << "|norm=" << ir.router.normalize_topk << "|rsf=" << ir.router.routed_scaling_factor
          << "|bias=" << ir.router.bias_correction << "|g=" << ir.router.n_groups << ':'
          << ir.router.topk_group << "|shared=" << ir.router.n_shared_experts
          << "|act=" << to_string(ir.ffn.activation) << "|fi=" << ir.ffn.expert_intermediate << ':'
          << ir.ffn.dense_intermediate << ':' << ir.ffn.shared_intermediate;

    // DSA, emitted ONLY when present.
    //
    // It belongs in the hash: which layers own an indexer and how many keys
    // survive selection are facts about what the model IS, and a KV checkpoint
    // written under one selection must not replay under another.
    //
    // Conditional, though, so that adding this field does not change the hash of
    // any model that has no indexer. An unconditional `|dsa=` would invalidate
    // every existing container's arch_hash and every KV checkpoint keyed on it —
    // a migration to describe an architecture none of them use. The five families
    // already admitted hash exactly as before; verified rather than assumed.
    if (ir.attention.family == AttentionFamily::MlaDsa) {
        canon << "|dsa=" << ir.attention.dsa.index_topk << ':' << ir.attention.dsa.n_index_heads
              << ':' << ir.attention.dsa.index_head_dim << ':';
        for (const auto k : ir.attention.dsa.layer_kinds) {
            canon << (k == IndexerKind::Full ? 'f' : k == IndexerKind::Shared ? 's' : '-');
        }
    }
    // KDA, on the same conditional terms as DSA above and for the same reason:
    // which layers carry a cache versus a recurrent state is what the model IS,
    // and a checkpoint written under one split must never replay under another.
    // Emitted only for this family, so no existing container's hash moves.
    if (ir.attention.family == AttentionFamily::MlaKda) {
        const auto& k = ir.attention.kda;
        canon << "|kda=" << k.n_heads << ':' << k.head_dim << ':' << k.conv_kernel << ':'
              << k.has_gate_bound << ':' << k.gate_lower_bound << ':' << k.full_rank_gate << ':'
              << ir.attention.mla.nope << ':' << ir.attention.mla.output_gate << ':';
        for (const auto kind : k.layer_kinds) canon << (kind == AttnLayerKind::Linear ? 'l' : 'f');
        canon << "|blkres=" << ir.block_residual.block_size;
    }
    // GDN, on the same conditional terms and for the same reason. Three of these
    // are load-bearing in ways the shared `|h=|kv=|hd=` prefix above cannot
    // carry:
    //
    //   * the key/value head SPLIT, because it sets the recurrent state size and
    //     two configs agreeing on everything else but this are two models;
    //   * the rotary WIDTH, which the general `|rope=` term omits — it hashes
    //     theta and the scaling kind only, so a checkpoint rotating 64 of 256
    //     channels and one rotating all 256 would otherwise hash identically;
    //   * the fused output gate, which doubles `q_proj` and changes what the
    //     block computes.
    if (ir.attention.family == AttentionFamily::GqaGdn) {
        const auto& g = ir.attention.gdn;
        canon << "|gdn=" << g.n_k_heads << ':' << g.n_v_heads << ':' << g.head_k_dim << ':'
              << g.head_v_dim << ':' << g.conv_kernel << ':' << ir.attention.rope.partial_dim
              << ':' << ir.attention.fused_output_gate << ':' << ir.ffn.shared_expert_gate << ':';
        for (const auto kind : g.layer_kinds) canon << (kind == AttnLayerKind::Linear ? 'l' : 'f');
    }
    // BSA, on the same conditional terms as DSA, KDA and GDN above: emitted only
    // for this family, so no existing container's hash moves.
    //
    // Every term is load-bearing and none is carried by the shared `|h=|kv=|hd=`
    // prefix:
    //
    //   * the four indexer dimensions, because two configs agreeing on
    //     everything else and disagreeing on `block_size` select different keys
    //     and are different models;
    //   * `local_blocks`, which decides whether a query can see its own block at
    //     all when the indexer scores it badly;
    //   * the rotary WIDTH, which the general `|rope=` term omits -- it hashes
    //     theta and the scaling kind only, so rotating 64 of 128 channels and
    //     rotating all 128 would otherwise hash identically;
    //   * the per-layer indexer map, for the reason DSA's is hashed: a KV
    //     checkpoint written under one split must never replay under another.
    if (ir.attention.family == AttentionFamily::GqaBsa) {
        const auto& b = ir.attention.bsa;
        canon << "|bsa=" << b.n_index_heads << ':' << b.index_head_dim << ':' << b.block_size
              << ':' << b.topk_blocks << ':' << b.local_blocks << ':'
              << ir.attention.rope.partial_dim << ':';
        for (const auto k : b.layer_kinds) canon << (k == IndexerKind::Full ? 'f' : '-');
    }
    // The clamped-SwiGLU parameters, on the same conditional terms as `situ`
    // below. `swiglu_limit` is hashed unconditionally for CompressedSparse
    // already; this covers the alpha, which nothing else does -- and two
    // checkpoints differing only in it compute a different gate on every token.
    if (ir.ffn.activation == Activation::SwiGluOai) {
        canon << "|oai=" << ir.ffn.swiglu_alpha << ':' << ir.ffn.swiglu_limit;
    }
    // The latent width, and the activation's parameters, are model identity: two
    // checkpoints differing only in `situ_beta` are two different models, and
    // the expert width decides what a converted expert's bytes even mean.
    //
    // Both conditional, so a family that has neither hashes exactly as before.
    if (ir.ffn.routed_expert_hidden != 0) {
        canon << "|latent=" << ir.ffn.routed_expert_hidden << ':' << ir.ffn.routed_expert_norm;
    }
    // Conditional, like everything else added after the fact: a model using the
    // plain convention hashes exactly as it did before this field existed.
    if (ir.rms_norm_scale != RmsNormScale::Weight) {
        canon << "|normscale=" << static_cast<int>(ir.rms_norm_scale);
    }
    if (ir.ffn.activation == Activation::Situ) {
        canon << "|situ=" << ir.ffn.situ_beta << ':' << ir.ffn.situ_linear_beta;
    }
    if (ir.modality.modality != Modality::Text) {
        canon << "|mm=" << to_string(ir.modality.modality) << ':'
              << ir.modality.media_placeholder_token_id << ':' << ir.modality.vision_layers << ':'
              << ir.modality.vision_hidden << ':' << ir.modality.vision_patch_size;
    }
    if (ir.attention.family == AttentionFamily::CompressedSparse) {
        const auto& c = ir.attention.compressed;
        canon << "|csa=" << c.q_lora_rank << ':' << c.rope_head_dim << ':' << c.o_groups << ':'
              << c.o_lora_rank << ':' << c.index_n_heads << ':' << c.index_head_dim << ':'
              << c.index_topk << ':' << c.compress_rope_theta << ':'
              << c.semantic_fp8_quant_dequant << ':' << c.semantic_fp4_quant_dequant << ':';
        for (const auto ratio : c.compress_ratios) canon << ratio << ',';
        canon << "|hc=" << ir.hyper_connections.multiplier << ':'
              << ir.hyper_connections.sinkhorn_iters << ':' << ir.hyper_connections.eps
              << "|hash=" << ir.router.n_hash_layers << "|limit=" << ir.ffn.swiglu_limit
              << "|maxctx=" << ir.topology.max_position_embeddings;
        canon << "|eos=";
        for (const auto token : ir.topology.eos_token_ids) canon << token << ',';
    }

    // Conditional for backwards compatibility: merely adding DSpark support to
    // Soma must not change the identity of existing V4 containers which record
    // `dspark: omitted`. A capable container necessarily has a different hash.
    if (ir.speculative.present) {
        const auto& d = ir.speculative;
        canon << "|dspark=" << d.n_layers << ':' << d.trained_block_size << ':'
              << d.noise_token_id << ':' << d.markov_rank << ':' << d.confidence_head << ':';
        for (const auto layer : d.target_layer_ids) canon << layer << ',';
        canon << ':' << d.routed_bytes << ':' << d.resident_bytes << ':' << d.expert_bytes
              << ':' << d.kv_bytes_per_sequence;
    }

    // The WHOLE quant map, every role, dtype AND group.
    //
    // It used to be four roles and dtype only. Two consequences, both of which
    // are the failure this hash exists to prevent:
    //
    //   * q4_g at group 128 and q4_g at group 64 hashed IDENTICALLY. They
    //     dequantize to different weights, so a KV checkpoint written under one
    //     and replayed under the other resumes a conversation the cache does not
    //     describe — fluent, wrong, and nothing detects it.
    //   * expert_up, shared_expert, norms and draft_head were not covered at
    //     all. Requantizing the shared expert produced the same hash as not
    //     requantizing it.
    //
    // Adding a role here CHANGES every hash, which is correct and is what
    // "requantization invalidates" means. It is also why the loop is over a
    // named list rather than an ad-hoc sequence of fields: a role added to
    // QuantMap and forgotten here is silent, and its symptom is a resumed
    // conversation that reads as the model being inconsistent.
    static constexpr TensorRole kRoles[] = {
        TensorRole::Embed,
        TensorRole::AttnProj,
        TensorRole::ExpertGate,
        TensorRole::ExpertUp,
        TensorRole::ExpertDown,
        TensorRole::SharedExpert,
        TensorRole::Router,
        TensorRole::DraftHead,
        TensorRole::Norms,
    };
    canon << "|q=";
    for (const auto role : kRoles) {
        const auto& spec = ir.quantization.for_role(role);
        canon << to_string(spec.dtype) << '@' << spec.group << ',';
    }

    const std::string blob = canon.str();
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(blob.data()), blob.size(), digest.data());

    static constexpr char kHex[] = "0123456789abcdef";
    out_hash.clear();
    out_hash.reserve(digest.size() * 2);
    for (const auto byte : digest) {
        out_hash.push_back(kHex[byte >> 4]);
        out_hash.push_back(kHex[byte & 0x0F]);
    }
    return {};
}

} // namespace soma
