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

    std::uint32_t freq = 1;
    if (const auto it = j.find("moe_layer_freq"); it != j.end() && it->is_number_integer()) {
        freq = std::max<std::uint32_t>(1, it->get<std::uint32_t>());
    }

    for (std::uint32_t i = 0; i < n_layers; ++i) {
        if (kinds[i] == LayerKind::Dense) continue;
        const bool only_mlp = std::find(mlp_only.begin(), mlp_only.end(), i) != mlp_only.end();
        if (only_mlp || (step > 1 && i % step != 0) || (freq > 1 && i % freq != 0)) {
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
    }
    return "unknown";
}

const char* to_string(ScoreFn score_fn) noexcept {
    switch (score_fn) {
    case ScoreFn::Softmax:
        return "softmax";
    case ScoreFn::Sigmoid:
        return "sigmoid";
    }
    return "unknown";
}

Status adapt_hf_config(std::string_view text, ArchIr& out) {
    json j;
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
                    "'; supported: olmoe, qwen3_moe, qwen2_moe, mixtral, deepseek_v2, "
                    "deepseek_v3, glm_moe_dsa"};
    }

    out = ArchIr{};
    out.source_model_type = model_type;
    out.adapter = model_type;

    // ── topology ─────────────────────────────────────────────────────────────
    // Family default when the key is absent — NOT a global constant.
    out.rms_norm_eps = get_or<float>(
        j, "rms_norm_eps", get_or<float>(j, "layer_norm_eps", traits.default_rms_eps));
    out.naming = traits.naming;

    out.topology.n_layers = get_or<std::uint32_t>(j, "num_hidden_layers", 0);
    out.topology.d_model = get_or<std::uint32_t>(j, "hidden_size", 0);
    out.topology.vocab_size = get_or<std::uint32_t>(j, "vocab_size", 0);
    out.topology.first_k_dense = get_or<std::uint32_t>(j, "first_k_dense_replace", 0);
    out.topology.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", false);
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
    if (attn.family == AttentionFamily::Mla || attn.family == AttentionFamily::MlaDsa) {
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

    if (get_or<bool>(j, "use_sliding_window", false)) {
        attn.sliding_window = get_or<std::uint32_t>(j, "sliding_window", 0);
    }
    if (auto st = parse_rope(j, attn.rope); !st.ok()) return st;

    // ── router ───────────────────────────────────────────────────────────────
    auto& router = out.router;
    if (const json* n = find_first(j, kExpertCountKeys)) {
        router.n_experts = n->get<std::uint32_t>();
    }
    router.top_k = get_or<std::uint32_t>(j, "num_experts_per_tok", 0);
    router.score_fn = (get_or<std::string>(j, "scoring_func", "softmax") == "sigmoid")
                          ? ScoreFn::Sigmoid
                          : ScoreFn::Softmax;
    router.normalize_topk = traits.force_normalize_topk || get_or<bool>(j, "norm_topk_prob", false);
    router.routed_scaling_factor = get_or<float>(j, "routed_scaling_factor", 1.0f);
    router.bias_correction = (get_or<std::string>(j, "topk_method", "greedy") == "noaux_tc");
    router.n_groups = std::max<std::uint32_t>(1, get_or<std::uint32_t>(j, "n_group", 1));
    router.topk_group = std::max<std::uint32_t>(1, get_or<std::uint32_t>(j, "topk_group", 1));
    router.n_shared_experts = get_or<std::uint32_t>(j, "n_shared_experts", 0);

    // ── ffn ──────────────────────────────────────────────────────────────────
    auto& ffn = out.ffn;
    const auto act = get_or<std::string>(j, "hidden_act", "silu");
    if (act == "silu" || act == "swish") {
        ffn.activation = Activation::SwiGlu;
    } else if (act == "gelu" || act == "gelu_new" || act == "gelu_pytorch_tanh") {
        ffn.activation = Activation::GeGlu;
    } else if (act == "relu2") {
        ffn.activation = Activation::Relu2;
    } else {
        return {StatusCode::Unsupported, "unsupported hidden_act '" + act + "'"};
    }
    ffn.has_gate = true;
    ffn.dense_intermediate = get_or<std::uint32_t>(j, "intermediate_size", 0);
    if (const json* m = find_first(j, kMoeIntermediateKeys)) {
        ffn.expert_intermediate = m->get<std::uint32_t>();
    } else {
        // Mixtral and OLMoE have no moe_intermediate_size: intermediate_size IS
        // the expert width.
        ffn.expert_intermediate = ffn.dense_intermediate;
    }
    // Two families, two ways of saying the same thing.
    //
    // Qwen2-MoE STATES the shared expert's width in `shared_expert_intermediate_size`.
    // DeepSeek DERIVES it as `moe_intermediate_size * n_shared_experts` and ships
    // no such key. Reading only the stated form gave DeepSeek a shared width of
    // zero — the tensors bind, the width is wrong, and the logits come out
    // finite and off by ~0.1 mean, which reads as a subtle attention bug rather
    // than a config one.
    ffn.shared_intermediate = get_or<std::uint32_t>(j, "shared_expert_intermediate_size", 0);
    if (ffn.shared_intermediate == 0 && router.n_shared_experts > 0) {
        ffn.shared_intermediate = ffn.expert_intermediate * router.n_shared_experts;
    }
    ffn.expert_layout = ExpertLayout::InterleavedGateUpDown;

    // G0/G1 read fp32 HF checkpoints directly; requantization happens at
    // admission, not here.
    out.quantization = QuantMap{};

    return validate_arch_ir(out);
}

Status validate_arch_ir(const ArchIr& ir) {
    if (ir.schema_version != kArchIrSchemaVersion) {
        return {StatusCode::VersionMismatch,
                "arch IR schema " + std::to_string(ir.schema_version) +
                    " != " + std::to_string(kArchIrSchemaVersion)};
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
