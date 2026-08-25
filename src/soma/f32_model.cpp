#include "soma/f32_model.hpp"

#include "soma/attention_backend.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/threading.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace soma {

void KvTransaction::begin(std::uint32_t max_rows) {
    entries_.clear();
    saved_.clear();
    max_rows_ = max_rows;
    active_ = true;
}

void KvTransaction::capture(std::uint32_t row, std::byte* destination, std::size_t bytes) {
    if (!active_ || destination == nullptr || bytes == 0 || row >= max_rows_) return;
    const auto offset = saved_.size();
    saved_.insert(saved_.end(), destination, destination + bytes);
    entries_.push_back({row, destination, offset, bytes});
}

void KvTransaction::rollback_from(std::uint32_t first_row) noexcept {
    if (!active_) return;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (it->row < first_row) continue;
        std::copy_n(saved_.data() + it->offset, it->bytes, it->destination);
    }
}

void KvTransaction::clear() noexcept {
    entries_.clear();
    saved_.clear();
    max_rows_ = 0;
    active_ = false;
}

namespace {

Status read_text(const fs::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "cannot open " + p.string()};
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return {};
}

Status bind_tensor(const SafeTensors& st, const std::string& name, std::span<const float>& out) {
    const TensorView* tv = nullptr;
    if (auto s = st.require(name, tv); !s.ok()) return s;
    if (tv->dtype != DType::F32) {
        return {StatusCode::Unsupported,
                name + " is " + to_string(tv->dtype) + "; the G0 path is fp32 only"};
    }
    out = tv->f32();
    return {};
}

Status
bind_tensor_optional(const SafeTensors& st, const std::string& name, std::span<const float>& out) {
    if (const TensorView* tv = st.find(name); tv != nullptr) {
        if (tv->dtype != DType::F32) {
            return {StatusCode::Unsupported, name + " is not fp32"};
        }
        out = tv->f32();
    }
    return {};
}

Status bind_weight_view(const TensorView& tv,
                        const std::string& name,
                        const QuantSpec& spec,
                        std::vector<QTensor>& owner,
                        WeightRef& out) {
    if (tv.rank() != 2) {
        return {StatusCode::InvalidArgument,
                name + " has rank " + std::to_string(tv.rank()) + "; expected 2"};
    }
    const auto rows = static_cast<std::uint32_t>(tv.dim(0));
    const auto cols = static_cast<std::uint32_t>(tv.dim(1));
    if (is_quantized(tv.dtype)) {
        const auto want_group = effective_group(cols, spec.group ? spec.group : kDefaultGroup);
        if (tv.dtype != spec.dtype || tv.group != want_group) {
            return {StatusCode::InvalidArgument,
                    name + " sidecar is " + to_string(tv.dtype) + " group " +
                        std::to_string(tv.group) + "; load requested " + to_string(spec.dtype) +
                        " group " + std::to_string(want_group)};
        }
        out = WeightRef::from_quantized_bytes(tv.bytes, tv.dtype, tv.group, rows, cols);
        return {};
    }
    if (tv.dtype != DType::F32) {
        return {StatusCode::Unsupported, name + " is not fp32 or a Soma qweight"};
    }
    if (!is_quantized(spec.dtype)) {
        out = WeightRef::from_f32(tv.f32(), rows, cols);
        return {};
    }
    owner.emplace_back();
    if (auto s = quantize_tensor(tv.f32(),
                                 rows,
                                 cols,
                                 spec.dtype,
                                 spec.group ? spec.group : kDefaultGroup,
                                 owner.back());
        !s.ok()) {
        return {s.code(), name + ": " + s.message()};
    }
    out = WeightRef::from_q(owner.back());
    return {};
}

/// Bind a projection, quantizing it if its ROLE says so.
///
/// The QTensor is appended to the model's owning vector and the WeightRef points
/// into it, so `quantized` must be reserved before any of this runs — a
/// reallocation would dangle every ref taken so far, silently.
Status bind_weight(F32Model& model, const std::string& name, TensorRole role, WeightRef& out) {
    const TensorView* tv = nullptr;
    if (auto s = model.weights.require(name, tv); !s.ok()) return s;
    return bind_weight_view(*tv, name, model.quant_map.for_role(role), model.quantized, out);
}

Status
bind_weight_optional(F32Model& model, const std::string& name, TensorRole role, WeightRef& out) {
    if (model.weights.find(name) == nullptr) return {};
    return bind_weight(model, name, role, out);
}

std::string layer_prefix(std::uint32_t layer) {
    return "model.layers." + std::to_string(layer) + '.';
}

/// Bind one already-located [rows, cols] block of f32, quantizing per the role.
///
/// The tail of `bind_weight`, split out so the fused expert reader can reuse it
/// on a SLICE of a rank-3 tensor. Duplicating those six lines instead would mean
/// the two paths could drift on which spec they consult.
Status bind_block(F32Model& model,
                  std::span<const float> src,
                  std::uint32_t rows,
                  std::uint32_t cols,
                  TensorRole role,
                  WeightRef& out) {
    const QuantSpec& spec = model.quant_map.for_role(role);
    if (!is_quantized(spec.dtype)) {
        out = WeightRef::from_f32(src, rows, cols);
        return {};
    }
    model.quantized.emplace_back();
    if (auto s = quantize_tensor(src,
                                 rows,
                                 cols,
                                 spec.dtype,
                                 spec.group ? spec.group : kDefaultGroup,
                                 model.quantized.back());
        !s.ok()) {
        return s;
    }
    out = WeightRef::from_q(model.quantized.back());
    return {};
}

/// The FUSED layout for a SINGLE, always-active FFN -- a dense layer's MLP or a
/// shared expert.
///
/// `gate_up_proj` is `[2 * inter, d_model]` and the split is contiguous ROWS,
/// gate first: upstream builds it as `Linear(hidden, 2 * inter)` and applies it
/// as `gate_up.chunk(2, dim=-1)`, so the two halves of the OUTPUT are the two
/// halves of the weight's rows.
///
/// The rank-3 expert reader is the same idea one dimension up, and this is
/// deliberately NOT folded into it: that one indexes experts and this one does
/// not, and the shape check is what catches a checkpoint whose config lies about
/// its width. Sharing them would mean a rank check that accepts either, which is
/// how a `[E, 2*inter, d]` tensor gets read as a `[2*inter, d]` one.
///
/// Returns NotFound rather than an error when the tensor is absent, because the
/// per-projection layout is equally valid and the caller tries that next.
Status bind_fused_glu(F32Model& model,
                      const std::string& prefix,
                      std::uint32_t d_model,
                      std::uint32_t inter,
                      TensorRole role,
                      WeightRef& gate,
                      WeightRef& up,
                      WeightRef& down) {
    const auto* gu = model.weights.find(prefix + "gate_up_proj.weight");
    const auto* dn = model.weights.find(prefix + "down_proj.weight");
    if (gu == nullptr || dn == nullptr) return {StatusCode::NotFound, "not the fused layout"};

    const auto where = prefix + "gate_up_proj.weight: ";
    if (gu->rank() != 2 || dn->rank() != 2) {
        return {StatusCode::InvalidArgument, where + "expected rank-2 gate_up_proj and down_proj"};
    }
    if (gu->dtype != DType::F32 || dn->dtype != DType::F32) {
        return {StatusCode::Unsupported, where + "not fp32 in the checkpoint"};
    }
    if (gu->dim(0) != 2 * static_cast<std::int64_t>(inter) || gu->dim(1) != d_model ||
        dn->dim(0) != d_model || dn->dim(1) != inter) {
        return {StatusCode::InvalidArgument,
                where + "shapes [" + std::to_string(gu->dim(0)) + "," + std::to_string(gu->dim(1)) +
                    "] / [" + std::to_string(dn->dim(0)) + "," + std::to_string(dn->dim(1)) +
                    "] disagree with the IR's " + std::to_string(inter) + " intermediate x " +
                    std::to_string(d_model) + " d_model"};
    }

    const auto src = gu->f32();
    const auto half = static_cast<std::size_t>(inter) * d_model;
    if (auto s = bind_block(model, src.subspan(0, half), inter, d_model, role, gate); !s.ok())
        return s;
    if (auto s = bind_block(model, src.subspan(half, half), inter, d_model, role, up); !s.ok())
        return s;
    return bind_block(model, dn->f32(), d_model, inter, role, down);
}

/// The FUSED expert layout: every expert stacked into one rank-3 tensor.
///
/// Shapes are checked against the IR rather than inferred from the tensor, so a
/// checkpoint whose intermediate size disagrees with its config fails here
/// instead of producing a container that loads and computes nonsense.
Status bind_fused_experts(F32Model& model,
                          const TensorView& gate_up,
                          const TensorView& down,
                          std::uint32_t layer,
                          F32LayerWeights& lw) {
    const auto& arch = model.arch;
    const auto E = arch.router.n_experts;
    const auto d = arch.topology.d_model;
    const auto inter = arch.ffn.expert_intermediate;
    const auto where = "layer " + std::to_string(layer) + " fused experts: ";

    if (gate_up.rank() != 3 || down.rank() != 3) {
        return {StatusCode::InvalidArgument, where + "expected rank-3 gate_up_proj and down_proj"};
    }
    if (gate_up.dtype != DType::F32 || down.dtype != DType::F32) {
        return {StatusCode::Unsupported, where + "not fp32 in the checkpoint"};
    }
    const auto want_gu = std::array<std::int64_t, 3>{E, 2 * static_cast<std::int64_t>(inter), d};
    const auto want_dn = std::array<std::int64_t, 3>{E, d, inter};
    for (std::size_t i = 0; i < 3; ++i) {
        if (gate_up.dim(i) != want_gu[i] || down.dim(i) != want_dn[i]) {
            return {StatusCode::InvalidArgument,
                    where + "shapes [" + std::to_string(gate_up.dim(0)) + "," +
                        std::to_string(gate_up.dim(1)) + "," + std::to_string(gate_up.dim(2)) +
                        "] / [" + std::to_string(down.dim(0)) + "," + std::to_string(down.dim(1)) +
                        "," + std::to_string(down.dim(2)) + "] disagree with the IR's " +
                        std::to_string(E) + " experts x " + std::to_string(inter) +
                        " intermediate x " + std::to_string(d) + " d_model"};
        }
    }

    const auto gu = gate_up.f32();
    const auto dn = down.f32();
    const auto gu_stride = static_cast<std::size_t>(2) * inter * d;
    const auto dn_stride = static_cast<std::size_t>(d) * inter;
    const auto half = static_cast<std::size_t>(inter) * d;

    for (std::uint32_t e = 0; e < E; ++e) {
        // gate = rows [0, inter), up = rows [inter, 2*inter). CONTIGUOUS, not
        // interleaved — see the caller.
        if (auto s = bind_block(model,
                                gu.subspan(e * gu_stride, half),
                                inter,
                                d,
                                TensorRole::ExpertGate,
                                lw.expert_gate[e]);
            !s.ok())
            return s;
        if (auto s = bind_block(model,
                                gu.subspan(e * gu_stride + half, half),
                                inter,
                                d,
                                TensorRole::ExpertUp,
                                lw.expert_up[e]);
            !s.ok())
            return s;
        if (auto s = bind_block(model,
                                dn.subspan(e * dn_stride, dn_stride),
                                d,
                                inter,
                                TensorRole::ExpertDown,
                                lw.expert_down[e]);
            !s.ok())
            return s;
    }
    return {};
}

} // namespace

// ── the loader half of the bind seam ─────────────────────────────────────────
//
// The loader knows how to read and quantize a NAMED tensor; the backend knows
// WHICH names exist. Each keeps its half, and neither needs the other's.

std::string LayerBindCtx::name(const char* suffix) const {
    return (prefix.empty() ? layer_prefix(layer) : prefix) + suffix;
}

Status bind_layer_f32(const LayerBindCtx& ctx,
                      const char* suffix,
                      std::span<const float>& out,
                      bool optional) {
    const auto n = ctx.name(suffix);
    if (optional && ctx.weights->find(n) == nullptr) return {};
    return bind_tensor(*ctx.weights, n, out);
}

Status bind_layer_weight(
    const LayerBindCtx& ctx, const char* suffix, TensorRole role, WeightRef& out, bool optional) {
    const auto n = ctx.name(suffix);
    const TensorView* tv = nullptr;
    if (optional && ctx.weights->find(n) == nullptr) return {};
    if (auto s = ctx.weights->require(n, tv); !s.ok()) return s;
    return bind_weight_view(*tv, n, ctx.quant->for_role(role), *ctx.owner, out);
}

Status bind_model_f32(const ModelBindCtx& ctx,
                      const char* name,
                      std::span<const float>& out,
                      bool optional) {
    if (optional && ctx.weights->find(name) == nullptr) return {};
    return bind_tensor(*ctx.weights, name, out);
}

Status bind_model_weight(
    const ModelBindCtx& ctx, const char* name, TensorRole role, WeightRef& out, bool optional) {
    const TensorView* tv = nullptr;
    if (optional && ctx.weights->find(name) == nullptr) return {};
    if (auto s = ctx.weights->require(name, tv); !s.ok()) return s;
    return bind_weight_view(*tv, std::string(name), ctx.quant->for_role(role), *ctx.owner, out);
}

void F32Workspace::reserve(const ArchIr& arch, std::uint32_t max_tokens) {
    const auto d = arch.topology.d_model;
    const auto hq = arch.attention.n_heads * arch.attention.head_dim;
    const auto hkv = arch.attention.n_kv_heads * arch.attention.head_dim;
    const std::size_t t = max_tokens;

    hidden.assign(t * d, 0.0f);
    residual.assign(t * d, 0.0f);
    normed.assign(t * d, 0.0f);
    attn_out.assign(t * d, 0.0f);
    attn_heads.assign(t * hq, 0.0f);

    q.assign(t * hq, 0.0f);
    k.assign(t * hkv, 0.0f);
    v.assign(t * hkv, 0.0f);
    // Allocated only for the families that fuse the gate into q_proj. Sized
    // unconditionally to zero otherwise, so `empty()` is a reliable test and no
    // other family pays for the buffer.
    if (arch.attention.fused_output_gate) {
        q_raw.assign(t * 2 * hq, 0.0f);
        attn_gate.assign(t * hq, 0.0f);
    } else {
        q_raw.clear();
        attn_gate.clear();
    }
    // One row per worker; ensure_score_scratch() grows this if the pool is
    // larger than one. Sized here too so the serial path needs no pool at all.
    scores.assign(t, 0.0f);

    const auto e = std::max<std::uint32_t>(1, arch.router.n_experts);
    const auto kk = std::max<std::uint32_t>(1, arch.router.top_k);
    router_logits.assign(t * e, 0.0f);
    expert_ids.assign(t * kk, 0u);
    expert_weights.assign(t * kk, 0.0f);

    const auto fi = std::max(arch.ffn.expert_intermediate,
                             std::max(arch.ffn.dense_intermediate, arch.ffn.shared_intermediate));
    gate_buf.assign(fi, 0.0f);
    up_buf.assign(fi, 0.0f);
    act_buf.assign(fi, 0.0f);
    ffn_out.assign(d, 0.0f);

    // One stride covers all four buffers, so it must fit the widest of them:
    // gate/up/act are `fi` wide and the down-projection output is `d`.
    ffn_stride = std::max(fi, d);
    ensure_ffn_scratch(1);

    union_counts.assign(e, 0u);
    union_experts.reserve(e);
    union_offsets.reserve(static_cast<std::size_t>(e) + 1);
    union_rows.assign(t * kk, 0u);
    union_weights.assign(t * kk, 0.0f);
    naive_expert_reads = 0;
    unique_expert_reads = 0;
}

Status load_f32_model(const std::string& dir, F32Model& out, const QuantMap& quant) {
    const fs::path root(dir);

    std::string cfg_text;
    if (auto s = read_text(root / "config.json", cfg_text); !s.ok()) return s;
    if (auto s = adapt_hf_config(cfg_text, out.arch); !s.ok()) return s;

    // Container metadata is the atomic capability commit for optional model
    // payloads. Config.json only says the upstream source declared DSpark; this
    // overlay proves the three draft stages are actually present and supplies
    // their exact byte accounting.
    if (fs::exists(root / "container_meta.json")) {
        std::string meta_text;
        if (auto s = read_text(root / "container_meta.json", meta_text); !s.ok()) return s;
        if (auto s = apply_container_quant(meta_text, out.arch); !s.ok()) return s;
    }

    // The IR for MLA is complete and validated — MlaSpec was co-designed with
    // GQA during the design pass and needed no change to admit DeepSeek. What is
    // NOT ready is the weight binding below, which names q_proj/k_proj/v_proj:
    // MLA has kv_a_proj_with_mqa, kv_a_layernorm and kv_b_proj instead.
    //
    // Refused here rather than allowed to fail on a missing tensor, so the
    // conformance table reads "MLA backend not implemented" instead of
    // "model.layers.0.self_attn.k_proj.weight not in checkpoint" — which names a
    // symptom and sends the reader looking for a broken checkpoint.
    // Asked of the BACKEND REGISTRY, not of the family.
    //
    // The first draft of this check named the family directly and the seam check
    // rejected it — correctly. "Which families can this loader bind weights for?"
    // is exactly the knowledge the core is not allowed to hold: adding a third
    // architecture would mean editing this line, which is the coupling the seam
    // exists to prevent. Whether a backend is registered answers the same
    // question and stays true as backends are added.
    const F32Backend* backend = resolve_f32_backend(out.arch);
    if (backend == nullptr || backend->bind_layer == nullptr) {
        return {StatusCode::Unsupported,
                std::string("no fp32 backend for attention family ") +
                    to_string(out.arch.attention.family)};
    }

    out.quant_map = quant;
    if (auto s = validate_quant_map(out.quant_map); !s.ok()) return s;

    // The IR describes what was actually loaded, not what the config said.
    //
    // Leaving arch.quantization at its all-f32 default while the weights are q4_g
    // means the IR lies about the model: the planner then predicts f32 footprints
    // and the container's expert-size check compares against the wrong number.
    // Both were observed before this line existed.
    out.arch.quantization = quant;
    if (auto s = validate_arch_ir(out.arch); !s.ok()) return s;

    // A container directory carries the dense half in dense.safetensors and the
    // routed experts in experts-*.bin. Detected here so expert binding can be
    // skipped consistently rather than failing halfway through the layer loop.
    out.experts_are_streamed = fs::exists(root / "soma.container");

    // Stamp the arch hash at load, not at admission — and AFTER the quant map,
    // which is the part that was wrong.
    //
    // Every format that gates on it — KV checkpoints, containers, the registry —
    // compares against `arch.arch_hash`, and until this was populated the
    // comparisons all still ran, comparing "" against "" and accepting
    // everything. Fixing that left a subtler version of the same fault: the hash
    // was computed three lines ABOVE the assignment below, so it covered the
    // all-f32 DEFAULT map rather than the map actually loaded. Every
    // quantization of one architecture therefore hashed identically at load,
    // which defeats the reason QuantMap is inside the hash at all — that the same
    // weights at two quantizations are two models, with two verdicts and two sets
    // of KV checkpoints. A checkpoint written under q4_g would replay under q8_0
    // with nothing detecting it, which is the exact failure the plan-side comment
    // in plan.cpp says this hash exists to prevent.
    //
    // Found by comparing `soma plan --quant-dense q4_g` against the same
    // container's own /internal/plan: `dense_resident_bytes` agreed to the byte
    // and the hashes did not (roadmap D42).
    if (auto s = compute_arch_hash(out.arch, out.arch.arch_hash); !s.ok()) return s;

    if (auto s = out.weights.open_dir(dir); !s.ok()) return s;

    const auto& arch = out.arch;

    // Reserve BEFORE any WeightRef is taken: every ref points into this vector
    // and a reallocation would dangle all of them at once.
    //
    // Counted rather than guessed, because the cost of being one short is not a
    // slow path, it is every previously-bound weight pointing at freed memory.
    // Worst case for ONE layer, quantized roles only (Router is F32 by rule and
    // never lands here):
    //
    //   attention   5  q_a + q_b (or q_proj) + kv_a + kv_b + o
    //   indexer     3  wq_b + wk + weights_proj      — DSA `full` layers
    //   routed    3*E  when the experts are not streamed
    //   shared      3  gate + up + down
    //   dense FFN   3  on `first_k_dense_replace` layers
    //
    // = 3*E + 14. The previous budget was 3*E + 10, which predates the indexer and
    // was already exactly at the limit for a shared-expert MoE layer.
    out.quantized.reserve(static_cast<std::size_t>(arch.topology.n_layers) *
                              (3 * static_cast<std::size_t>(arch.router.n_experts) + 14) +
                          2);
    if (auto s = bind_weight(out, "model.embed_tokens.weight", TensorRole::Embed, out.embed);
        !s.ok())
        return s;
    if (auto s = bind_tensor(out.weights, "model.norm.weight", out.out_norm); !s.ok()) return s;

    if (auto s = bind_weight_optional(out, "lm_head.weight", TensorRole::Embed, out.out_head);
        !s.ok())
        return s;
    if (out.out_head.empty()) {
        if (!arch.topology.tie_word_embeddings) {
            return {StatusCode::NotFound,
                    "lm_head.weight missing and tie_word_embeddings is false"};
        }
        // Tied embeddings: safetensors stores the shared storage once
        // (save_model drops the duplicate), so the head aliases the embedding.
        out.out_head = out.embed;
    }

    if (backend->bind_model != nullptr) {
        ModelBindCtx ctx{&out.weights, &out.quant_map, &out.quantized};
        if (const auto rc = backend->bind_model(arch, ctx, out.arch_payload);
            rc != StatusCode::Ok) {
            return {rc, "binding architecture model weights failed"};
        }
    }

    out.layers.resize(arch.topology.n_layers);
    for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l) {
        auto& lw = out.layers[l];
        lw.kind = arch.topology.layer_kinds[l];
        const auto p = layer_prefix(l);

        if (auto s = bind_tensor(out.weights, p + "input_layernorm.weight", lw.input_norm); !s.ok())
            return s;
        if (auto s =
                bind_tensor(out.weights, p + "post_attention_layernorm.weight", lw.post_attn_norm);
            !s.ok())
            return s;

        // Attention tensors are bound BY THE BACKEND, into a payload it owns.
        // The core does not know their names, their count, or their shapes — the
        // whole difference between GQA and MLA lives on the other side of this
        // one call.
        LayerBindCtx ctx;
        ctx.weights = &out.weights;
        ctx.quant = &out.quant_map;
        ctx.owner = &out.quantized;
        ctx.layer = l;
        if (const auto rc = backend->bind_layer(arch, ctx, lw.attn); rc != StatusCode::Ok) {
            return {rc, "binding attention weights failed at layer " + std::to_string(l)};
        }

        if (lw.kind == LayerKind::Moe) {
            const auto& nm = arch.naming;
            if (auto s = bind_tensor(out.weights, p + nm.moe_block + '.' + nm.router, lw.router);
                !s.ok())
                return s;

            // A container directory holds only the DENSE half; routed experts
            // live in experts-*.bin and are acquired through MemoryHierarchy.
            // Binding them here would fail, and skipping silently would leave the
            // resident table empty in a way only visible as wrong output — so the
            // presence of soma.container is what decides, explicitly.
            //
            // A GUARD, not a `continue`. It used to be `if
            // (out.experts_are_streamed) continue;`, which skipped the rest of
            // this branch — and the SHARED experts are bound below it. So every
            // container-served MoE model with shared experts silently dropped
            // their contribution: bound optionally, left empty, and the forward's
            // `if (!lw.shared_gate.empty())` then skipped them without a word.
            //
            // Precisely the failure the comment above warns about, one scope
            // further down. Measured as 5.7e-01 max|logit| between a
            // container-loaded and a source-loaded forward on DeepSeek-V2-Lite,
            // and it vanishes when n_shared_experts is set to 0 — which is what
            // identified it (roadmap D19). Affects DeepSeek, Moonlight, Qwen2-MoE
            // and GLM-5.2; every family whose shared expert fires on every token.
            if (!out.experts_are_streamed) {
                lw.expert_gate.resize(arch.router.n_experts);
                lw.expert_up.resize(arch.router.n_experts);
                lw.expert_down.resize(arch.router.n_experts);

                // Two layouts exist upstream and this reads both, exactly as
                // convert.py does (roadmap D4).
                //
                //   per-expert  experts.<i>.{gate,up,down}_proj.weight
                //   fused       experts.gate_up_proj [E, 2*inter, d_model]
                //               experts.down_proj    [E, d_model, inter]
                //
                // The converter learned the fused layout first, which left the
                // engine able to CONVERT GLM-5.2 and unable to load the source it
                // was converted from — so conformance, which runs against the fp32
                // source, could not run at all.
                //
                // The gate/up split is CONTIGUOUS rows, gate first, and that was
                // settled by measurement rather than by reading: reconstructing
                // from rows [0:inter] and [inter:] reproduces the reference's
                // `chunk(2, dim=-1)` at 0.0e+00 while the interleaved reading is
                // off by 1.35. gpt_oss stores the same concept interleaved, which
                // is why guessing was refused.
                const auto ep = p + nm.moe_block + ".experts.";
                const auto* fused_gu = out.weights.find(ep + "gate_up_proj");
                const auto* fused_dn = out.weights.find(ep + "down_proj");
                if (fused_gu != nullptr && fused_dn != nullptr) {
                    if (auto s = bind_fused_experts(out, *fused_gu, *fused_dn, l, lw); !s.ok())
                        return s;
                } else {
                    for (std::uint32_t e = 0; e < arch.router.n_experts; ++e) {
                        const auto pe = ep + std::to_string(e) + '.';
                        if (auto s = bind_weight(out,
                                                 pe + nm.expert_gate,
                                                 TensorRole::ExpertGate,
                                                 lw.expert_gate[e]);
                            !s.ok())
                            return s;
                        if (auto s = bind_weight(
                                out, pe + nm.expert_up, TensorRole::ExpertUp, lw.expert_up[e]);
                            !s.ok())
                            return s;
                        if (auto s = bind_weight(out,
                                                 pe + nm.expert_down,
                                                 TensorRole::ExpertDown,
                                                 lw.expert_down[e]);
                            !s.ok())
                            return s;
                    }
                }
            }
            if (arch.ffn.routed_expert_hidden != 0) {
                // Required, not optional. The IR says this layer routes into a
                // latent space; a checkpoint without the projections is a broken
                // conversion, and running full-width instead would feed every
                // expert the wrong shape.
                const auto& blk = arch.naming.moe_block;
                // SharedExpert, NOT AttnProj — the same role the dense-layer MLP
                // and the shared expert take, and for the same reason: these are
                // dense FFN-side tensors the converter keeps at F32.
                //
                // Measured, not assumed. Quantizing them with everything else at
                // q4_g takes this family's error amplification from 2.91 (with
                // the latent MoE removed, i.e. the same range as the other seven
                // fixtures) to 6.55. They sit in front of and behind EVERY
                // routed expert, so their error is common-mode across the top-k
                // rather than averaging out — which is exactly the error a
                // quantized MoE otherwise gets to cancel.
                //
                // The trade is also lopsided in production: at 7168 x 3584 over
                // 92 layers they are ~4.7 B parameters, so q4 saves ~2 GB of a
                // ~1.5 TB model — a tenth of a percent — for correlated error on
                // every token's entire routed contribution.
                if (auto s = bind_weight_optional(out,
                                                  p + blk + ".routed_expert_down_proj.weight",
                                                  TensorRole::SharedExpert,
                                                  lw.latent_down);
                    !s.ok())
                    return s;
                if (auto s = bind_weight_optional(out,
                                                  p + blk + ".routed_expert_up_proj.weight",
                                                  TensorRole::SharedExpert,
                                                  lw.latent_up);
                    !s.ok())
                    return s;
                if (lw.latent_down.empty() || lw.latent_up.empty()) {
                    return {StatusCode::NotFound,
                            "layer " + std::to_string(l) +
                                " declares a latent MoE but has no routed_expert_"
                                "down_proj/up_proj"};
                }
                if (arch.ffn.routed_expert_norm) {
                    if (auto s = bind_tensor(
                            out.weights, p + blk + ".routed_expert_norm.weight", lw.latent_norm);
                        !s.ok())
                        return s;
                }
            }
            if (arch.router.n_shared_experts > 0) {
                // The shared expert is fused on the same families whose routed
                // experts are, and for the same reason: it is a `DenseMLP` built
                // by the same constructor. Tried FIRST so that a checkpoint
                // carrying both layouts -- none does, but nothing prevents it --
                // resolves the same way the routed experts above do.
                const auto shared_fused = bind_fused_glu(out,
                                                         p + arch.naming.shared_block + '.',
                                                         arch.topology.d_model,
                                                         arch.ffn.shared_intermediate,
                                                         TensorRole::SharedExpert,
                                                         lw.shared_gate,
                                                         lw.shared_up,
                                                         lw.shared_down);
                if (!shared_fused.ok() && shared_fused.code() != StatusCode::NotFound) {
                    return shared_fused;
                }
                if (!shared_fused.ok()) {
                    if (auto s =
                            bind_weight_optional(out,
                                                 p + arch.naming.shared_block + ".gate_proj.weight",
                                                 TensorRole::SharedExpert,
                                                 lw.shared_gate);
                        !s.ok())
                        return s;
                    if (auto s =
                            bind_weight_optional(out,
                                                 p + arch.naming.shared_block + ".up_proj.weight",
                                                 TensorRole::SharedExpert,
                                                 lw.shared_up);
                        !s.ok())
                        return s;
                    if (auto s =
                            bind_weight_optional(out,
                                                 p + arch.naming.shared_block + ".down_proj.weight",
                                                 TensorRole::SharedExpert,
                                                 lw.shared_down);
                        !s.ok())
                        return s;
                }

                // The scalar branch gate. A SIBLING of the shared block, not a
                // member of it: upstream names it `mlp.shared_expert_gate` while
                // the expert itself is `mlp.shared_expert.*`, so it hangs off the
                // MoE block rather off the shared one.
                if (arch.ffn.shared_expert_gate) {
                    if (auto s = bind_weight_optional(out,
                                                      p + arch.naming.moe_block +
                                                          ".shared_expert_gate.weight",
                                                      TensorRole::SharedExpert,
                                                      lw.shared_scale);
                        !s.ok())
                        return s;
                }
            }
        } else {
            // SharedExpert, NOT the Expert* roles.
            //
            // A `first_k_dense_replace` layer's FFN is not a routed expert: the
            // converter writes it into dense.safetensors as F32 and never
            // quantizes it, exactly as it treats a shared expert. Binding it with
            // ExpertGate/Up/Down made the quant map apply to it, so loading the
            // SOURCE with a container's map quantized a tensor the CONTAINER
            // keeps at F32 — two paths, two precisions, same weights.
            //
            // Measured before the fix: a container-loaded forward diverged from a
            // source-loaded one by 5.6e-01 max|logit| on DeepSeek-V2-Lite and
            // 5.4e-01 on Moonlight, and by exactly 0.0 on the three fixtures with
            // no dense layer. That is the whole of roadmap D19, and nothing saw
            // it because no test had ever compared a container's OUTPUT to
            // anything (roadmap D19).
            //
            // The container is the authority here: it is what production serves.
            // These two roles are the "always-active FFN, resident, F32" family,
            // and the loader agreeing with the converter about that is the point.
            // Fused first, per-projection second -- see bind_fused_glu. A dense
            // layer's width is `ffn.dense_intermediate`, which is NOT
            // `expert_intermediate` on a family that states the two separately:
            // MiniMax-M3's dense layers are 12288 wide against its experts'
            // 3072, and binding one against the other's width fails the shape
            // check here rather than reading four times the rows it should.
            const auto dense_fused = bind_fused_glu(out,
                                                    p + arch.naming.dense_block + '.',
                                                    arch.topology.d_model,
                                                    arch.ffn.dense_intermediate,
                                                    TensorRole::SharedExpert,
                                                    lw.dense_gate,
                                                    lw.dense_up,
                                                    lw.dense_down);
            if (!dense_fused.ok() && dense_fused.code() != StatusCode::NotFound) {
                return dense_fused;
            }
            if (!dense_fused.ok()) {
                if (auto s = bind_weight(out,
                                         p + arch.naming.dense_block + ".gate_proj.weight",
                                         TensorRole::SharedExpert,
                                         lw.dense_gate);
                    !s.ok())
                    return s;
                if (auto s = bind_weight(out,
                                         p + arch.naming.dense_block + ".up_proj.weight",
                                         TensorRole::SharedExpert,
                                         lw.dense_up);
                    !s.ok())
                    return s;
                if (auto s = bind_weight(out,
                                         p + arch.naming.dense_block + ".down_proj.weight",
                                         TensorRole::SharedExpert,
                                         lw.dense_down);
                    !s.ok())
                    return s;
            }
        }
    }

    // Section extents for a packed container expert. Needed only by the
    // streaming path, but computed unconditionally so the two modes cannot
    // disagree about a layout.
    {
        const auto fi = arch.ffn.expert_intermediate;
        const auto d = arch.topology.d_model;
        if (fi > 0 && d > 0) {
            const auto sz = [&](std::uint32_t rows, std::uint32_t cols, TensorRole role) {
                const auto& spec = out.quant_map.for_role(role);
                return static_cast<std::uint32_t>(quantized_tensor_bytes(
                    spec.dtype, rows, cols, spec.group ? spec.group : kDefaultGroup));
            };
            out.expert_gate_bytes = sz(fi, d, TensorRole::ExpertGate);
            out.expert_up_bytes = sz(fi, d, TensorRole::ExpertUp);
            out.expert_down_bytes = sz(d, fi, TensorRole::ExpertDown);
        }
    }

    if (resolve_f32_backend(out.arch) == nullptr) {
        return {StatusCode::Unsupported,
                std::string("no fp32 backend for attention family ") +
                    to_string(out.arch.attention.family)};
    }
    return {};
}

namespace {

/// gate/up/down over one token. Activation comes from the IR, so this is
/// family-invariant and stays in core.
/// Rows applied against one expert per pass.
///
/// Sized so the gathered inputs and the three intermediates stay in L2 while the
/// weights stream through: at d_model 2048 and intermediate 768 that is
/// 8 x (2048 + 3x768 + 2048) x 4 B = ~200 KB. Larger tiles amortise the weight
/// traffic further but start evicting the weights they exist to reuse.
constexpr std::uint32_t kExpertTile = 8;

/// SOMA_PREFETCH_DEPTH: 0 disables prefetch, >0 overrides the cache-derived depth.
std::uint32_t prefetch_depth_override(std::uint32_t derived) noexcept {
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool have = (_dupenv_s(&buf, &len, "SOMA_PREFETCH_DEPTH") == 0 && buf != nullptr);
    const std::string v = have ? std::string(buf) : std::string();
    std::free(buf);
#else
    const char* raw = std::getenv("SOMA_PREFETCH_DEPTH");
    const std::string v = raw ? std::string(raw) : std::string();
#endif
    if (v.empty()) return derived;
    const long n = std::strtol(v.c_str(), nullptr, 10);
    return (n < 0) ? derived : static_cast<std::uint32_t>(n);
}

/// One expert applied to a TILE of rows.
///
/// The fix for the locality the batch union gave up. Expert-major order reads
/// each expert once from disk — the point of the union — and then, one row at a
/// time, re-read all ~2.9 MB of its weights from memory for every row that
/// selected it. Measured cost: ~9% of wall clock at 512 rows.
///
/// Gathering the tile's inputs into a contiguous buffer costs tile x d_model
/// floats (64 KB at tile 8) against 2.9 MB of weight traffic saved per extra
/// row, and it turns three scattered strided walks into three dense ones.
///
/// Bit-identical to the per-row path: every output is still one dot product
/// accumulated in one order. Only which operand stays in cache changes.
void apply_glu_expert_tile(const ArchIr& arch,
                           const WeightRef& gate_w,
                           const WeightRef& up_w,
                           const WeightRef& down_w,
                           const float* normed,
                           std::uint32_t d_model,
                           std::uint32_t inter,
                           const std::uint32_t* rows,
                           const float* row_weights,
                           std::uint32_t tile,
                           F32Workspace& ws,
                           float* out_base) noexcept {
    ws.ensure_tile_scratch(tile, d_model, inter);

    for (std::uint32_t t = 0; t < tile; ++t) {
        std::copy_n(normed + static_cast<std::size_t>(rows[t]) * d_model,
                    d_model,
                    ws.tile_x.data() + static_cast<std::size_t>(t) * d_model);
    }

    matmul_tiled(gate_w, ws.tile_x.data(), tile, ws.tile_gate.data());
    matmul_tiled(up_w, ws.tile_x.data(), tile, ws.tile_up.data());

    for (std::uint32_t t = 0; t < tile; ++t) {
        const auto off = static_cast<std::size_t>(t) * inter;
        const std::span<const float> g(ws.tile_gate.data() + off, inter);
        const std::span<const float> u(ws.tile_up.data() + off, inter);
        const std::span<float> a(ws.tile_act.data() + off, inter);
        if (arch.ffn.swiglu_limit > 0.0f) {
            const float limit = arch.ffn.swiglu_limit;
            for (std::uint32_t i = 0; i < inter; ++i) {
                ws.tile_gate[off + i] = std::min(ws.tile_gate[off + i], limit);
                ws.tile_up[off + i] = std::clamp(ws.tile_up[off + i], -limit, limit);
            }
        }
        switch (arch.ffn.activation) {
        case Activation::SwiGlu:
            f32::swiglu(g, u, inter, a);
            break;
        case Activation::GeGlu:
            f32::geglu(g, u, inter, a);
            break;
        case Activation::Relu2:
            f32::relu2_glu(g, u, inter, a);
            break;
        case Activation::Situ:
            f32::situ_glu(g, u, inter, arch.ffn.situ_beta, arch.ffn.situ_linear_beta, a);
            break;
        case Activation::SwiGluOai:
            f32::swiglu_oai(g, u, inter, arch.ffn.swiglu_alpha, a);
            break;
        }
    }

    matmul_tiled(down_w, ws.tile_act.data(), tile, ws.tile_out.data());

    for (std::uint32_t t = 0; t < tile; ++t) {
        const float* src = ws.tile_out.data() + static_cast<std::size_t>(t) * d_model;
        float* dst = out_base + static_cast<std::size_t>(rows[t]) * d_model;
        const float wgt = row_weights[t];
        for (std::uint32_t i = 0; i < d_model; ++i)
            dst[i] += wgt * src[i];
    }
}

void apply_glu_expert(const ArchIr& arch,
                      const WeightRef& gate_w,
                      const WeightRef& up_w,
                      const WeightRef& down_w,
                      const float* x,
                      std::uint32_t d_model,
                      std::uint32_t inter,
                      float weight,
                      FfnScratch s,
                      float* out) noexcept {
    // Scratch is passed in rather than taken from the workspace: the rows that
    // selected one expert are applied CONCURRENTLY, and four shared buffers would
    // be four races. This is the change that makes the MoE loop parallelisable at
    // the row level, where the useful granularity is.
    const std::span<const float> xs(x, d_model);
    matvec(gate_w, xs, s.gate);
    matvec(up_w, xs, s.up);

    if (arch.ffn.swiglu_limit > 0.0f) {
        const float limit = arch.ffn.swiglu_limit;
        for (std::uint32_t i = 0; i < inter; ++i) {
            s.gate[i] = std::min(s.gate[i], limit);
            s.up[i] = std::clamp(s.up[i], -limit, limit);
        }
    }

    switch (arch.ffn.activation) {
    case Activation::SwiGlu:
        f32::swiglu(s.gate, s.up, inter, s.act);
        break;
    case Activation::GeGlu:
        f32::geglu(s.gate, s.up, inter, s.act);
        break;
    case Activation::Relu2:
        f32::relu2_glu(s.gate, s.up, inter, s.act);
        break;
    case Activation::Situ:
        f32::situ_glu(s.gate, s.up, inter, arch.ffn.situ_beta, arch.ffn.situ_linear_beta, s.act);
        break;
    case Activation::SwiGluOai:
        f32::swiglu_oai(s.gate, s.up, inter, arch.ffn.swiglu_alpha, s.act);
        break;
    }

    matvec(down_w, s.act.subspan(0, inter), s.out);
    for (std::uint32_t i = 0; i < d_model; ++i)
        out[i] += weight * s.out[i];
}

} // namespace

Status KvCache::open(const ArchIr& arch, std::uint32_t max_ctx) {
    if (max_ctx == 0) return {StatusCode::InvalidArgument, "kv cache of zero context"};
    max_ctx_ = max_ctx;

    // The width comes FROM the backend. `n_kv_heads * head_dim` is GQA's shape,
    // and it was applied to every family — including one whose cache holds a
    // compressed latent and no per-head K or V at all.
    const auto* backend = resolve_f32_backend(arch);
    const auto* attention = resolve_attention_backend(arch.attention.family);
    if (attention != nullptr && attention->kv_bytes_for_context != nullptr) {
        const auto bytes = attention->kv_bytes_for_context(arch, max_ctx);
        try {
            opaque_.assign(static_cast<std::size_t>(bytes), std::byte{0});
        } catch (const std::bad_alloc&) {
            return {StatusCode::CapacityPressure,
                    "could not allocate " + std::to_string(bytes) + " bytes of opaque KV"};
        }
        k_.clear();
        v_.clear();
        max_ctx_ = max_ctx;
        k_hkv_ = 0;
        v_hkv_ = 0;
        length_ = 0;
        n_layers_ = arch.topology.n_layers;
        return {};
    }
    opaque_.clear();
    const auto geom = (backend != nullptr && backend->kv_geometry != nullptr)
                          ? backend->kv_geometry(arch)
                          : KvGeometry{arch.attention.n_kv_heads * arch.attention.head_dim,
                                       arch.attention.n_kv_heads * arch.attention.head_dim};
    k_hkv_ = geom.k_floats;
    v_hkv_ = geom.v_floats; // 0 is legitimate: the family stores no second plane
    if (k_hkv_ == 0) return {StatusCode::InvalidArgument, "kv cache of zero width"};
    const auto layers = static_cast<std::size_t>(arch.topology.n_layers);
    k_.assign(static_cast<std::size_t>(max_ctx_) * k_hkv_ * layers, 0.0f);
    v_.assign(static_cast<std::size_t>(max_ctx_) * v_hkv_ * layers, 0.0f);
    length_ = 0;
    n_layers_ = arch.topology.n_layers;
    return {};
}

Status KvCache::begin_tentative(std::uint32_t max_rows) {
    if (max_rows == 0) return {StatusCode::InvalidArgument, "zero-row KV transaction"};
    if (transaction_.active()) {
        return {StatusCode::InvalidArgument, "KV transaction is already active"};
    }
    if (max_rows > max_ctx_ - length_) {
        return {StatusCode::InvalidArgument, "KV transaction exceeds remaining context"};
    }
    try {
        transaction_.begin(max_rows);
    } catch (const std::bad_alloc&) {
        return {StatusCode::CapacityPressure, "could not allocate KV reverse journal"};
    }
    return {};
}

Status KvCache::commit_tentative_prefix(std::uint32_t accepted_rows) {
    if (!transaction_.active()) {
        return {StatusCode::InvalidArgument, "no active KV transaction"};
    }
    transaction_.rollback_from(accepted_rows);
    length_ += accepted_rows;
    transaction_.clear();
    return {};
}

void KvCache::abort_tentative() noexcept {
    if (!transaction_.active()) return;
    transaction_.rollback_from(0);
    transaction_.clear();
}

void F32Workspace::ensure_latent(std::uint32_t n_tokens, std::uint32_t width) {
    const auto need = static_cast<std::size_t>(n_tokens) * width;
    if (latent_in.size() < need) latent_in.assign(need, 0.0f);
    if (latent_out.size() < need) latent_out.assign(need, 0.0f);
}

void F32Workspace::ensure_tile_scratch(std::uint32_t tile,
                                       std::uint32_t d_model,
                                       std::uint32_t inter) {
    const auto xw = static_cast<std::size_t>(tile) * d_model;
    const auto iw = static_cast<std::size_t>(tile) * inter;
    if (tile_x.size() < xw) tile_x.assign(xw, 0.0f);
    if (tile_out.size() < xw) tile_out.assign(xw, 0.0f);
    if (tile_gate.size() < iw) tile_gate.assign(iw, 0.0f);
    if (tile_up.size() < iw) tile_up.assign(iw, 0.0f);
    if (tile_act.size() < iw) tile_act.assign(iw, 0.0f);
}

void F32Workspace::ensure_ffn_scratch(std::uint32_t n_workers) {
    const auto need = static_cast<std::size_t>(std::max(1u, n_workers)) * 4u * ffn_stride;
    if (ffn_scratch.size() < need) ffn_scratch.assign(need, 0.0f);
}

FfnScratch F32Workspace::worker_ffn(std::uint32_t worker) noexcept {
    float* base = ffn_scratch.data() + static_cast<std::size_t>(worker) * 4u * ffn_stride;
    return {std::span<float>(base, ffn_stride),
            std::span<float>(base + ffn_stride, ffn_stride),
            std::span<float>(base + 2u * ffn_stride, ffn_stride),
            std::span<float>(base + 3u * ffn_stride, ffn_stride)};
}

void F32Workspace::ensure_score_scratch(std::uint32_t n_workers, std::uint32_t n_tokens) {
    // Called on the serial side of every attention layer, before the parallel
    // region opens — never from inside one, where growing a shared vector would
    // invalidate pointers other workers are already holding.
    const std::size_t need =
        static_cast<std::size_t>(std::max(1u, n_workers)) * std::max(1u, n_tokens);
    if (scores.size() < need) scores.assign(need, 0.0f);
}

void build_expert_union(std::uint32_t n_rows,
                        std::uint32_t top_k,
                        std::uint32_t n_experts,
                        const std::uint32_t* ids,
                        const float* weights,
                        F32Workspace& ws) {
    const std::size_t n_sel = static_cast<std::size_t>(n_rows) * top_k;

    ws.union_counts.assign(n_experts, 0u);
    for (std::size_t i = 0; i < n_sel; ++i) {
        if (ids[i] < n_experts) ++ws.union_counts[ids[i]];
    }

    // Ascending expert id. Any stable order would do; an unstable one would make
    // the float accumulation order — and therefore the low bits of every output
    // row — depend on iteration order.
    ws.union_experts.clear();
    ws.union_offsets.clear();
    ws.union_offsets.push_back(0);
    for (std::uint32_t e = 0; e < n_experts; ++e) {
        if (ws.union_counts[e] == 0) continue;
        ws.union_experts.push_back(e);
        ws.union_offsets.push_back(ws.union_offsets.back() + ws.union_counts[e]);
    }

    // Reuse union_counts as the per-expert write cursor.
    std::uint32_t cursor = 0;
    for (const auto e : ws.union_experts) {
        ws.union_counts[e] = ws.union_offsets[cursor];
        ++cursor;
    }

    ws.union_rows.assign(n_sel, 0u);
    ws.union_weights.assign(n_sel, 0.0f);
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        for (std::uint32_t s = 0; s < top_k; ++s) {
            const auto i = static_cast<std::size_t>(r) * top_k + s;
            const auto e = ids[i];
            if (e >= n_experts) continue;
            const auto at = ws.union_counts[e]++;
            ws.union_rows[at] = r;
            ws.union_weights[at] = weights[i];
        }
    }
}

ExpertHandle acquire_expert(const F32Model& model, LayerIndex layer, ExpertId expert) {
    ExpertHandle h;

    if (model.streamed_experts == nullptr) {
        const auto& lw = model.layers[layer];
        if (expert < lw.expert_gate.size()) {
            h.gate = lw.expert_gate[expert];
            h.up = lw.expert_up[expert];
            h.down = lw.expert_down[expert];
        }
        return h;
    }

    // Streamed. The pin is taken FIRST and moved into the handle, so the bytes
    // cannot be evicted while the three views point into them.
    h.pin = model.streamed_experts->acquire(layer, expert);
    if (!h.pin) return h;

    const auto blob = h.pin.bytes();
    const auto need = static_cast<std::size_t>(model.expert_gate_bytes) + model.expert_up_bytes +
                      model.expert_down_bytes;
    if (blob.size() < need) {
        // The container's expert is smaller than the IR's layout implies. Refuse
        // rather than read past the section boundary into the next projection —
        // that would produce finite, wrong numbers.
        h.pin = MemoryHierarchy::ExpertRef{};
        return h;
    }

    const auto fi = model.arch.ffn.expert_intermediate;
    const auto d = model.arch.topology.d_model;
    const auto& qm = model.quant_map;
    const auto grp = [](const QuantSpec& s, std::uint32_t cols) {
        return effective_group(cols, s.group ? s.group : kDefaultGroup);
    };

    std::size_t off = 0;
    h.gate = WeightRef::from_quantized_bytes(blob.subspan(off, model.expert_gate_bytes),
                                             qm.expert_gate.dtype,
                                             grp(qm.expert_gate, d),
                                             fi,
                                             d);
    off += model.expert_gate_bytes;
    h.up = WeightRef::from_quantized_bytes(
        blob.subspan(off, model.expert_up_bytes), qm.expert_up.dtype, grp(qm.expert_up, d), fi, d);
    off += model.expert_up_bytes;
    h.down = WeightRef::from_quantized_bytes(blob.subspan(off, model.expert_down_bytes),
                                             qm.expert_down.dtype,
                                             grp(qm.expert_down, fi),
                                             d,
                                             fi);
    return h;
}

/// The single forward body, shared by the teacher-forced and batched-decode
/// entry points.
///
/// ONE implementation on purpose. The two paths differ only in where attention
/// gets its keys and values; everything else — embedding, norms, routing, the
/// expert union, the head — is identical. A second copy would drift, and the
/// drift would show up as the batched path disagreeing with the conformance
/// path, which is the one comparison that must stay trustworthy.
///
/// `rows` empty selects the teacher-forced path.
Status forward_impl(const F32Model& model,
                    std::span<const TokenId> tokens,
                    std::span<const KvRow> rows,
                    F32Workspace& ws,
                    std::vector<float>& out_logits,
                    HiddenStateTaps* taps) {
    const auto& arch = model.arch;
    const auto d = arch.topology.d_model;
    const auto vocab = arch.topology.vocab_size;
    const auto n = static_cast<std::uint32_t>(tokens.size());
    if (n == 0) return {StatusCode::InvalidArgument, "empty token sequence"};

    const F32Backend* backend = resolve_f32_backend(arch);
    if (backend == nullptr) {
        return {StatusCode::Unsupported, "no fp32 backend"};
    }
    if (model.experts_are_streamed && model.streamed_experts == nullptr) {
        // Loaded from a container but never given a hierarchy. Every expert
        // lookup would return an empty handle; failing here names the cause
        // instead of surfacing as "could not acquire expert 0 at layer 0".
        return {StatusCode::InvalidArgument,
                "model was loaded from a container but streamed_experts is null; "
                "routed experts have nowhere to come from"};
    }

    ws.reserve(arch, n);
    if (taps != nullptr) {
        taps->n_rows = n;
        taps->d_model = d;
        taps->values.assign(static_cast<std::size_t>(taps->layers.size()) * n * d, 0.0f);
    }

    // Embedding lookup.
    for (std::uint32_t t = 0; t < n; ++t) {
        const auto id = tokens[t];
        if (id >= vocab) {
            return {StatusCode::InvalidArgument,
                    "token id " + std::to_string(id) + " >= vocab " + std::to_string(vocab)};
        }
        const auto row = row_block(model.embed, id, 1);
        if (row.empty()) return {StatusCode::Internal, "embedding row view is empty"};
        if (auto s = dequantize(
                row, std::span<float>(ws.hidden).subspan(static_cast<std::size_t>(t) * d, d));
            !s.ok()) {
            return {s.code(), "embedding lookup: " + s.message()};
        }
    }

    // A selection computed for a previous prompt has the wrong length and the
    // wrong contents. Dropping it here means a backend that forgets to publish one
    // fails on the next forward instead of silently reusing the last.
    ws.reset_arch_state();
    if (backend->begin_forward != nullptr) {
        if (const auto rc = backend->begin_forward(
                arch, model.arch_payload, tokens.data(), n, ws, ws.hidden.data());
            rc != StatusCode::Ok) {
            return {rc, "architecture begin-forward hook failed"};
        }
    }

    for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l) {
        const auto& lw = model.layers[l];
        ws.current_layer = l;
        ws.sink(l, "hidden_in", ws.hidden.data(), static_cast<std::size_t>(n) * d);

        // ── attention block ──────────────────────────────────────────────────
        if (backend->pre_attention != nullptr) {
            if (const auto rc = backend->pre_attention(arch, lw, n, ws, ws.hidden.data());
                rc != StatusCode::Ok)
                return {rc, "pre-attention hook failed at layer " + std::to_string(l)};
        }
        for (std::uint32_t t = 0; t < n; ++t) {
            const auto off = static_cast<std::size_t>(t) * d;
            f32::rmsnorm_into(std::span<const float>(ws.hidden).subspan(off, d),
                              lw.input_norm,
                              d,
                              arch.rms_norm_eps,
                              std::span<float>(ws.normed).subspan(off, d),
                              arch.rms_norm_weight_offset());
        }
        const auto arc =
            rows.empty()
                ? backend->attention(arch, lw, ws.normed.data(), n, ws, ws.attn_out.data())
                : backend->attention_kv(
                      arch, lw, ws.normed.data(), n, l, rows.data(), ws, ws.attn_out.data());
        if (const auto rc = arc; rc != StatusCode::Ok) {
            return {rc, "attention failed at layer " + std::to_string(l)};
        }
        // Three taps per layer is the minimum that localises a defect without a
        // second run: `hidden_in` says whether the layer was handed good input,
        // `attn_out` separates attention from the FFN, and `hidden_out` says
        // whether the layer as a whole is where divergence begins.
        ws.sink(l, "attn_out", ws.attn_out.data(), static_cast<std::size_t>(n) * d);

        if (backend->merge_attention != nullptr) {
            if (const auto rc =
                    backend->merge_attention(arch, lw, ws.attn_out.data(), n, ws, ws.hidden.data());
                rc != StatusCode::Ok)
                return {rc, "attention merge hook failed at layer " + std::to_string(l)};
        } else {
            for (std::size_t i = 0; i < static_cast<std::size_t>(n) * d; ++i)
                ws.hidden[i] += ws.attn_out[i];
        }

        // ── feed-forward block ───────────────────────────────────────────────
        if (backend->pre_ffn != nullptr) {
            if (const auto rc = backend->pre_ffn(arch, lw, n, ws, ws.hidden.data());
                rc != StatusCode::Ok)
                return {rc, "pre-ffn hook failed at layer " + std::to_string(l)};
        }
        for (std::uint32_t t = 0; t < n; ++t) {
            const auto off = static_cast<std::size_t>(t) * d;
            f32::rmsnorm_into(std::span<const float>(ws.hidden).subspan(off, d),
                              lw.post_attn_norm,
                              d,
                              arch.rms_norm_eps,
                              std::span<float>(ws.normed).subspan(off, d),
                              arch.rms_norm_weight_offset());
        }

        if (lw.kind == LayerKind::Dense) {
            for (std::uint32_t t = 0; t < n; ++t) {
                const auto off = static_cast<std::size_t>(t) * d;
                std::fill_n(ws.attn_out.begin() + off, d, 0.0f);
                apply_glu_expert(arch,
                                 lw.dense_gate,
                                 lw.dense_up,
                                 lw.dense_down,
                                 ws.normed.data() + off,
                                 d,
                                 arch.ffn.dense_intermediate,
                                 1.0f,
                                 ws.worker_ffn(0),
                                 ws.attn_out.data() + off);
            }
        } else {
            const auto n_exp = arch.router.n_experts;
            const auto top_k = arch.router.top_k;

            f32::matmul(lw.router, ws.normed, n, n_exp, d, ws.router_logits);
            if (const auto rc = backend->route(arch,
                                               lw,
                                               tokens.data(),
                                               ws.router_logits.data(),
                                               n,
                                               ws.expert_ids.data(),
                                               ws.expert_weights.data());
                rc != StatusCode::Ok) {
                return {rc, "routing failed at layer " + std::to_string(l)};
            }
            ws.sink(
                l, "router_logits", ws.router_logits.data(), static_cast<std::size_t>(n) * n_exp);
            ws.sink(
                l, "router_weights", ws.expert_weights.data(), static_cast<std::size_t>(n) * top_k);
            if (ws.sink) {
                std::vector<float> route_ids(static_cast<std::size_t>(n) * top_k);
                std::vector<float> route_dense(static_cast<std::size_t>(n) * n_exp, 0.0f);
                std::transform(ws.expert_ids.begin(),
                               ws.expert_ids.begin() + route_ids.size(),
                               route_ids.begin(),
                               [](std::uint32_t id) { return static_cast<float>(id); });
                for (std::uint32_t t = 0; t < n; ++t)
                    for (std::uint32_t k = 0; k < top_k; ++k)
                        route_dense[static_cast<std::size_t>(t) * n_exp +
                                    ws.expert_ids[static_cast<std::size_t>(t) * top_k + k]] +=
                            ws.expert_weights[static_cast<std::size_t>(t) * top_k + k];
                ws.sink(l, "router_ids", route_ids.data(), route_ids.size());
                ws.sink(l, "router_dense", route_dense.data(), route_dense.size());
            }

            // ── the latent space the routed experts live in ──────────────────
            //
            // `routed_expert_width()` is d_model for every family without a
            // latent MoE, so everything below is the ordinary path when there is
            // none. When there is one, the residual is projected DOWN and the
            // experts see only the narrower vector.
            //
            // The ROUTER is deliberately upstream of this: it scored the
            // full-width input above. Routing on the projected vector would be a
            // different model that still selects plausible experts.
            const auto ew = arch.routed_expert_width();
            const bool latent = arch.ffn.routed_expert_hidden != 0;
            const float* expert_x = ws.normed.data();
            float* expert_out = ws.attn_out.data();
            if (latent) {
                ws.ensure_latent(n, ew);
                soma::matmul(
                    lw.latent_down,
                    std::span<const float>(ws.normed).first(static_cast<std::size_t>(n) * d),
                    n,
                    std::span<float>(ws.latent_in).first(static_cast<std::size_t>(n) * ew));
                std::fill_n(ws.latent_out.begin(), static_cast<std::size_t>(n) * ew, 0.0f);
                expert_x = ws.latent_in.data();
                expert_out = ws.latent_out.data();
            } else {
                for (std::uint32_t t = 0; t < n; ++t) {
                    std::fill_n(ws.attn_out.begin() + static_cast<std::size_t>(t) * d, d, 0.0f);
                }
            }

            // ── the batch-union ──────────────────────────────────────────────
            //
            // EXPERT-MAJOR, not row-major. The previous loop acquired an expert
            // once per (row, slot) — n * top_k acquires per layer, against at
            // most n_experts distinct ones. Grouping first means each expert is
            // read ONCE and applied to every row that selected it.
            //
            // The read cost is per-expert and independent of how many rows
            // consume it. That is the whole reason aggregate throughput in the
            // disk-bound regime scales better than linearly in batch size.
            build_expert_union(n, top_k, n_exp, ws.expert_ids.data(), ws.expert_weights.data(), ws);

            ws.naive_expert_reads += static_cast<std::uint64_t>(n) * top_k;
            ws.unique_expert_reads += ws.union_experts.size();

            ws.ensure_ffn_scratch(ThreadPool::global().size());

            // ── overlap the reads with the arithmetic ────────────────────────
            //
            // The union already names every expert this layer will touch, in the
            // order it will touch them. That makes this the one place in the
            // engine where prefetching requires no prediction at all: queue the
            // next few, and their reads happen on loader threads while the
            // current expert is being applied.
            //
            // Depth is bounded by the CACHE, not by taste. Queue more experts
            // than the per-layer cap and the prefetch evicts entries it has
            // already fetched but not yet used — turning a latency win into extra
            // I/O. Half the cap leaves room for the ones in use.
            // Prefetch was default-OFF for a while because it made the forward
            // non-deterministic: `unique_expert_reads` is a pure function of
            // routing and it varied run to run (2898/2902/2920 against a stable
            // 2918), which meant some expert's WEIGHTS were occasionally wrong.
            //
            // The cause was not in this loop or in the cache. ExpertStore::read
            // did `seekg` then `read` on a shared std::ifstream, and a file stream
            // has ONE position: two threads interleaving as seek(A), seek(B),
            // read(A) hand the first thread expert B's bytes, successfully and
            // silently. Positional reads (pread / ReadFile+OVERLAPPED) carry the
            // offset in the call, so concurrent reads cannot interfere. The
            // forward is bit-identical again, and tests/soma/streamed_determinism_g3
            // now guards it.
            std::uint32_t depth = 0;
            if (model.streamed_experts != nullptr) {
                const auto cap = model.streamed_experts->cap_per_layer();
                // Bounded by the CACHE: queue more than it holds and the prefetch
                // evicts entries it has fetched but not yet used, turning a
                // latency win into extra I/O.
                const auto derived = (cap == 0) ? 8u : std::max(1u, std::min(8u, cap / 2u));
                depth = prefetch_depth_override(derived);
                if (depth > 0) {
                    const auto n_pre = std::min<std::size_t>(depth, ws.union_experts.size());
                    model.streamed_experts->prefetch_ahead(
                        l, std::span<const ExpertId>(ws.union_experts.data(), n_pre));
                }
            }

            for (std::size_t u = 0; u < ws.union_experts.size(); ++u) {
                const auto e = ws.union_experts[u];

                // Queue the one `depth` ahead before touching this one, so the
                // loaders always have work outstanding while compute proceeds.
                if (model.streamed_experts != nullptr && depth > 0) {
                    const auto ahead = u + depth;
                    if (ahead < ws.union_experts.size()) {
                        model.streamed_experts->prefetch_ahead(
                            l, std::span<const ExpertId>(&ws.union_experts[ahead], 1));
                    }
                }

                // ONE acquire for every row that routed here. The handle holds
                // the cache borrow while all of them are applied, so the bytes
                // cannot be evicted mid-group.
                const auto exp = acquire_expert(model, l, e);
                if (!exp.valid()) {
                    return {StatusCode::CapacityPressure,
                            "could not acquire expert " + std::to_string(e) + " at layer " +
                                std::to_string(l)};
                }

                // ROW-TILED, not row-parallel.
                //
                // The previous version parallelised over this expert's rows and
                // applied the expert once per row. That is correct and it wastes
                // the union's whole point: the expert is read once from disk and
                // then re-read from memory once per row — ~2.9 MB of weight
                // traffic per token, and a measured ~9% of wall clock at 512
                // rows.
                //
                // Tiling inverts the loop so the weights stream once per tile
                // instead. The parallelism moves INSIDE matmul_tiled, over weight
                // rows, where outputs are still disjoint and the split stays
                // bit-identical.
                const auto first = ws.union_offsets[u];
                const auto last = ws.union_offsets[u + 1];
                for (auto i = first; i < last; i += kExpertTile) {
                    const auto tile = std::min<std::uint32_t>(kExpertTile, last - i);
                    apply_glu_expert_tile(arch,
                                          exp.gate,
                                          exp.up,
                                          exp.down,
                                          expert_x,
                                          ew,
                                          arch.ffn.expert_intermediate,
                                          &ws.union_rows[i],
                                          &ws.union_weights[i],
                                          tile,
                                          ws,
                                          expert_out);
                }
            }

            if (latent) {
                // Norm the COMBINED top-k output, then project back up.
                //
                // Order matters twice over. The norm is after the weighted sum,
                // so it cannot be folded into the experts; and the up-projection
                // OVERWRITES `attn_out` rather than accumulating, because the
                // routed half is the whole of it until the shared expert adds
                // itself below.
                if (!lw.latent_norm.empty()) {
                    for (std::uint32_t t = 0; t < n; ++t) {
                        f32::rmsnorm(std::span<float>(ws.latent_out)
                                         .subspan(static_cast<std::size_t>(t) * ew, ew),
                                     lw.latent_norm,
                                     ew,
                                     arch.rms_norm_eps);
                    }
                }
                soma::matmul(
                    lw.latent_up,
                    std::span<const float>(ws.latent_out).first(static_cast<std::size_t>(n) * ew),
                    n,
                    std::span<float>(ws.attn_out).first(static_cast<std::size_t>(n) * d));
            }

            if (!lw.shared_gate.empty()) {
                // The shared expert reads the FULL-WIDTH input, not the latent
                // one, and adds into the up-projected result. It is outside the
                // latent space entirely.
                // The branch gate, when this family has one, rides in as the
                // CONTRIBUTION WEIGHT rather than as a separate pass over the
                // output. `apply_glu_expert` already scales what it accumulates,
                // which is exactly `sigmoid(w · x) * shared(x)` — and doing it
                // this way means the gated and ungated families share one code
                // path instead of two that must agree.
                std::vector<float> scale;
                if (!lw.shared_scale.empty()) {
                    scale.assign(n, 0.0f);
                    soma::matmul(
                        lw.shared_scale,
                        std::span<const float>(ws.normed.data(), static_cast<std::size_t>(n) * d),
                        n,
                        std::span<float>(scale));
                    for (auto& s : scale)
                        s = 1.0f / (1.0f + std::exp(-s));
                }
                for (std::uint32_t t = 0; t < n; ++t) {
                    const auto off = static_cast<std::size_t>(t) * d;
                    apply_glu_expert(arch,
                                     lw.shared_gate,
                                     lw.shared_up,
                                     lw.shared_down,
                                     ws.normed.data() + off,
                                     d,
                                     arch.ffn.shared_intermediate,
                                     scale.empty() ? 1.0f : scale[t],
                                     ws.worker_ffn(0),
                                     ws.attn_out.data() + off);
                }
            }
        }

        if (backend->merge_ffn != nullptr) {
            if (const auto rc =
                    backend->merge_ffn(arch, lw, ws.attn_out.data(), n, ws, ws.hidden.data());
                rc != StatusCode::Ok)
                return {rc, "ffn merge hook failed at layer " + std::to_string(l)};
        } else {
            for (std::size_t i = 0; i < static_cast<std::size_t>(n) * d; ++i)
                ws.hidden[i] += ws.attn_out[i];
        }
        ws.sink(l, "hidden_out", ws.hidden.data(), static_cast<std::size_t>(n) * d);
        if (taps != nullptr) {
            for (std::size_t ordinal = 0; ordinal < taps->layers.size(); ++ordinal) {
                if (taps->layers[ordinal] != l) continue;
                float* dst = taps->values.data() + ordinal * static_cast<std::size_t>(n) * d;
                if (backend->export_layer_hidden != nullptr) {
                    if (const auto rc =
                            backend->export_layer_hidden(arch, l, n, ws, ws.hidden.data(), dst);
                        rc != StatusCode::Ok) {
                        return {rc, "layer hidden export failed at layer " + std::to_string(l)};
                    }
                } else {
                    std::copy_n(ws.hidden.data(), static_cast<std::size_t>(n) * d, dst);
                }
            }
        }
    }

    if (backend->end_forward != nullptr) {
        if (const auto rc = backend->end_forward(arch, model.arch_payload, n, ws, ws.hidden.data());
            rc != StatusCode::Ok)
            return {rc, "architecture end-forward hook failed"};
    }

    out_logits.assign(static_cast<std::size_t>(n) * vocab, 0.0f);
    for (std::uint32_t t = 0; t < n; ++t) {
        const auto off = static_cast<std::size_t>(t) * d;
        f32::rmsnorm_into(std::span<const float>(ws.hidden).subspan(off, d),
                          model.out_norm,
                          d,
                          arch.rms_norm_eps,
                          std::span<float>(ws.normed).subspan(off, d),
                          arch.rms_norm_weight_offset());
        matvec(model.out_head,
               std::span<const float>(ws.normed).subspan(off, d),
               std::span<float>(out_logits).subspan(static_cast<std::size_t>(t) * vocab, vocab));
    }
    return {};
}

Status forward_f32(const F32Model& model,
                   std::span<const TokenId> tokens,
                   F32Workspace& ws,
                   std::vector<float>& out_logits) {
    return forward_impl(model, tokens, {}, ws, out_logits, nullptr);
}

Status forward_step_f32(const F32Model& model,
                        std::span<const TokenId> tokens,
                        std::span<const KvRow> rows,
                        F32Workspace& ws,
                        std::vector<float>& out_logits,
                        HiddenStateTaps* taps) {
    if (rows.size() != tokens.size()) {
        return {StatusCode::InvalidArgument,
                "forward_step_f32: " + std::to_string(tokens.size()) + " tokens against " +
                    std::to_string(rows.size()) + " KV rows"};
    }
    if (rows.empty()) return {StatusCode::InvalidArgument, "empty batch"};
    return forward_impl(model, tokens, rows, ws, out_logits, taps);
}

Status generate_greedy_f32(const F32Model& model,
                           std::span<const TokenId> prefix,
                           std::uint32_t n_new,
                           F32Workspace& ws,
                           std::vector<TokenId>& out_tokens) {
    // Recomputes the whole prefix each step, matching the oracle's loop exactly.
    // A KV cache would be faster and would introduce a second code path whose
    // divergence from this one is precisely what G0 is not yet in a position to
    // detect. Caching arrives with the scheduler at G3.
    std::vector<TokenId> cur(prefix.begin(), prefix.end());
    std::vector<float> logits;
    out_tokens.clear();
    out_tokens.reserve(n_new);

    const auto vocab = model.arch.topology.vocab_size;
    for (std::uint32_t step = 0; step < n_new; ++step) {
        if (auto s = forward_f32(model, cur, ws, logits); !s.ok()) return s;
        const float* last = logits.data() + static_cast<std::size_t>(cur.size() - 1) * vocab;
        std::uint32_t best = 0;
        for (std::uint32_t i = 1; i < vocab; ++i) {
            if (last[i] > last[best]) best = i;
        }
        out_tokens.push_back(best);
        cur.push_back(best);
    }
    return {};
}

} // namespace soma
