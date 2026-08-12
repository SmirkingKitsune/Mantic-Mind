#pragma once

// Soma — the attention seam.
//
// Attention backends are CODE, not config. MLA and GQA differ in cache shape and
// decode algebra; weight absorption has no GQA analogue. Those are not values
// you can put in a struct.
//
// This interface is co-designed against BOTH families before either is
// implemented, because shaping it to one and discovering that at the second is
// the classic way this goes wrong. Every decision below is annotated with which
// family forced it.

#include "soma/arch_ir.hpp"
#include "soma/model.hpp"
#include "soma/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace soma {

class MemoryHierarchy;

/// Opaque tag identifying a KV checkpoint's on-disk layout. A checkpoint written
/// by one attention family must never be replayable by another — different cache
/// shape means silently wrong output rather than a clean failure.
///
/// Values are OWNED BY BACKENDS and derived from a format name, not enumerated
/// here. An enum listing every family would mean adding a third architecture
/// required editing a core header, which is exactly the coupling the seam
/// exists to prevent.
using KvFormatId = std::uint32_t;

inline constexpr KvFormatId kKvFormatInvalid = 0;

/// FNV-1a over the format name. Defined inline because it must be usable in a
/// constexpr context; it is a pure hash, not architecture logic.
constexpr KvFormatId kv_format_id(std::string_view name) noexcept {
    std::uint32_t hash = 2166136261u;
    for (const char c : name) {
        hash ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
        hash *= 16777619u;
    }
    return hash == kKvFormatInvalid ? 1u : hash;
}

/// A struct of function pointers, resolved once at load.
///
/// Hot calls devirtualize behind -DSOMA_ARCH=<name>. The generic pointer path
/// stays compiled and is what the conformance harness exercises, so a divergence
/// between the two is a test failure rather than a production surprise.
struct AttentionBackend {
    const char* name = nullptr;
    AttentionFamily family = AttentionFamily::Unknown;
    KvFormatId persist_format_id = kKvFormatInvalid;

    /// The ONLY cache property that crosses the seam.
    ///
    /// The core allocates opaque bytes (KvRegion) and hands the backend a
    /// region; it never learns what is inside. Worked numbers at fp16:
    ///   Qwen3-30B-A3B   gqa  2×4×128  = 1024 elem/tok/layer → 98 KB/tok
    ///   DeepSeek-V2-Lite mla 512+64   =  576 elem/tok/layer → 31 KB/tok
    ///
    /// GQA's KV is a first-class planner input, not a footnote: at 32k context
    /// Qwen3 wants 3.2 GB of the same RAM the expert cache wants, and a planner
    /// that sized the expert cache first would thrash on long contexts and look
    /// like an unrelated bug.
    std::size_t (*kv_bytes_per_token)(const ArchIr& arch) noexcept = nullptr;

    /// How many bytes one (rows x cols) tensor of `role` occupies under the
    /// model's quantization. Supplied BY the planner TO the backend below.
    ///
    /// Row-aware, because the effective group is the largest divisor of `cols`
    /// not exceeding the requested one — a flat element count disagrees for any
    /// tensor narrower than the group.
    using ByteSizer = std::uint64_t (*)(const ArchIr&,
                                        std::uint32_t rows,
                                        std::uint32_t cols,
                                        TensorRole role);

    /// Attention WEIGHT bytes for one layer — the resident cost, as opposed to
    /// the per-token cache cost above.
    ///
    /// Here for the same reason `kv_bytes_per_token` is: the shapes differ by
    /// family and the planner must not know how. It was a formula in plan.cpp,
    /// written against GQA (`q + 2*(n_kv_heads x head_dim) + o`) and applied to
    /// everything, which charged MLA for two per-head projections it does not
    /// have — 1.66x over on real containers, ~2.4x on GLM-5.2. The seam check
    /// refused the obvious fix of branching on the family in core, correctly:
    /// that is exactly the knowledge this pointer exists to hold.
    ///
    /// The backend owns the SHAPES and the caller owns the QUANTIZATION, which
    /// is why `sizer` is passed in rather than the backend assuming a dtype. A
    /// first version returned bytes and hardcoded fp32; it agreed with the old
    /// formula on the f32 fixtures and silently disagreed by ~8x on every
    /// quantized plan, which the verdict table caught.
    std::uint64_t (*weight_bytes_per_layer)(const ArchIr& arch, ByteSizer sizer) noexcept = nullptr;

    /// Intended to be called exactly once from load_model() when the production
    /// loader lands.
    ///
    /// Both current families implement this as a no-op. MLA absorption moves the
    /// up-projection to the query side per decode step; folding at load would keep
    /// a prohibitively large transposed fp32 copy resident. The hook remains the
    /// architecture-owned seam for any future load-time weight transformation.
    Status (*prepare_weights)(ModelState& model) = nullptr;

    /// Both take the WHOLE batch and loop per-sequence internally.
    ///
    /// Each sequence attends its own KV slot at its own length, so v1 loops. The
    /// signature takes a batch anyway so a future paged-KV backend with a block
    /// table can batch the loop away without changing this interface.
    StatusCode (*prefill)(const ModelState& model,
                          ExecScratch& exec,
                          SeqBatch& batch,
                          LayerIndex layer,
                          std::span<const float> in,
                          std::span<float> out) noexcept = nullptr;

    StatusCode (*decode)(const ModelState& model,
                         ExecScratch& exec,
                         SeqBatch& batch,
                         LayerIndex layer,
                         std::span<const float> in,
                         std::span<float> out) noexcept = nullptr;

    /// Initialize a freshly allocated region (zero, or write a layout header).
    StatusCode (*init_kv_region)(const ArchIr& arch, KvRegion& region) noexcept = nullptr;
};

/// Resolve a backend for a family. Returns nullptr for an unsupported family.
///
/// This is the ONE place the core switches on AttentionFamily, and it runs once
/// per load. A switch on family anywhere in a loop is a seam violation.
const AttentionBackend* resolve_attention_backend(AttentionFamily family) noexcept;

} // namespace soma
