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

    /// Called exactly once, at load, from load_model().
    ///
    /// MLA folds up-projections into Q here so decode never materializes full
    /// K/V. GQA's implementation is a no-op. This hook exists BECAUSE weight
    /// absorption has no GQA analogue — putting it in the loader or in admission
    /// would leak MLA into the core.
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
