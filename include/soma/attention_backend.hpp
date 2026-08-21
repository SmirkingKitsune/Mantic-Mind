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

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace soma {

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
    /// The core allocates the cache and hands the backend rows into it; it never
    /// learns what is inside. Worked numbers at **fp32**, which is what the cache
    /// actually holds — `KvCache` is a `std::vector<float>` and nothing selects
    /// a narrower dtype:
    ///   Qwen3-30B-A3B   gqa  2×4×128  = 1024 elem/tok/layer × 48 → 192 KiB/tok
    ///   DeepSeek-V2-Lite mla 512+64   =  576 elem/tok/layer × 27 →  61 KiB/tok
    ///
    /// These were written at fp16 — 98 KB and 31 KB — beside implementations
    /// that multiply by `sizeof(float)`. Exactly half the real figure, on the
    /// optimistic side, for the one quantity this comment exists to say is not a
    /// footnote (D45).
    ///
    /// GQA's KV is a first-class planner input: at 32k context Qwen3 wants
    /// **6.4 GB** of the same RAM the expert cache wants, and a planner that
    /// sized the expert cache first would thrash on long contexts and look like
    /// an unrelated bug.
    std::size_t (*kv_bytes_per_token)(const ArchIr& arch) noexcept = nullptr;

    /// Exact per-sequence bytes at a requested context.  Compressed caches are
    /// not linear in context because each layer has its own ratio and fixed
    /// carry/window state. Null falls back to kv_bytes_per_token * context.
    std::uint64_t (*kv_bytes_for_context)(const ArchIr& arch,
                                          std::uint32_t context) noexcept = nullptr;

    /// Opaque caches own their persistence geometry too. The serializer writes
    /// only state live at `length`; the restorer may target a different context
    /// capacity. Existing plane-based formats leave both null and retain their
    /// byte-identical checkpoint path.
    Status (*serialize_kv)(const ArchIr& arch,
                           std::span<const std::byte> source,
                           std::uint32_t source_context,
                           std::uint32_t length,
                           std::vector<std::byte>& payload) = nullptr;
    Status (*restore_kv)(const ArchIr& arch,
                         std::span<const std::byte> payload,
                         std::uint32_t length,
                         std::span<std::byte> destination,
                         std::uint32_t destination_context) = nullptr;

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

    /// Exact attention-owned resident bytes for the whole model.
    ///
    /// Optional and preferred over `weight_bytes_per_layer`. Architectures with
    /// heterogeneous layers cannot be represented exactly by an average: V4's
    /// ratio-4 layers own indexers and its other layers do not. Older uniform
    /// backends leave this null and retain their byte-identical planning path.
    std::uint64_t (*resident_weight_bytes)(const ArchIr& arch, ByteSizer sizer) noexcept = nullptr;
};

// This struct once carried prepare_weights/prefill/decode/init_kv_region as well,
// against ExecScratch, SeqBatch and KvRegion. No family ever implemented any of
// them, nothing ever called them, and the path that actually serves is
// `F32Backend` — quantized SIMD kernels, streamed experts, batch-union CSR,
// ragged batches, per-sequence KV.
//
// They were removed rather than filled in. A second forward path for three
// families, kept in exact agreement with the first, buys nothing measurable: the
// engine is bounded by bytes/token over disk bandwidth and already runs at that
// ceiling (GLM-5.2 reads 14.4 GB per token; 1 tok/s would need 14.4 GB/s
// sustained). Declared-and-never-defined is also the exact shape of D16 and D39,
// where a confident comment over an absent definition read as covered.
//
// What remains is what the planner and the checkpoint reader genuinely ask of a
// backend: how big the cache is, how big the weights are, and which on-disk KV
// layout this family writes.

/// Resolve a backend for a family. Returns nullptr for an unsupported family.
///
/// This is the ONE place the core switches on AttentionFamily, and it runs once
/// per load. A switch on family anywhere in a loop is a seam violation.
const AttentionBackend* resolve_attention_backend(AttentionFamily family) noexcept;

} // namespace soma
