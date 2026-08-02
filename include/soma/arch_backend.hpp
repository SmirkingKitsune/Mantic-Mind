#pragma once

// Soma — the architecture seam.
//
// Everything model-specific is here: attention family, router semantics,
// activation, norm placement, RoPE variant, expert layout, draft head. A struct
// of function pointers plus a compile-time descriptor, resolved once at load.
//
// DEPENDENCY RULE, enforced by CI: include/soma/arch/*.hpp may include core
// headers. Core headers may NOT include arch/, and may not mention an
// architecture-specific identifier. See docs/architecture.md §11.

#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/model.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <span>

namespace soma {

/// Router output for one batch row: which experts fire, and with what weight.
struct RouterOut {
    std::span<ExpertId> expert_ids; ///< [n_rows × top_k]
    std::span<float> weights;       ///< [n_rows × top_k]
};

struct ArchBackend {
    const char* name = nullptr;
    const AttentionBackend* attention = nullptr;

    /// Top-k selection from F32 logits.
    ///
    /// This function decides WHICH EXPERTS FIRE, which is why its input is F32
    /// unconditionally and why quantizing the router is rejected at admission
    /// rather than merely discouraged. Families differ here in ways that are not
    /// parameterizable: sigmoid vs softmax scoring, pre-top-k bias correction,
    /// group-limited routing, and whether weights are renormalized after
    /// selection.
    StatusCode (*route)(const ArchIr& arch,
                        std::span<const float> logits_f32,
                        std::uint32_t n_rows,
                        RouterOut& out) noexcept = nullptr;

    /// Apply one expert's weights to every row that selected it.
    ///
    /// Called once per unique expert per step, with the CSR row list from the
    /// batch-union. The expert's bytes are already resident and pinned by the
    /// caller's ExpertRef — this function must not perform I/O.
    StatusCode (*apply_expert)(const ModelState& model,
                               ExecScratch& exec,
                               LayerIndex layer,
                               ExpertId expert,
                               CByteSpan expert_bytes,
                               std::span<const std::uint32_t> row_indices,
                               std::span<const float> row_weights) noexcept = nullptr;

    /// Dense (non-MoE) FFN, for layers where LayerKind::Dense.
    StatusCode (*dense_ffn)(const ModelState& model,
                            ExecScratch& exec,
                            LayerIndex layer,
                            std::uint32_t n_rows) noexcept = nullptr;

    /// Shared experts, applied to every row unconditionally when present.
    StatusCode (*shared_experts)(const ModelState& model,
                                 ExecScratch& exec,
                                 LayerIndex layer,
                                 std::uint32_t n_rows) noexcept = nullptr;

    StatusCode (*apply_norm)(const ModelState& model,
                             ExecScratch& exec,
                             const DenseTensor& weight,
                             std::uint32_t n_rows) noexcept = nullptr;

    StatusCode (*apply_rope)(const ArchIr& arch,
                             std::span<float> q,
                             std::span<float> k,
                             std::span<const std::uint32_t> positions) noexcept = nullptr;

    /// Admission-time check. Rejects configurations this backend cannot execute
    /// faithfully, BEFORE conversion spends hours on them.
    Status (*validate)(const ArchIr& arch) = nullptr;
};

/// Resolve the backend for a parsed IR. Runs once, at load.
const ArchBackend* resolve_arch_backend(const ArchIr& arch) noexcept;

/// Every backend compiled into this binary. Used by the conformance harness and
/// by `soma admit-verify` to report what this build can accept.
std::span<const ArchBackend* const> registered_arch_backends() noexcept;

} // namespace soma
