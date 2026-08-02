// Soma — the architecture resolver.
//
// THE single core translation unit permitted to include soma/arch/ and to
// switch on AttentionFamily. tools/ci/check_seam.py allow-lists exactly this
// file and the backends themselves.
//
// Everything here runs ONCE per model load. A switch on family anywhere in a
// loop is a seam violation regardless of which file it appears in — the point
// of resolving to a descriptor is that the hot path never asks the question
// again.

#include "soma/arch/gqa.hpp"
#include "soma/arch/mla.hpp"
#include "soma/arch_backend.hpp"
#include "soma/f32_model.hpp"

#include <array>

namespace soma {

const F32Backend* resolve_f32_backend(const ArchIr& arch) noexcept {
    switch (arch.attention.family) {
    case AttentionFamily::Mha:
    case AttentionFamily::Gqa:
        // MHA is the n_kv_heads == n_heads case of the same backend: the
        // only difference is a repeat factor of 1. Giving it a separate
        // implementation would mean two code paths to keep in agreement for
        // no behavioural gain.
        return &arch::gqa::f32_backend();

    case AttentionFamily::Mla:
        return &arch::mla::f32_backend();

    case AttentionFamily::MlaDsa:
        // DSA adds a sparse key indexer on top of MLA. Sharing the MLA
        // backend would run it as dense attention: finite, plausible, and
        // not the model that was asked for.
        return nullptr;

    case AttentionFamily::Unknown:
        return nullptr;
    }
    return nullptr;
}

const AttentionBackend* resolve_attention_backend(AttentionFamily family) noexcept {
    switch (family) {
    case AttentionFamily::Mha:
    case AttentionFamily::Gqa:
        return &arch::gqa::attention_backend();
    case AttentionFamily::Mla:
    case AttentionFamily::MlaDsa:
    case AttentionFamily::Unknown:
        return nullptr;
    }
    return nullptr;
}

const ArchBackend* resolve_arch_backend(const ArchIr& arch) noexcept {
    switch (arch.attention.family) {
    case AttentionFamily::Mha:
    case AttentionFamily::Gqa:
        return &arch::gqa::backend();
    case AttentionFamily::Mla:
    case AttentionFamily::MlaDsa:
    case AttentionFamily::Unknown:
        return nullptr;
    }
    return nullptr;
}

std::span<const ArchBackend* const> registered_arch_backends() noexcept {
    // Reported by `soma admit-verify` so an operator can see what this build
    // accepts before spending hours converting something it cannot run.
    static const std::array<const ArchBackend*, 1> kAll{&arch::gqa::backend()};
    return {kAll.data(), kAll.size()};
}

} // namespace soma
