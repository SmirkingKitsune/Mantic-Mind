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
        // DSA is MLA plus a sparse key indexer, so it IS the MLA backend — which
        // branches on the family internally to select keys before the softmax.
        //
        // This returned nullptr until the indexer existed, and that was the right
        // answer while it did not: sharing the backend then would have run GLM-5.2
        // as dense attention — finite, plausible, and not the model that was
        // asked for. `arch_supported` is derived from this function rather than
        // declared, so flipping this line is what makes the plan's verdict change.
        return &arch::mla::f32_backend();

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
        return &arch::mla::attention_backend();

    case AttentionFamily::MlaDsa:
        // DSA's attention IS MLA plus a learned sparse key indexer, so MLA's
        // SIZING is the right answer minus the indexer's own weights, which are
        // not modelled. Returning it is deliberate and this instance carries no
        // execution members: whether a DSA model can be RUN is answered by
        // resolve_f32_backend, which still says no. Reporting nothing here would
        // size every DSA model's attention at zero, which is a worse lie than
        // an under-count with a reason attached.
        return &arch::mla::attention_backend();

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
