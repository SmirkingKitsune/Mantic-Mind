#pragma once

// Soma — quantization, specified per TENSOR ROLE.
//
// Not per tensor, not globally. The role map lives in arch.json §5; the kernels
// that consume it are core. No architecture-specific concept appears here.

#include "soma/types.hpp"

#include <cstdint>
#include <string_view>

namespace soma {

enum class DType : std::uint8_t {
    F32 = 0,
    F16,
    BF16,
    Q8_0,
    Q6_G,
    Q5_G,
    Q4_G,
    Q4_0,
};

/// Tensor roles. Quantization is chosen per role because roles differ in what
/// precision *means* for them — see `Router` below.
enum class TensorRole : std::uint8_t {
    Embed = 0,
    AttnProj,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    SharedExpert,
    Router, ///< MUST be F32. Enforced at admission, not by convention.
    DraftHead,
    Norms,
};

struct QuantSpec {
    DType dtype = DType::F32;
    std::uint32_t group = 0; ///< group-scale width; 0 = per-tensor scale
};

/// The full per-role map, as carried in arch.json.
struct QuantMap {
    QuantSpec embed;
    QuantSpec attn_proj;
    QuantSpec expert_gate;
    QuantSpec expert_up;
    QuantSpec expert_down;
    QuantSpec shared_expert;
    QuantSpec router;
    QuantSpec draft_head;
    QuantSpec norms;

    const QuantSpec& for_role(TensorRole role) const noexcept;
};

/// Rejects any map whose `router` is not F32.
///
/// Quantizing router logits does not degrade precision — it changes WHICH
/// EXPERTS FIRE. That is a semantic change, and a model whose router is
/// quantized is a different model. It will fail conformance stage 2 in a way
/// that reads as a kernel bug for as long as it takes someone to check, which is
/// why this is a hard gate rather than a lint.
Status validate_quant_map(const QuantMap& map);

std::size_t dtype_bits(DType dtype) noexcept;
const char* to_string(DType dtype) noexcept;

/// Inverse of to_string(). Returns false on an unrecognised name.
///
/// A named function rather than a local table in each caller: the container
/// writes these strings and the plan reads them, so the two directions are a
/// WIRE FORMAT. Two independent tables agree until one gains a format.
bool parse_dtype(std::string_view text, DType& out) noexcept;
const char* to_string(TensorRole role) noexcept;

} // namespace soma
