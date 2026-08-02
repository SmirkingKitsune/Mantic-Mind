#include "soma/quant.hpp"

namespace soma {

std::size_t dtype_bits(DType dtype) noexcept {
    switch (dtype) {
    case DType::F32:
        return 32;
    case DType::F16:
        return 16;
    case DType::BF16:
        return 16;
    case DType::Q8_0:
        return 8;
    case DType::Q6_G:
        return 6;
    case DType::Q5_G:
        return 5;
    case DType::Q4_G:
        return 4;
    case DType::Q4_0:
        return 4;
    }
    return 0;
}

const char* to_string(DType dtype) noexcept {
    switch (dtype) {
    case DType::F32:
        return "f32";
    case DType::F16:
        return "f16";
    case DType::BF16:
        return "bf16";
    case DType::Q8_0:
        return "q8_0";
    case DType::Q6_G:
        return "q6_g";
    case DType::Q5_G:
        return "q5_g";
    case DType::Q4_G:
        return "q4_g";
    case DType::Q4_0:
        return "q4_0";
    }
    return "unknown";
}

const char* to_string(TensorRole role) noexcept {
    switch (role) {
    case TensorRole::Embed:
        return "embed";
    case TensorRole::AttnProj:
        return "attn_proj";
    case TensorRole::ExpertGate:
        return "expert_gate";
    case TensorRole::ExpertUp:
        return "expert_up";
    case TensorRole::ExpertDown:
        return "expert_down";
    case TensorRole::SharedExpert:
        return "shared_expert";
    case TensorRole::Router:
        return "router";
    case TensorRole::DraftHead:
        return "draft_head";
    case TensorRole::Norms:
        return "norms";
    }
    return "unknown";
}

const QuantSpec& QuantMap::for_role(TensorRole role) const noexcept {
    switch (role) {
    case TensorRole::Embed:
        return embed;
    case TensorRole::AttnProj:
        return attn_proj;
    case TensorRole::ExpertGate:
        return expert_gate;
    case TensorRole::ExpertUp:
        return expert_up;
    case TensorRole::ExpertDown:
        return expert_down;
    case TensorRole::SharedExpert:
        return shared_expert;
    case TensorRole::Router:
        return router;
    case TensorRole::DraftHead:
        return draft_head;
    case TensorRole::Norms:
        return norms;
    }
    return router;
}

Status validate_quant_map(const QuantMap& map) {
    // Hard gate, not a lint.
    //
    // Quantizing router logits does not degrade precision — it changes WHICH
    // EXPERTS FIRE. A model whose router is quantized is a different model, and
    // it fails conformance stage 2 in a way that reads as a kernel bug for as
    // long as it takes someone to check. See schemas/arch-ir.md §5.
    if (map.router.dtype != DType::F32) {
        return {StatusCode::InvalidArgument,
                std::string("router must be f32, got ") + to_string(map.router.dtype) +
                    "; quantizing the router changes which experts fire, which is a "
                    "semantic change rather than a precision one"};
    }
    if (map.norms.dtype != DType::F32) {
        return {StatusCode::InvalidArgument,
                std::string("norms must be f32, got ") + to_string(map.norms.dtype)};
    }
    return {};
}

} // namespace soma
