#pragma once

#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/prompt_codec.hpp"

namespace soma::arch::compressed_sparse {

const soma::F32Backend& f32_backend() noexcept;
const soma::AttentionBackend& attention_backend() noexcept;
const soma::PromptCodec& prompt_codec() noexcept;

} // namespace soma::arch::compressed_sparse
