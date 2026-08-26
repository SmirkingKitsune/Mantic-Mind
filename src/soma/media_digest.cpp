// Soma — the media digest chain. See media_digest.hpp for why it exists.
//
// Deliberately NOT in kv_checkpoint_header.cpp. That translation unit is linked
// alone by the node, which reads a checkpoint header before spawning an engine,
// and it names nothing from the model, the kernels or OpenSSL. The header only
// CARRIES these 32 bytes; computing them is the engine's business.

#include "soma/media_digest.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <cstring>

namespace soma {

namespace {

void put_u32_le(unsigned char* p, std::uint32_t v) noexcept {
    for (int i = 0; i < 4; ++i)
        p[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
}

} // namespace

bool MediaDigest::empty() const noexcept {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == 0; });
}

std::string MediaDigest::hex() const {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(bytes.size() * 2);
    for (const auto b : bytes) {
        s.push_back(kHex[(b >> 4) & 0xF]);
        s.push_back(kHex[b & 0xF]);
    }
    return s;
}

void media_digest_fold(MediaDigest& acc,
                       std::uint32_t position,
                       std::span<const float> row) noexcept {
    // Two one-shot hashes rather than SHA256_Init/Update/Final, which OpenSSL 3
    // deprecates and this project builds with warnings as errors. The EVP
    // incremental API is the sanctioned replacement and allocates a context that
    // can fail — leaving this function either not noexcept or silently wrong on
    // OOM. Hashing the row down to 32 bytes first keeps the chain message small
    // enough to live on the stack, so there is nothing to allocate and nothing
    // to fail.
    //
    // Raw float bytes. Byte equality is STRICTER than numeric equality — -0.0
    // and +0.0 digest differently — and the strictness fails in the safe
    // direction: a spurious cold start, never a cache attached to media that did
    // not build it.
    unsigned char row_hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(row.data()), row.size_bytes(), row_hash);

    // acc || position || H(row). The previous accumulator goes in FIRST, so the
    // chain is order-dependent: the same two rows folded in the other order is
    // a different digest, which is what makes "the digest of a prefix" mean
    // something. The position is in there because the same row at a different
    // prompt position is a different context.
    unsigned char message[sizeof(acc.bytes) + 4 + SHA256_DIGEST_LENGTH];
    std::memcpy(message, acc.bytes.data(), acc.bytes.size());
    put_u32_le(message + acc.bytes.size(), position);
    std::memcpy(message + acc.bytes.size() + 4, row_hash, sizeof(row_hash));
    SHA256(message, sizeof(message), acc.bytes.data());
}

MediaDigest media_digest_prefix(std::span<const std::uint32_t> positions,
                                std::span<const float> values,
                                std::uint32_t d_model,
                                std::uint32_t upto) noexcept {
    MediaDigest acc{};
    if (d_model == 0) return acc;
    const auto width = static_cast<std::size_t>(d_model);
    if (values.size() != positions.size() * width) {
        // A shape the caller was supposed to have validated. Returning empty
        // rather than folding what fits: a partial digest is a value that
        // compares unequal to everything and would read as "the media changed".
        return acc;
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] >= upto) break; // ascending, so nothing later qualifies
        media_digest_fold(acc, positions[i], values.subspan(i * width, width));
    }
    return acc;
}

Status validate_prompt_embeddings(const PromptEmbeddings& e,
                                  std::uint32_t d_model,
                                  std::uint32_t prompt_len) {
    if (e.positions.empty()) {
        if (!e.values.empty()) {
            return {StatusCode::InvalidArgument,
                    "prompt embeddings carry " + std::to_string(e.values.size()) +
                        " values for no positions"};
        }
        return {};
    }
    if (d_model == 0) return {StatusCode::InvalidArgument, "d_model is zero"};
    const auto expected = e.positions.size() * static_cast<std::size_t>(d_model);
    if (e.values.size() != expected) {
        return {StatusCode::InvalidArgument,
                "prompt embeddings hold " + std::to_string(e.values.size()) + " values; " +
                    std::to_string(e.positions.size()) + " positions x d_model " +
                    std::to_string(d_model) + " needs " + std::to_string(expected)};
    }
    for (std::size_t i = 0; i < e.positions.size(); ++i) {
        if (e.positions[i] >= prompt_len) {
            return {StatusCode::InvalidArgument,
                    "prompt embedding position " + std::to_string(e.positions[i]) +
                        " is past the end of a " + std::to_string(prompt_len) + "-token prompt"};
        }
        if (i > 0 && e.positions[i] <= e.positions[i - 1]) {
            return {StatusCode::InvalidArgument,
                    "prompt embedding positions must be strictly ascending; " +
                        std::to_string(e.positions[i - 1]) + " is followed by " +
                        std::to_string(e.positions[i])};
        }
    }
    return {};
}

} // namespace soma
