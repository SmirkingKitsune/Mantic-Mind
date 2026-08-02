#pragma once

// Soma — the compiled tokenizer.
//
// Deliberately NOT part of the seam. Admission compiles tokenizer.json into this
// normalized form and the runtime loads it as DATA. Two architectures sharing a
// tokenizer therefore share zero code in arch/, which is correct.
//
// The pretokenizer is a compiled byte-class NFA, not a live regex engine —
// pulling a regex library into the hot path to re-derive a fixed partition every
// prompt is cost for nothing, and it is a runtime dependency the engine
// otherwise does not need.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace soma {

inline constexpr std::uint32_t kTokenizerFormatVersion = 1;

enum class MessageRole : std::uint8_t { System = 0, User, Assistant, Tool };

struct ChatMessage {
    MessageRole role = MessageRole::User;
    std::string_view content;
    std::string_view tool_call_id;
};

/// The chat template, resolved to TOKEN IDS at admission rather than rendered to
/// a string and re-tokenized at runtime. Round-tripping through text is where
/// off-by-one-special-token bugs come from.
struct ChatTemplate {
    std::vector<TokenId> bos;
    std::vector<TokenId> eos;
    std::vector<TokenId> role_prefix[4];
    std::vector<TokenId> role_suffix[4];
    std::vector<TokenId> generation_prompt;
    std::vector<TokenId> thinking_open;
    std::vector<TokenId> thinking_close;
    bool supports_thinking = false;
};

class CompiledTokenizer {
public:
    CompiledTokenizer();
    CompiledTokenizer(const CompiledTokenizer&) = delete;
    CompiledTokenizer& operator=(const CompiledTokenizer&) = delete;
    ~CompiledTokenizer();

    /// Refuses to load across a format-version mismatch.
    Status open(const std::string& compiled_path);
    void close();

    TokenizerKind kind() const noexcept;
    std::uint32_t vocab_size() const noexcept;
    bool byte_fallback() const noexcept;

    Status encode(std::string_view text, std::vector<TokenId>& out) const;
    Status decode(std::span<const TokenId> tokens, std::string& out) const;

    /// Incremental decode for streaming, holding partial UTF-8 across calls so a
    /// multi-byte codepoint split across two tokens never emits a replacement
    /// character mid-stream.
    class Streamer {
    public:
        explicit Streamer(const CompiledTokenizer& tokenizer);
        ~Streamer();
        Status push(TokenId token, std::string& out_delta);
        Status flush(std::string& out_delta);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    const ChatTemplate& chat_template() const noexcept;
    Status apply_chat_template(std::span<const ChatMessage> messages,
                               bool add_generation_prompt,
                               bool enable_thinking,
                               std::vector<TokenId>& out) const;

    bool is_special(TokenId token) const noexcept;
    bool is_eog(TokenId token) const noexcept; ///< end-of-generation

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Round-trip a corpus and compare against the reference recorded at admission.
///
/// ADMISSION IS GATED ON THIS. A tokenizer that does not reproduce HF
/// `tokenizers` byte-for-byte is the cheapest possible bug to catch here and one
/// of the most expensive to catch at G2, where it presents as "the model is
/// subtly stupid" rather than as a tokenizer fault.
Status verify_roundtrip(const CompiledTokenizer& tokenizer,
                        std::span<const std::string> corpus,
                        const std::string& expected_sha,
                        std::string& out_sha);

} // namespace soma
