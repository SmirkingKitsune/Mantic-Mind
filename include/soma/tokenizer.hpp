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

/// 2 adds the compiled chat template. Version 1 is still ACCEPTED and means the
/// file carries none — a real state (a family whose template the compiler
/// refused, or a fixture compiled before templates existed) rather than a
/// defect. Bumping rather than appending-and-hoping is what keeps a truncated
/// file distinguishable from an older one.
inline constexpr std::uint32_t kTokenizerFormatVersion = 2;
inline constexpr std::uint32_t kTokenizerMinFormatVersion = 1;
inline constexpr std::uint32_t kChatOracleFormatVersion = 1;

enum class MessageRole : std::uint8_t { System = 0, User, Assistant, Tool };

struct ChatMessage {
    MessageRole role = MessageRole::User;
    std::string_view content;
    std::string_view tool_call_id;
};

/// Flags describing what the source template actually does. Every one of them
/// was MEASURED against the real Jinja renderer by tools/admission/chat_template.py
/// — none is inferred from template source — and the numbering is mirrored there.
namespace chat_flag {
inline constexpr std::uint32_t kSupportsThinking = 1u << 0;
inline constexpr std::uint32_t kAssistantSplitsThink = 1u << 1;
inline constexpr std::uint32_t kAssistantStrips = 1u << 2;
inline constexpr std::uint32_t kClearThinkingSettable = 1u << 3;
inline constexpr std::uint32_t kClearThinkingDefault = 1u << 4;
inline constexpr std::uint32_t kEnableThinking = 1u << 5;
inline constexpr std::uint32_t kReasoningEffort = 1u << 6;
inline constexpr std::uint32_t kAssistantDropsThink = 1u << 7;
} // namespace chat_flag

/// One prologue, selected by `(reasoning_effort, enable_thinking)`.
///
/// A product rather than two independent fields because the two are not
/// independent: GLM-5.2 suppresses its whole `Reasoning Effort` system block when
/// thinking is off, at which point the effort it was asked for stops mattering.
struct ChatPrologue {
    std::string effort; ///< empty = this template's default effort
    bool enable_thinking = true;
    std::vector<TokenId> ids;
};

/// The chat template, resolved to TOKEN IDS at admission rather than rendered to
/// a string and re-tokenized at runtime. Round-tripping through text is where
/// off-by-one-special-token bugs come from.
///
/// This is a SCAFFOLD, not a program. Admission ran the model's real Jinja
/// template against probe conversations and read the framing off the text around
/// each probe; what is left for the engine is concatenation. There is no Jinja
/// interpreter here and there must not be one — a second renderer would have to
/// stay bug-for-bug identical with the first forever, and when it drifted the
/// symptom would be a correctly-served model answering a subtly differently
/// framed prompt.
struct ChatTemplate {
    /// False when the checkpoint shipped no template, or shipped one this
    /// compiler refused. Both are real states and neither is a defect; `serve`
    /// falls back to flattening messages, which is honest and visibly worse.
    bool present = false;
    std::uint32_t flags = 0;

    std::vector<TokenId> bos;
    std::vector<ChatPrologue> prologues;

    /// Four per role, because a RUN of same-role messages is framed once as a
    /// run and once per message. GLM-5.3 opens a run of tool results with
    /// `<|observation|>` and wraps each in `<tool_response>`; Qwen3 closes the
    /// run with `<|im_end|>`. Prefix and suffix alone get one family wrong.
    std::vector<TokenId> run_prefix[4];
    std::vector<TokenId> prefix[4];
    std::vector<TokenId> suffix[4];
    std::vector<TokenId> run_suffix[4];

    /// The assistant prefix has two spellings and both are measured rather than
    /// composed: `prefix[Assistant]` already carries whatever empty thinking
    /// block the template emits (`<think></think>` on GLM, nothing on Qwen3),
    /// while this is what a turn WITH reasoning opens with. They are
    /// ALTERNATIVES — emitting both would emit the empty block twice.
    std::vector<TokenId> assistant_prefix_thinking;
    std::vector<TokenId> thinking_close;

    std::vector<TokenId> generation_prompt;
    std::vector<TokenId> generation_prompt_nothink;

    bool has(std::uint32_t flag) const noexcept { return (flags & flag) != 0; }

    /// Exact match on both axes, then the default effort at this thinking
    /// setting, then the plain prologue. The middle step is what makes an
    /// unrecognized effort behave the way the TEMPLATE behaves: GLM-5.3 renders
    /// `medium` as `Max` rather than erroring, and an engine that refused it
    /// would be stricter than the model.
    const std::vector<TokenId>& prologue_for(const std::string& effort,
                                             bool enable_thinking) const noexcept;
};

/// What a request may ask of the template beyond the messages themselves.
struct ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    /// UNSET is not the same request as `false`. GLM-5.2 clears prior reasoning
    /// by default and GLM-5.3 keeps it, so a caller who said nothing must get
    /// the template's own answer rather than either literal.
    bool clear_thinking = false;
    bool clear_thinking_set = false;
    std::string reasoning_effort;
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

    /// Build a prompt's token ids from a conversation.
    ///
    /// Refuses rather than ignoring an option the compiled template cannot
    /// honour: a caller who asks for `enable_thinking: false` on a template that
    /// has no such switch has been told something false by a silent success, and
    /// the whole point of measuring the template was to know which is which.
    Status apply_chat_template(std::span<const ChatMessage> messages,
                               const ChatOptions& options,
                               std::vector<TokenId>& out) const;

    bool is_special(TokenId token) const noexcept;
    bool is_eog(TokenId token) const noexcept; ///< end-of-generation

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// One calibration string and the ids HF's `tokenizers` produced for it.
struct TokenizerOracleCase {
    std::string text;
    std::vector<TokenId> ids;
};

/// Read a `SOMATORC` oracle, as written by tools/admission/compile_tokenizer.py.
///
/// In the library rather than in the test that first needed it, for the same
/// reason the KV checkpoint header is: it is a FORMAT with more than one reader,
/// and two parsers that must agree is how they stop agreeing. The comparison
/// stays with each caller — one parser, independent verdicts.
Status read_tokenizer_oracle(const std::string& path, std::vector<TokenizerOracleCase>& out);

/// One conversation and the ids HF's own renderer plus `tokenizers` produced.
struct ChatOracleCase {
    std::vector<MessageRole> roles;
    std::vector<std::string> contents;
    ChatOptions options;
    std::vector<TokenId> ids;
};

/// Read a `SOMACHAT` oracle, as written by tools/admission/compile_tokenizer.py.
///
/// This is the grader for the whole chat-template mechanism, and it is a real
/// one rather than a restatement: the ids on the right were produced by rendering
/// the model's OWN Jinja template and tokenizing the result as one string, so an
/// engine that assembles the same conversation out of precompiled pieces and
/// lands on the same ids has been checked against the model's actual framing —
/// not against a second copy of this engine's opinion of it.
Status read_chat_oracle(const std::string& path, std::vector<ChatOracleCase>& out);

struct RoundTripResult {
    std::uint32_t cases = 0;
    std::uint32_t encode_ok = 0; ///< ids identical to HF's
    std::uint32_t decode_ok = 0; ///< decode(HF's ids) reproduces the source text

    /// The first case that failed, rendered for a human. Empty when clean —
    /// "which one and where" is the whole difference between a usable tokenizer
    /// bug report and "conformance failed".
    std::string first_failure;

    bool clean() const noexcept { return cases > 0 && encode_ok == cases && decode_ok == cases; }
};

/// Round-trip the oracle's corpus and compare against HF's own answer.
///
/// ADMISSION IS GATED ON THIS. A tokenizer that does not reproduce HF
/// `tokenizers` byte-for-byte is the cheapest possible bug to catch here and one
/// of the most expensive to catch at G2, where it presents as "the model is
/// subtly stupid" rather than as a tokenizer fault.
///
/// Takes the oracle's IDS rather than a digest of them. A hash can only say
/// "different"; the ids say which case, which position, and what was expected —
/// and the digest bought nothing, since both sides are on the same host.
Status verify_roundtrip(const CompiledTokenizer& tokenizer,
                        std::span<const TokenizerOracleCase> oracle,
                        RoundTripResult& out);

/// Rebuild each oracle conversation and compare ids. `first_failure` names the
/// case and the position, because "the chat template does not match" is a bug
/// report nobody can act on.
Status verify_chat_template(const CompiledTokenizer& tokenizer,
                            const std::vector<ChatOracleCase>& cases,
                            RoundTripResult& out);


} // namespace soma
