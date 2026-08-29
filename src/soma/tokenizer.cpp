// Soma — the compiled tokenizer.
//
// Deliberately NOT part of the seam: admission compiles tokenizer.json into this
// format and the runtime loads it as DATA. Two architectures sharing a tokenizer
// share zero code in arch/, which is correct.
//
// BPE runs in the BYTE domain. HF's byte-level encoding is a bijection between
// bytes and a fixed set of codepoints, so merging over raw bytes is equivalent
// to merging over those codepoints — and means the merge loop needs no Unicode
// handling at all. tools/admission/compile_tokenizer.py decodes the vocab back
// to bytes on the way in.
//
// The pretokenizer is a compiled ordered-alternation program with character
// classes as codepoint-range tables, produced by the same script. Python has
// full Unicode data; the engine carries none.
//
// NOT IMPLEMENTED: NFC normalization. C++ has no normalization without ICU and a
// composition table dwarfs the rest of this format. The compiler verifies every
// calibration string is NFC-stable and reports how many it dropped, so the gate
// states its coverage instead of passing by accident.

#include "soma/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace soma {

namespace {

constexpr char kMagic[8] = {'S', 'O', 'M', 'A', 'T', 'O', 'K', '\0'};
constexpr std::uint32_t kFlagAddPrefixSpace = 1u << 2;
constexpr std::uint32_t kInf = 0xFFFFFFFFu;

constexpr std::uint32_t kItemClass = 0;
constexpr std::uint32_t kItemLiteralCi = 1;

constexpr std::uint32_t kAltPlain = 0;
constexpr std::uint32_t kAltWsNotFollowedByNonSpace = 1;
constexpr std::uint32_t kAltWsThenNewlines = 2;

struct CharClass {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    bool negated = false;

    bool contains(std::uint32_t cp) const noexcept {
        // Ranges are sorted and disjoint; binary search on the upper bound.
        auto it =
            std::lower_bound(ranges.begin(), ranges.end(), cp, [](const auto& r, std::uint32_t v) {
                return r.second < v;
            });
        const bool hit = (it != ranges.end() && cp >= it->first && cp <= it->second);
        return negated ? !hit : hit;
    }
};

struct ProgItem {
    std::uint32_t kind = kItemClass;
    std::uint32_t class_idx = 0;
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
    std::vector<std::string> literals; // kItemLiteralCi
};

struct Alternative {
    std::uint32_t behaviour = kAltPlain;
    std::vector<ProgItem> items;
};

/// Decode one UTF-8 codepoint. Returns bytes consumed, 0 on malformed input.
std::size_t utf8_next(const std::string& s, std::size_t i, std::uint32_t& cp) noexcept {
    const auto n = s.size();
    if (i >= n) return 0;
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        cp = b0;
        return 1;
    }
    auto cont = [&](std::size_t k) -> bool {
        return i + k < n && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
    };
    if ((b0 & 0xE0) == 0xC0 && cont(1)) {
        cp = ((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        return 2;
    }
    if ((b0 & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        cp = ((b0 & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        return 3;
    }
    if ((b0 & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        cp = ((b0 & 0x07u) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        return 4;
    }
    cp = b0;
    return 1; // treat malformed bytes as latin-1, matching HF's leniency
}

struct Reader {
    const std::byte* p = nullptr;
    const std::byte* end = nullptr;
    bool ok = true;

    std::uint32_t u32() noexcept {
        if (!ok || p + 4 > end) {
            ok = false;
            return 0;
        }
        std::uint32_t v = 0;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }

    std::string str() {
        const auto n = u32();
        if (!ok || p + n > end) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

/// Pair key for the merge table. Length-prefixed so ("ab","c") and ("a","bc")
/// cannot collide — they would otherwise hash identically and silently pick the
/// wrong merge rank.
std::string pair_key(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + b.size() + 5);
    k.push_back(static_cast<char>(a.size() & 0xFF));
    k.push_back(static_cast<char>((a.size() >> 8) & 0xFF));
    k += a;
    k.push_back('\0');
    k += b;
    return k;
}

} // namespace

struct CompiledTokenizer::Impl {
    std::uint32_t flags = 0;
    std::vector<std::string> vocab;                       // id -> raw bytes
    std::unordered_map<std::string, std::uint32_t> ids;   // raw bytes -> id
    std::unordered_map<std::string, std::uint32_t> ranks; // pair_key -> rank
    std::vector<CharClass> classes;
    std::vector<Alternative> program;

    struct Added {
        std::string content;
        std::uint32_t id = 0;
        bool special = false;
    };

    std::vector<Added> added;
    ChatTemplate chat;

    // ── pretokenizer ─────────────────────────────────────────────────────────

    std::size_t match_alt(const Alternative& alt, const std::string& text, std::size_t pos) const {
        const auto n = text.size();
        std::size_t i = pos;

        if (!alt.items.empty() && alt.items[0].kind == kItemLiteralCi) {
            for (const auto& lit : alt.items[0].literals) {
                if (pos + lit.size() > n) continue;
                bool eq = true;
                for (std::size_t k = 0; k < lit.size(); ++k) {
                    const char a = text[pos + k];
                    const char b = lit[k];
                    const char la = (a >= 'A' && a <= 'Z') ? static_cast<char>(a + 32) : a;
                    const char lb = (b >= 'A' && b <= 'Z') ? static_cast<char>(b + 32) : b;
                    if (la != lb) {
                        eq = false;
                        break;
                    }
                }
                if (eq) return lit.size();
            }
            return 0;
        }

        if (alt.behaviour == kAltWsNotFollowedByNonSpace) {
            // \s+(?!\S). Greedy \s+ then backtrack one: the run matches in full
            // only at end of text, otherwise all but its last character — which
            // is how the following alternative gets its leading space.
            const auto& ws = classes[alt.items[0].class_idx];
            std::size_t e = pos, last = pos;
            while (e < n) {
                std::uint32_t cp = 0;
                const auto len = utf8_next(text, e, cp);
                if (len == 0 || !ws.contains(cp)) break;
                last = e;
                e += len;
            }
            if (e == pos) return 0;
            if (e == n) return e - pos;
            return (last > pos) ? last - pos : 0;
        }

        if (alt.behaviour == kAltWsThenNewlines) {
            // \s*[\r\n]+. Greedy \s* backtracks so the match ends at the LAST
            // newline in the run; trailing spaces after it belong to the next
            // alternative.
            const auto& ws = classes[alt.items[0].class_idx];
            const auto& nl = classes[alt.items[1].class_idx];
            std::size_t e = pos;
            std::size_t last_nl_end = 0;
            while (e < n) {
                std::uint32_t cp = 0;
                const auto len = utf8_next(text, e, cp);
                if (len == 0 || !ws.contains(cp)) break;
                e += len;
                if (nl.contains(cp)) last_nl_end = e;
            }
            return (last_nl_end > pos) ? last_nl_end - pos : 0;
        }

        for (const auto& item : alt.items) {
            const auto& cls = classes[item.class_idx];
            std::uint32_t count = 0;
            while (i < n && count < item.max_count) {
                std::uint32_t cp = 0;
                const auto len = utf8_next(text, i, cp);
                if (len == 0 || !cls.contains(cp)) break;
                i += len;
                ++count;
            }
            if (count < item.min_count) return 0;
        }
        return i - pos;
    }

    void pretokenize(const std::string& text, std::vector<std::string>& out) const {
        std::size_t pos = 0;
        while (pos < text.size()) {
            std::size_t taken = 0;
            for (const auto& alt : program) {
                taken = match_alt(alt, text, pos);
                if (taken > 0) break;
            }
            if (taken == 0) {
                // No alternative matched: emit one codepoint so the loop cannot
                // stall. Reaching here means the program does not cover the
                // input, which is a compiler bug rather than a runtime one.
                std::uint32_t cp = 0;
                taken = utf8_next(text, pos, cp);
                if (taken == 0) taken = 1;
            }
            out.emplace_back(text, pos, taken);
            pos += taken;
        }
    }

    // ── BPE ──────────────────────────────────────────────────────────────────

    void bpe(const std::string& chunk, std::vector<TokenId>& out) const {
        if (chunk.empty()) return;

        // Start from single bytes.
        std::vector<std::string> parts;
        parts.reserve(chunk.size());
        for (const char c : chunk)
            parts.emplace_back(1, c);

        while (parts.size() > 1) {
            std::uint32_t best_rank = kInf;
            std::size_t best_i = 0;
            for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
                const auto it = ranks.find(pair_key(parts[i], parts[i + 1]));
                if (it != ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_rank == kInf) break;
            parts[best_i] += parts[best_i + 1];
            parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
        }

        for (const auto& part : parts) {
            const auto it = ids.find(part);
            if (it != ids.end()) {
                out.push_back(it->second);
            } else {
                // Unmergeable byte with no vocab entry. Without byte_fallback
                // there is nothing correct to emit, so drop rather than invent
                // an id — and the round-trip gate will catch it as a mismatch.
                for (const char c : part) {
                    const auto b = ids.find(std::string(1, c));
                    if (b != ids.end()) out.push_back(b->second);
                }
            }
        }
    }
};

CompiledTokenizer::CompiledTokenizer() : impl_(std::make_unique<Impl>()) {}

CompiledTokenizer::~CompiledTokenizer() = default;

void CompiledTokenizer::close() {
    impl_ = std::make_unique<Impl>();
}

TokenizerKind CompiledTokenizer::kind() const noexcept {
    return TokenizerKind::Bpe;
}

std::uint32_t CompiledTokenizer::vocab_size() const noexcept {
    return static_cast<std::uint32_t>(impl_->vocab.size());
}

bool CompiledTokenizer::byte_fallback() const noexcept {
    return false;
}

bool CompiledTokenizer::is_special(TokenId token) const noexcept {
    for (const auto& a : impl_->added) {
        if (a.id == token) return a.special;
    }
    return false;
}

bool CompiledTokenizer::is_eog(TokenId token) const noexcept {
    return is_special(token);
}

Status CompiledTokenizer::open(const std::string& path) {
    close();
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "cannot open " + path};

    // Read as char and view as bytes: std::byte has no implicit conversion from
    // char, so istreambuf_iterator cannot fill a vector<byte> directly.
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() < sizeof(kMagic) + 8) {
        return {StatusCode::InvalidArgument, path + ": too small"};
    }
    if (std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
        return {StatusCode::InvalidArgument, path + ": bad magic"};
    }

    const auto* base = reinterpret_cast<const std::byte*>(raw.data());
    Reader r{base + sizeof(kMagic), base + raw.size(), true};
    const auto version = r.u32();
    if (version < kTokenizerMinFormatVersion || version > kTokenizerFormatVersion) {
        return {StatusCode::VersionMismatch,
                path + ": format version " + std::to_string(version) + " is outside " +
                    std::to_string(kTokenizerMinFormatVersion) + ".." +
                    std::to_string(kTokenizerFormatVersion)};
    }

    auto& impl = *impl_;
    impl.flags = r.u32();
    const auto n_vocab = r.u32();
    const auto n_merges = r.u32();
    const auto n_added = r.u32();
    const auto n_classes = r.u32();
    const auto n_alts = r.u32();

    impl.vocab.resize(n_vocab);
    for (std::uint32_t i = 0; i < n_vocab && r.ok; ++i) {
        impl.vocab[i] = r.str();
        if (!impl.vocab[i].empty()) impl.ids.emplace(impl.vocab[i], i);
    }
    for (std::uint32_t i = 0; i < n_merges && r.ok; ++i) {
        auto a = r.str();
        auto b = r.str();
        impl.ranks.emplace(pair_key(a, b), i);
    }
    impl.added.resize(n_added);
    for (std::uint32_t i = 0; i < n_added && r.ok; ++i) {
        impl.added[i].content = r.str();
        impl.added[i].id = r.u32();
        impl.added[i].special = (r.u32() != 0);
    }
    impl.classes.resize(n_classes);
    for (std::uint32_t i = 0; i < n_classes && r.ok; ++i) {
        const auto n_ranges = r.u32();
        impl.classes[i].negated = (r.u32() != 0);
        impl.classes[i].ranges.reserve(n_ranges);
        for (std::uint32_t k = 0; k < n_ranges && r.ok; ++k) {
            const auto lo = r.u32();
            const auto hi = r.u32();
            impl.classes[i].ranges.emplace_back(lo, hi);
        }
    }
    impl.program.resize(n_alts);
    for (std::uint32_t i = 0; i < n_alts && r.ok; ++i) {
        impl.program[i].behaviour = r.u32();
        const auto n_items = r.u32();
        impl.program[i].items.resize(n_items);
        for (std::uint32_t k = 0; k < n_items && r.ok; ++k) {
            auto& item = impl.program[i].items[k];
            item.kind = r.u32();
            if (item.kind == kItemLiteralCi) {
                const auto n_lit = r.u32();
                item.literals.reserve(n_lit);
                for (std::uint32_t q = 0; q < n_lit && r.ok; ++q)
                    item.literals.push_back(r.str());
            } else {
                item.class_idx = r.u32();
                item.min_count = r.u32();
                item.max_count = r.u32();
            }
        }
    }

    // ── the compiled chat template (format 2) ────────────────────────────────
    //
    // A v1 file stops here and carries none. That is not a degraded v2 file: it
    // is a tokenizer for a family whose template the compiler refused, or one
    // compiled before templates existed, and `serve` has an honest fallback for
    // both.
    if (version >= 2) {
        const auto ids = [&r]() {
            std::vector<TokenId> v(r.u32());
            for (auto& id : v)
                id = r.u32();
            return v;
        };
        if (r.u32() != 0) {
            auto& chat = impl.chat;
            chat.present = true;
            chat.flags = r.u32();
            chat.bos = ids();
            const auto n_prologues = r.u32();
            chat.prologues.resize(n_prologues);
            for (auto& p : chat.prologues) {
                p.effort = r.str();
                p.enable_thinking = (r.u32() != 0);
                p.ids = ids();
            }
            for (std::size_t role = 0; role < 4; ++role) {
                chat.run_prefix[role] = ids();
                chat.prefix[role] = ids();
                chat.suffix[role] = ids();
                chat.run_suffix[role] = ids();
            }
            chat.assistant_prefix_thinking = ids();
            chat.thinking_close = ids();
            chat.generation_prompt = ids();
            chat.generation_prompt_nothink = ids();
        }
    }

    if (!r.ok) return {StatusCode::InvalidArgument, path + ": truncated"};

    // Longest content first, so a token that is a prefix of another cannot win.
    std::sort(impl.added.begin(), impl.added.end(), [](const auto& a, const auto& b) {
        return a.content.size() > b.content.size();
    });
    return {};
}

Status CompiledTokenizer::encode(std::string_view text, std::vector<TokenId>& out) const {
    out.clear();
    if (impl_->vocab.empty()) return {StatusCode::InvalidArgument, "tokenizer not loaded"};

    std::string work(text);
    if ((impl_->flags & kFlagAddPrefixSpace) != 0 && !work.empty() && work[0] != ' ') {
        work.insert(work.begin(), ' ');
    }
    // NFC (the tokenizer format's bit-0 normalization flag) is intentionally not
    // applied — see the file header. The compiler guarantees the calibration corpus
    // is NFC-stable and reports what it dropped, so this is a stated limitation
    // rather than a silent one.

    std::vector<std::string> chunks;

    // Added tokens are matched against the RAW text before pretokenization and
    // must survive verbatim: BPE splitting "<|im_start|>" would be wrong in a
    // way that produces valid-looking ids.
    std::size_t pos = 0;
    while (pos < work.size()) {
        std::size_t hit_at = std::string::npos;
        const Impl::Added* hit = nullptr;
        for (const auto& a : impl_->added) {
            if (a.content.empty()) continue;
            const auto at = work.find(a.content, pos);
            if (at != std::string::npos && at < hit_at) {
                hit_at = at;
                hit = &a;
            }
        }
        if (hit == nullptr) {
            chunks.clear();
            impl_->pretokenize(work.substr(pos), chunks);
            for (const auto& c : chunks)
                impl_->bpe(c, out);
            break;
        }
        if (hit_at > pos) {
            chunks.clear();
            impl_->pretokenize(work.substr(pos, hit_at - pos), chunks);
            for (const auto& c : chunks)
                impl_->bpe(c, out);
        }
        out.push_back(hit->id);
        pos = hit_at + hit->content.size();
    }
    return {};
}

Status CompiledTokenizer::decode(std::span<const TokenId> tokens, std::string& out) const {
    out.clear();
    for (const auto t : tokens) {
        if (t >= impl_->vocab.size()) {
            return {StatusCode::InvalidArgument, "token id " + std::to_string(t) + " out of range"};
        }
        out += impl_->vocab[t];
    }
    return {};
}

// ── incremental decode ───────────────────────────────────────────────────────
//
// decode() is a concatenation of vocab entries, so a token's bytes do not depend
// on its neighbours and streaming is exact rather than approximate. The only
// thing that cannot be decided from one token is where a UTF-8 codepoint ENDS: a
// byte-fallback vocabulary emits one token per byte, so a three-byte CJK
// character arrives as three tokens and the first two are not text.
//
// The invariant, and what the test checks: for any token sequence, the deltas
// from push() concatenated with flush() equal decode() of the same sequence,
// byte for byte. Streaming changes when bytes are handed over, never which.

namespace {

/// Bytes occupied by the sequence this byte leads, or 0 if it does not lead one
/// (a continuation byte, or an invalid lead).
std::size_t utf8_seq_len(unsigned char b) noexcept {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

/// Length of the longest prefix of `buf` that ends on a codepoint boundary.
///
/// Bounded at three bytes of lookback, which is what makes malformed input safe:
/// an incomplete sequence is at most a lead plus two continuations, so anything
/// further back is complete no matter what it is. Bytes that are not a valid
/// prefix of a multi-byte sequence are released rather than held, matching
/// utf8_next()'s latin-1 leniency — holding them would stall the stream forever
/// waiting for a continuation that is never coming.
std::size_t complete_prefix(const std::string& buf) noexcept {
    const auto n = buf.size();
    const auto lookback = n < 3 ? n : std::size_t{3};
    for (std::size_t back = 1; back <= lookback; ++back) {
        const auto i = n - back;
        const auto len = utf8_seq_len(static_cast<unsigned char>(buf[i]));
        if (len == 0) continue; // continuation byte: keep walking back to its lead
        return (len <= back) ? n : i;
    }
    return n;
}

} // namespace

struct CompiledTokenizer::Streamer::Impl {
    const CompiledTokenizer* tok = nullptr;
    std::string held; ///< a codepoint's bytes so far, never more than three
};

CompiledTokenizer::Streamer::Streamer(const CompiledTokenizer& tokenizer)
    : impl_(std::make_unique<Impl>()) {
    impl_->tok = &tokenizer;
}

CompiledTokenizer::Streamer::~Streamer() = default;

Status CompiledTokenizer::Streamer::push(TokenId token, std::string& out_delta) {
    out_delta.clear();
    const auto& vocab = impl_->tok->impl_->vocab;
    if (token >= vocab.size()) {
        return {StatusCode::InvalidArgument, "token id " + std::to_string(token) + " out of range"};
    }
    impl_->held += vocab[token];
    const auto cut = complete_prefix(impl_->held);
    if (cut == 0) return {};
    out_delta.assign(impl_->held, 0, cut);
    impl_->held.erase(0, cut);
    return {};
}

// ── the chat template ────────────────────────────────────────────────────────

const std::vector<TokenId>& ChatTemplate::prologue_for(const std::string& effort,
                                                       bool enable_thinking) const noexcept {
    for (const auto& want_effort : {effort, std::string{}}) {
        for (const auto& p : prologues) {
            if (p.enable_thinking == enable_thinking && p.effort == want_effort) return p.ids;
        }
    }
    return bos;
}

const ChatTemplate& CompiledTokenizer::chat_template() const noexcept {
    return impl_->chat;
}

namespace {

/// Split an assistant turn's content into `(content, reasoning)`.
///
/// Mirrors `split_reasoning` in tools/admission/chat_template.py, which is the
/// implementation the oracle was generated against. Three states, not two: a
/// template may re-emit a historical `<think>` block (GLM), remove it (Qwen3),
/// or have no thinking channel at all — and passing a removed block through
/// would hand the model its own scratchpad back as though it were the answer.
void split_reasoning(const ChatTemplate& chat,
                     const ChatMessage& message,
                     std::size_t index,
                     std::ptrdiff_t last_user,
                     bool clear_thinking,
                     std::string_view& content,
                     std::string_view& reasoning) {
    content = message.content;
    reasoning = {};
    if (message.role != MessageRole::Assistant) return;

    const bool thinking_aware = chat.has(chat_flag::kSupportsThinking) ||
                                chat.has(chat_flag::kAssistantDropsThink);
    if (thinking_aware) {
        static constexpr std::string_view kClose = "</think>";
        static constexpr std::string_view kOpen = "<think>";
        if (const auto close = content.find(kClose); close != std::string_view::npos) {
            auto head = content.substr(0, close);
            content = content.substr(close + kClose.size());
            const auto open = head.rfind(kOpen);
            reasoning = (open == std::string_view::npos) ? head : head.substr(open + kOpen.size());
        }
    }
    if (chat.has(chat_flag::kAssistantDropsThink)) reasoning = {};
    if (clear_thinking && static_cast<std::ptrdiff_t>(index) <= last_user) reasoning = {};

    if (chat.has(chat_flag::kAssistantStrips)) {
        const auto is_space = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        };
        while (!content.empty() && is_space(content.front())) content.remove_prefix(1);
        while (!content.empty() && is_space(content.back())) content.remove_suffix(1);
    }
}

} // namespace

Status CompiledTokenizer::apply_chat_template(std::span<const ChatMessage> messages,
                                              const ChatOptions& options,
                                              std::vector<TokenId>& out) const {
    out.clear();
    const auto& chat = impl_->chat;
    if (!chat.present) {
        return {StatusCode::Unsupported,
                "this container carries no compiled chat template"};
    }

    // Refuse an option this template cannot honour rather than ignoring it.
    // Silently accepting `enable_thinking: false` on a template with no such
    // switch tells the caller something false, and knowing which templates have
    // one is the entire reason admission measured them.
    if (!options.enable_thinking && !chat.has(chat_flag::kEnableThinking)) {
        return {StatusCode::Unsupported,
                "this model's chat template has no enable_thinking switch; it "
                "frames every turn the same way whether or not thinking is asked for"};
    }
    if (options.clear_thinking_set && !chat.has(chat_flag::kClearThinkingSettable)) {
        return {StatusCode::Unsupported,
                "this model's chat template does not take clear_thinking; it "
                "always " +
                    std::string(chat.has(chat_flag::kClearThinkingDefault)
                                    ? "drops"
                                    : "keeps") +
                    " reasoning from turns before the last user message"};
    }
    if (!options.reasoning_effort.empty() && !chat.has(chat_flag::kReasoningEffort)) {
        return {StatusCode::Unsupported,
                "this model's chat template does not take reasoning_effort"};
    }

    const bool clear_thinking = (options.clear_thinking_set &&
                                 chat.has(chat_flag::kClearThinkingSettable))
                                    ? options.clear_thinking
                                    : chat.has(chat_flag::kClearThinkingDefault);

    std::ptrdiff_t last_user = -1;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == MessageRole::User) last_user = static_cast<std::ptrdiff_t>(i);
    }

    const auto append = [&out](const std::vector<TokenId>& ids) {
        out.insert(out.end(), ids.begin(), ids.end());
    };
    std::vector<TokenId> scratch;
    const auto append_text = [&](std::string_view text) -> Status {
        if (text.empty()) return {};
        if (auto st = encode(text, scratch); !st.ok()) return st;
        out.insert(out.end(), scratch.begin(), scratch.end());
        return {};
    };

    append(chat.prologue_for(options.reasoning_effort, options.enable_thinking));

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        const auto role = static_cast<std::size_t>(m.role);
        if (role >= 4) return {StatusCode::InvalidArgument, "message role out of range"};
        if (i == 0 || messages[i - 1].role != m.role) append(chat.run_prefix[role]);

        std::string_view content;
        std::string_view reasoning;
        split_reasoning(chat, m, i, last_user, clear_thinking, content, reasoning);

        // The two assistant prefixes are ALTERNATIVES, not a base plus an
        // insertion: prefix[Assistant] already carries the empty thinking block,
        // so emitting both would emit it twice.
        if (!reasoning.empty()) {
            append(chat.assistant_prefix_thinking);
            if (auto st = append_text(reasoning); !st.ok()) return st;
            append(chat.thinking_close);
        } else {
            append(chat.prefix[role]);
        }
        if (auto st = append_text(content); !st.ok()) return st;
        append(chat.suffix[role]);
        if (i + 1 == messages.size() || messages[i + 1].role != m.role) {
            append(chat.run_suffix[role]);
        }
    }

    if (options.add_generation_prompt) {
        append(options.enable_thinking ? chat.generation_prompt
                                       : chat.generation_prompt_nothink);
    }
    return {};
}

// ── the admission gate ───────────────────────────────────────────────────────

Status read_tokenizer_oracle(const std::string& path, std::vector<TokenizerOracleCase>& out) {
    out.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "no tokenizer oracle at " + path};

    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMATORC", 8) != 0) {
        return {StatusCode::InvalidArgument, path + ": not a tokenizer oracle (bad magic)"};
    }
    const auto u32 = [&]() -> std::uint32_t {
        std::uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    if (const auto version = u32(); version != 1) {
        return {StatusCode::VersionMismatch,
                path + ": oracle version " + std::to_string(version) + " != 1"};
    }
    const auto n = u32();
    out.resize(n);
    for (auto& c : out) {
        const auto len = u32();
        c.text.resize(len);
        if (len > 0) in.read(c.text.data(), len);
        const auto k = u32();
        c.ids.resize(k);
        for (auto& id : c.ids)
            id = u32();
    }
    if (!in) return {StatusCode::InvalidArgument, path + ": truncated oracle"};
    return {};
}

Status verify_roundtrip(const CompiledTokenizer& tokenizer,
                        std::span<const TokenizerOracleCase> oracle,
                        RoundTripResult& out) {
    out = {};
    out.cases = static_cast<std::uint32_t>(oracle.size());
    if (oracle.empty()) {
        // An empty corpus passes every check there is, which is exactly why it
        // must not be allowed to look like a pass.
        return {StatusCode::InvalidArgument, "the oracle is empty; there is nothing to verify"};
    }

    std::vector<TokenId> ids;
    std::string round;
    for (const auto& c : oracle) {
        if (auto st = tokenizer.encode(c.text, ids); st.ok()) {
            if (ids == c.ids) {
                ++out.encode_ok;
            } else if (out.first_failure.empty()) {
                out.first_failure = "encode \"" + c.text.substr(0, 40) + "\": got " +
                                    std::to_string(ids.size()) + " ids, want " +
                                    std::to_string(c.ids.size());
                for (std::size_t i = 0; i < ids.size() && i < c.ids.size(); ++i) {
                    if (ids[i] != c.ids[i]) {
                        out.first_failure += " (first diff at " + std::to_string(i) + ": " +
                                             std::to_string(ids[i]) + " vs " +
                                             std::to_string(c.ids[i]) + ")";
                        break;
                    }
                }
            }
        } else if (out.first_failure.empty()) {
            out.first_failure = "encode \"" + c.text.substr(0, 40) + "\": " + st.message();
        }

        // Decode is checked against HF's ids, not ours: otherwise an encode bug
        // that a decode bug happens to invert would pass both halves.
        if (tokenizer.decode(c.ids, round).ok() && round == c.text) {
            ++out.decode_ok;
        } else if (out.first_failure.empty()) {
            out.first_failure = "decode of \"" + c.text.substr(0, 40) + "\" did not round-trip";
        }
    }
    return {};
}

Status read_chat_oracle(const std::string& path, std::vector<ChatOracleCase>& out) {
    out.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "no chat oracle at " + path};

    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMACHAT", 8) != 0) {
        return {StatusCode::InvalidArgument, path + ": not a chat oracle (bad magic)"};
    }
    const auto u32 = [&]() -> std::uint32_t {
        std::uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    const auto str = [&]() {
        std::string s(u32(), '\0');
        if (!s.empty()) in.read(s.data(), static_cast<std::streamsize>(s.size()));
        return s;
    };
    if (const auto version = u32(); version != kChatOracleFormatVersion) {
        return {StatusCode::VersionMismatch,
                path + ": chat oracle version " + std::to_string(version) +
                    " != " + std::to_string(kChatOracleFormatVersion)};
    }
    out.resize(u32());
    for (auto& c : out) {
        const auto n = u32();
        c.roles.resize(n);
        c.contents.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto role = u32();
            if (role >= 4) return {StatusCode::InvalidArgument, path + ": role out of range"};
            c.roles[i] = static_cast<MessageRole>(role);
            c.contents[i] = str();
        }
        c.options.add_generation_prompt = (u32() != 0);
        c.options.enable_thinking = (u32() != 0);
        // Tri-state on the wire: 0 unset, 1 false, 2 true. "Unset" is a
        // different request from "false" on every template whose own default is
        // true, and collapsing the two would make the oracle unable to grade the
        // case that distinguishes GLM-5.2 from GLM-5.3.
        const auto clear = u32();
        c.options.clear_thinking_set = (clear != 0);
        c.options.clear_thinking = (clear == 2);
        c.options.reasoning_effort = str();
        c.ids.resize(u32());
        for (auto& id : c.ids)
            id = u32();
    }
    if (!in) return {StatusCode::InvalidArgument, path + ": truncated chat oracle"};
    return {};
}

Status verify_chat_template(const CompiledTokenizer& tokenizer,
                            const std::vector<ChatOracleCase>& cases,
                            RoundTripResult& out) {
    out = {};
    out.cases = static_cast<std::uint32_t>(cases.size());
    if (cases.empty()) {
        // An empty oracle passes every check there is, which is exactly why it
        // must not be allowed to look like one.
        return {StatusCode::InvalidArgument, "the chat oracle is empty; nothing to verify"};
    }
    // There is no decode half here — a prompt is assembled, never round-tripped
    // — so decode_ok tracks encode_ok to keep `clean()` meaning what it says
    // rather than silently reporting half a result as a pass.
    std::vector<TokenId> ids;
    std::vector<ChatMessage> messages;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const auto& oracle = cases[c];
        messages.clear();
        for (std::size_t i = 0; i < oracle.roles.size(); ++i) {
            messages.push_back(ChatMessage{oracle.roles[i], oracle.contents[i], {}});
        }
        const auto st = tokenizer.apply_chat_template(messages, oracle.options, ids);
        if (!st.ok()) {
            if (out.first_failure.empty()) {
                out.first_failure = "case " + std::to_string(c) + ": " + st.message();
            }
            continue;
        }
        if (ids == oracle.ids) {
            ++out.encode_ok;
            ++out.decode_ok;
            continue;
        }
        if (!out.first_failure.empty()) continue;
        out.first_failure = "case " + std::to_string(c) + " (" +
                            std::to_string(oracle.roles.size()) + " message(s)): got " +
                            std::to_string(ids.size()) + " ids, want " +
                            std::to_string(oracle.ids.size());
        for (std::size_t i = 0; i < ids.size() && i < oracle.ids.size(); ++i) {
            if (ids[i] != oracle.ids[i]) {
                out.first_failure += " (first diff at " + std::to_string(i) + ": " +
                                     std::to_string(ids[i]) + " vs " +
                                     std::to_string(oracle.ids[i]) + ")";
                break;
            }
        }
    }
    return {};
}

Status CompiledTokenizer::Streamer::flush(std::string& out_delta) {
    // Unconditional, including a partial codepoint. At end of stream the missing
    // bytes are not late, they are absent — and withholding them would make the
    // streamed text differ from decode(), which is the one thing this must not do.
    out_delta = std::move(impl_->held);
    impl_->held.clear();
    return {};
}

} // namespace soma
