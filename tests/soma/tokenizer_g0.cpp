// Soma — G0 tokenizer round-trip against HF `tokenizers`.
//
// Gate: the compiled tokenizer reproduces HF's ids BYTE-FOR-BYTE over the
// calibration corpus, and decode(encode(x)) == x.
//
// This is the cheapest possible place to catch a tokenizer fault and one of the
// most expensive to catch later: at G2 a mis-tokenizing model does not crash, it
// presents as "the model is subtly stupid".
//
// Usage: tokenizer_g0 <tokenizers_dir> [name ...]

#include "soma/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Case = soma::TokenizerOracleCase;

std::string escape(const std::string& s, std::size_t cap = 40) {
    std::ostringstream o;
    for (std::size_t i = 0; i < s.size() && o.str().size() < cap; ++i) {
        const char c = s[i];
        if (c == '\n')
            o << "\\n";
        else if (c == '\r')
            o << "\\r";
        else if (c == '\t')
            o << "\\t";
        else if (static_cast<unsigned char>(c) < 0x20)
            o << '?';
        else
            o << c;
    }
    return o.str();
}

/// Does `s` end on a codepoint boundary?
///
/// Written out here rather than reusing the implementation's helper, on purpose:
/// a test that calls the function under test to decide whether the function under
/// test is right proves only that it is self-consistent. Malformed bytes are
/// treated as latin-1, matching what the tokenizer's own utf8_next does.
bool ends_on_codepoint_boundary(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        std::size_t len = 1;
        if ((b & 0xE0) == 0xC0)
            len = 2;
        else if ((b & 0xF0) == 0xE0)
            len = 3;
        else if ((b & 0xF8) == 0xF0)
            len = 4;
        if (len > 1) {
            if (i + len > s.size()) return false; // truncated at the end
            for (std::size_t k = 1; k < len; ++k) {
                if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) {
                    len = 1; // not actually a sequence; latin-1 that lead byte
                    break;
                }
            }
        }
        i += len;
    }
    return true;
}

struct Report {
    bool loaded = false;
    bool skipped = false;
    std::string skip_reason;
    int cases = 0, encode_ok = 0, decode_ok = 0;

    /// Streaming: deltas concatenated with the flush tail must equal decode().
    int stream_ok = 0;
    /// Deltas that were held back because a codepoint was still incomplete —
    /// reported so a green stream column cannot mean "no split ever happened".
    int held = 0;
    /// The synthetic byte-fallback probe: -1 not applicable, 0 failed, 1 passed.
    int split_probe = -1;

    std::string first_failure;
};

Report run(const fs::path& dir) {
    Report rep;

    if (fs::exists(dir / "tokenizer.unsupported")) {
        std::ifstream in(dir / "tokenizer.unsupported");
        std::getline(in, rep.skip_reason);
        rep.skipped = true;
        return rep;
    }

    soma::CompiledTokenizer tok;
    if (auto st = tok.open((dir / "tokenizer.soma").string()); !st.ok()) {
        rep.first_failure = st.message();
        return rep;
    }
    rep.loaded = true;

    // The oracle PARSER is the library's — one reader for one format, the same
    // rule the KV checkpoint header follows. The comparisons below stay here, so
    // the test still reaches its own verdict.
    std::vector<Case> cases;
    if (auto st = soma::read_tokenizer_oracle((dir / "tokenizer_oracle.bin").string(), cases);
        !st.ok()) {
        rep.first_failure = st.message();
        return rep;
    }
    rep.cases = static_cast<int>(cases.size());

    std::vector<soma::TokenId> ids;
    std::string round;
    for (const auto& c : cases) {
        if (auto st = tok.encode(c.text, ids); !st.ok()) {
            if (rep.first_failure.empty()) rep.first_failure = st.message();
            continue;
        }
        const bool match = (ids == c.ids);
        if (match) {
            ++rep.encode_ok;
        } else if (rep.first_failure.empty()) {
            std::ostringstream o;
            o << "\"" << escape(c.text) << "\" got " << ids.size() << " ids, want " << c.ids.size();
            for (std::size_t i = 0; i < std::min(ids.size(), c.ids.size()); ++i) {
                if (ids[i] != c.ids[i]) {
                    o << " (first diff at " << i << ": " << ids[i] << " vs " << c.ids[i] << ")";
                    break;
                }
            }
            rep.first_failure = o.str();
        }

        // decode(oracle ids) must reproduce the source text exactly. Checked
        // against the ORACLE's ids, not ours, so a decode bug cannot be masked
        // by an encode bug that happens to invert it.
        const auto& golden = c.ids;
        const bool decoded = tok.decode(golden, round).ok();
        if (decoded && round == c.text) ++rep.decode_ok;

        // Streaming must change WHEN bytes are handed over, never which. Run the
        // same golden ids through the Streamer and rebuild the text from deltas.
        if (!decoded) continue;
        soma::CompiledTokenizer::Streamer st(tok);
        std::string rebuilt, delta;
        bool boundaries_ok = true;
        for (const auto id : golden) {
            if (!st.push(id, delta).ok()) {
                boundaries_ok = false;
                break;
            }
            if (delta.empty()) {
                ++rep.held;
            } else if (!ends_on_codepoint_boundary(delta)) {
                // The whole reason this class exists: a delta that ends mid-
                // codepoint is not text, and a client appending it to a UTF-8
                // buffer gets a replacement character that never goes away.
                boundaries_ok = false;
                if (rep.first_failure.empty()) {
                    rep.first_failure = "delta ends mid-codepoint: \"" + escape(delta) + "\"";
                }
                break;
            }
            rebuilt += delta;
        }
        if (!boundaries_ok) continue;
        (void)st.flush(delta);
        rebuilt += delta;
        if (rebuilt == round) {
            ++rep.stream_ok;
        } else if (rep.first_failure.empty()) {
            rep.first_failure = "streamed text != decode(): \"" + escape(rebuilt) + "\" vs \"" +
                                escape(round) + "\"";
        }
    }

    // ── the synthetic split ──────────────────────────────────────────────────
    //
    // The corpus may or may not contain a codepoint that splits across tokens,
    // and a gate whose interesting case shows up by luck is not a gate. A
    // byte-fallback vocabulary has one token per byte, so the split can be built
    // deliberately: find the ids that decode to the individual bytes of a CJK
    // character, feed exactly those, and require the character back.
    {
        const std::string target = "\xE4\xB8\xAD"; // U+4E2D
        std::vector<soma::TokenId> pieces;
        for (const char want : target) {
            bool found = false;
            for (std::uint32_t id = 0; id < tok.vocab_size() && !found; ++id) {
                std::string one;
                if (!tok.decode(std::span<const soma::TokenId>(&id, 1), one).ok()) continue;
                if (one.size() == 1 && one[0] == want) {
                    pieces.push_back(id);
                    found = true;
                }
            }
            if (!found) break;
        }
        if (pieces.size() == target.size()) {
            soma::CompiledTokenizer::Streamer st(tok);
            std::string rebuilt, delta;
            int emitted = 0;
            for (const auto id : pieces) {
                if (!st.push(id, delta).ok()) break;
                if (!delta.empty()) {
                    ++emitted;
                    rebuilt += delta;
                }
            }
            (void)st.flush(delta);
            rebuilt += delta;
            // Exactly one delta, arriving on the third byte. Two would mean a
            // partial codepoint went out; zero would mean nothing was emitted
            // until flush, which is a stall rather than a stream.
            bool ok = (rebuilt == target) && (emitted == 1);
            if (!ok && rep.first_failure.empty()) {
                rep.first_failure = "split probe: \"" + escape(rebuilt) + "\" in " +
                                    std::to_string(emitted) + " deltas";
            }

            // flush() with something actually held. Nothing else reaches this:
            // the corpus is valid UTF-8, so no case ends mid-codepoint and the
            // flush above always returns empty. A stream that ends on a truncated
            // sequence — a turn that hits max_tokens partway through a character —
            // must still hand those bytes over, or the streamed text and the
            // final text disagree by the tail.
            soma::CompiledTokenizer::Streamer partial(tok);
            std::string d, held;
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                if (!partial.push(pieces[i], d).ok()) break;
                held += d; // must stay empty: neither byte completes the codepoint
            }
            (void)partial.flush(d);
            const bool flush_ok = held.empty() && d == target.substr(0, pieces.size() - 1);
            if (!flush_ok && rep.first_failure.empty()) {
                rep.first_failure = "flush withheld a partial codepoint: got " +
                                    std::to_string(d.size()) + " bytes after " +
                                    std::to_string(held.size()) + " streamed";
            }
            ok = ok && flush_ok;
            rep.split_probe = ok ? 1 : 0;
        }
    }
    return rep;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tokenizers");
    if (!fs::is_directory(root)) {
        std::cerr << "tokenizers directory not found: " << root.string() << "\n";
        return 2;
    }

    std::vector<fs::path> dirs;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i)
            dirs.push_back(root / argv[i]);
    } else {
        for (const auto& e : fs::directory_iterator(root)) {
            if (e.is_directory()) dirs.push_back(e.path());
        }
        std::sort(dirs.begin(), dirs.end());
    }

    int failures = 0, passed = 0, skipped = 0, probed = 0;
    std::cout << std::left << std::setw(30) << "tokenizer" << std::setw(10) << "encode"
              << std::setw(10) << "decode" << std::setw(10) << "stream" << std::setw(8) << "held"
              << std::setw(8) << "split"
              << "detail\n"
              << std::string(100, '-') << "\n";

    for (const auto& dir : dirs) {
        const Report r = run(dir);
        std::cout << std::left << std::setw(30) << dir.filename().string();

        if (r.skipped) {
            ++skipped;
            std::cout << std::setw(10) << "-" << std::setw(10) << "-" << std::setw(10) << "-"
                      << std::setw(8) << "-" << std::setw(8) << "-"
                      << "SKIP: " << r.skip_reason.substr(0, 46) << "\n";
            continue;
        }
        if (!r.loaded) {
            ++failures;
            std::cout << std::setw(10) << "ERR" << std::setw(10) << "-" << std::setw(10) << "-"
                      << std::setw(8) << "-" << std::setw(8) << "-" << r.first_failure << "\n";
            continue;
        }
        const bool ok = (r.encode_ok == r.cases) && (r.decode_ok == r.cases) &&
                        (r.stream_ok == r.cases) && (r.split_probe != 0);
        ok ? ++passed : ++failures;
        if (r.split_probe == 1) ++probed;
        const auto frac = [&](int n) { return std::to_string(n) + "/" + std::to_string(r.cases); };
        std::cout << std::setw(10) << frac(r.encode_ok) << std::setw(10) << frac(r.decode_ok)
                  << std::setw(10) << frac(r.stream_ok) << std::setw(8) << r.held << std::setw(8)
                  << (r.split_probe < 0 ? "n/a" : (r.split_probe == 1 ? "OK" : "FAIL"))
                  << (ok ? "" : r.first_failure) << "\n";
    }

    std::cout << std::string(100, '-') << "\n"
              << passed << " passed, " << failures << " failed, " << skipped << " skipped, "
              << probed << " exercised a split codepoint\n";
    return failures == 0 ? 0 : 1;
}
