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

struct Case {
    std::string text;
    std::vector<std::uint32_t> ids;
};

bool read_oracle(const fs::path& p, std::vector<Case>& out, std::string& err) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { err = "cannot open " + p.string(); return false; }
    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMATORC", 8) != 0) { err = "bad magic"; return false; }

    auto u32 = [&]() -> std::uint32_t {
        std::uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    if (u32() != 1) { err = "unsupported oracle version"; return false; }
    const auto n = u32();
    out.resize(n);
    for (auto& c : out) {
        const auto len = u32();
        c.text.resize(len);
        if (len > 0) in.read(c.text.data(), len);
        const auto k = u32();
        c.ids.resize(k);
        for (auto& id : c.ids) id = u32();
    }
    if (!in) { err = "short read"; return false; }
    return true;
}

std::string escape(const std::string& s, std::size_t cap = 40) {
    std::ostringstream o;
    for (std::size_t i = 0; i < s.size() && o.str().size() < cap; ++i) {
        const char c = s[i];
        if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (static_cast<unsigned char>(c) < 0x20) o << '?';
        else o << c;
    }
    return o.str();
}

struct Report {
    bool loaded = false;
    bool skipped = false;
    std::string skip_reason;
    int cases = 0, encode_ok = 0, decode_ok = 0;
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

    std::vector<Case> cases;
    std::string err;
    if (!read_oracle(dir / "tokenizer_oracle.bin", cases, err)) {
        rep.first_failure = err;
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
        const bool match =
            ids.size() == c.ids.size() &&
            std::equal(ids.begin(), ids.end(), c.ids.begin(),
                       [](soma::TokenId a, std::uint32_t b) { return a == b; });
        if (match) {
            ++rep.encode_ok;
        } else if (rep.first_failure.empty()) {
            std::ostringstream o;
            o << "\"" << escape(c.text) << "\" got " << ids.size() << " ids, want "
              << c.ids.size();
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
        std::vector<soma::TokenId> golden(c.ids.begin(), c.ids.end());
        if (tok.decode(golden, round).ok() && round == c.text) ++rep.decode_ok;
    }
    return rep;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tokenizers");
    if (!fs::is_directory(root)) {
        std::cerr << "tokenizers directory not found: " << root.string() << "\n";
        return 2;
    }

    std::vector<fs::path> dirs;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) dirs.push_back(root / argv[i]);
    } else {
        for (const auto& e : fs::directory_iterator(root)) {
            if (e.is_directory()) dirs.push_back(e.path());
        }
        std::sort(dirs.begin(), dirs.end());
    }

    int failures = 0, passed = 0, skipped = 0;
    std::cout << std::left << std::setw(34) << "tokenizer" << std::setw(12) << "encode"
              << std::setw(12) << "decode" << "detail\n"
              << std::string(96, '-') << "\n";

    for (const auto& dir : dirs) {
        const Report r = run(dir);
        std::cout << std::left << std::setw(34) << dir.filename().string();

        if (r.skipped) {
            ++skipped;
            std::cout << std::setw(12) << "-" << std::setw(12) << "-"
                      << "SKIP: " << r.skip_reason.substr(0, 58) << "\n";
            continue;
        }
        if (!r.loaded) {
            ++failures;
            std::cout << std::setw(12) << "ERR" << std::setw(12) << "-" << r.first_failure << "\n";
            continue;
        }
        const bool ok = (r.encode_ok == r.cases) && (r.decode_ok == r.cases);
        ok ? ++passed : ++failures;
        std::cout << std::setw(12) << (std::to_string(r.encode_ok) + "/" + std::to_string(r.cases))
                  << std::setw(12) << (std::to_string(r.decode_ok) + "/" + std::to_string(r.cases))
                  << (ok ? "" : r.first_failure) << "\n";
    }

    std::cout << std::string(96, '-') << "\n"
              << passed << " passed, " << failures << " failed, " << skipped << " skipped\n";
    return failures == 0 ? 0 : 1;
}
