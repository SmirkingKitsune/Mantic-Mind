// Soma — the admission ladder as `soma conform` runs it.
//
// The gate here is not "does it pass". It is that a stage which DID NOT RUN is
// reported as skipped rather than as a pass. Most of this ladder cannot run on a
// serving host — stages 1 and 3 need a `transformers` oracle for the specific
// model — and a pipeline that recorded those as passing would hand every model a
// verdict that looks validated and is only computed.
//
// Two fixture directories, because they exercise opposite halves:
//
//   containers/  a converted container: quantized formats, dense weights, no
//                compiled tokenizer (the tiny model's vocab is 512 and the
//                committed tokenizer's is 151936 — they are not the same model)
//   tokenizers/  a compiled tokenizer and its HF oracle, no container
//
// Neither is complete, which is the point: each proves one stage runs and the
// other reports why it could not.
//
// Usage: conform_g8 <soma_exe> <container_dir> <tokenizer_dir>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(58) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// Run `soma conform --json` and parse it. Returns a null json on failure.
json conform(const std::string& exe, const std::string& dir, int& out_rc) {
    const auto tmp = fs::temp_directory_path() / "soma_conform_g8.json";
    const std::string cmd =
        "\"\"" + exe + "\" conform --model-dir \"" + dir + "\" --json > \"" + tmp.string() + "\"\"";
    out_rc = std::system(cmd.c_str());
    std::ifstream in(tmp, std::ios::binary);
    if (!in) return {};
    try {
        json j;
        in >> j;
        return j;
    } catch (const std::exception&) {
        return {};
    }
}

/// The one stage by name, or a null json.
json stage_of(const json& doc, const std::string& name) {
    for (const auto& s : doc.value("stages", json::array())) {
        if (s.value("stage", std::string{}) == name) return s;
    }
    return {};
}

std::string status_of(const json& doc, const std::string& name) {
    const auto s = stage_of(doc, name);
    return s.is_null() ? std::string{"<absent>"} : s.value("status", std::string{"<none>"});
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: conform_g8 <soma_exe> <container_dir> <tokenizer_dir>\n";
        return 2;
    }
    const std::string exe = argv[1], container = argv[2], tokenizer = argv[3];

    // ── 1. a converted container ─────────────────────────────────────────────
    std::cout << "\n1. a converted container: the codec runs, the tokenizer cannot\n";
    int rc = 0;
    const auto c = conform(exe, container, rc);
    check(rc == 0, "conform exits 0", "rc=" + std::to_string(rc));
    check(!c.is_null(), "and produces JSON");
    if (c.is_null()) return 1;

    check(status_of(c, "quant_codec") == "passed",
          "quant_codec ran and passed",
          status_of(c, "quant_codec"));

    // The codec check is a PACKING check: measured bits/weight against a formula
    // written independently of quantized_tensor_bytes(). Assert the numbers are
    // real rather than trusting the verdict — a stage reporting `passed` with no
    // weights behind it is the failure this catches.
    {
        const auto detail = stage_of(c, "quant_codec").value("detail", json::object());
        const auto formats = detail.value("formats", json::array());
        check(!formats.empty(),
              "against the container's own declared formats",
              std::to_string(formats.size()) + " formats");
        bool measured = true, plausible = true;
        for (const auto& f : formats) {
            if (f.value("weights", 0ull) == 0 || f.value("tensors", 0ull) == 0) measured = false;
            const double bpw = f.value("bits_per_weight", 0.0);
            const double want = f.value("bits_per_weight_expected", 0.0);
            const double rel = f.value("rel_rms", 1.0);
            // Not a re-derivation of the tool's own check — a sanity band. A
            // packing bug lands at rel_rms near 1.0, two orders of magnitude out.
            if (bpw <= 0.0 || want <= 0.0 || rel <= 0.0 || rel >= 0.5) plausible = false;
        }
        check(measured, "with a non-zero weight and tensor count");
        check(plausible, "and rel_rms far below the packing-bug regime");
    }

    check(status_of(c, "tokenizer_roundtrip") == "skipped",
          "tokenizer_roundtrip is SKIPPED, not passed",
          status_of(c, "tokenizer_roundtrip"));
    {
        const auto d = stage_of(c, "tokenizer_roundtrip").value("detail", json::object());
        check(!d.value("reason", std::string{}).empty(),
              "and says why",
              d.value("reason", std::string{}));
    }

    // ── 2. the stages that need a transformers oracle ────────────────────────
    //
    // These are the ones that would be most tempting to fake, because the ladder
    // reads as incomplete without them. They are skipped, and each says what it
    // would need — a reader can act on that; a bare `false` sends them here.
    std::cout << "\n2. the stages this host cannot run\n";
    for (const char* name : {"fp32_tiny_tf", "real_logit_kl", "accuracy_floor"}) {
        check(
            status_of(c, name) == "skipped", std::string(name) + " is skipped", status_of(c, name));
        const auto d = stage_of(c, name).value("detail", json::object());
        const auto reason = d.value("reason", std::string{});
        check(!reason.empty(), std::string("  and names what it needs"), reason.substr(0, 44));
    }

    // ── 3. a compiled tokenizer, no container ────────────────────────────────
    std::cout << "\n3. a compiled tokenizer with no container: the other half\n";
    const auto t = conform(exe, tokenizer, rc);
    check(rc == 0 && !t.is_null(), "conform exits 0 and produces JSON");
    if (!t.is_null()) {
        check(status_of(t, "tokenizer_roundtrip") == "passed",
              "tokenizer_roundtrip ran and passed",
              status_of(t, "tokenizer_roundtrip"));
        const auto d = stage_of(t, "tokenizer_roundtrip").value("detail", json::object());
        const auto cases = d.value("cases", 0u);
        check(cases > 0, "over a non-empty corpus", std::to_string(cases) + " cases");
        check(d.value("encode_ok", 0u) == cases && d.value("decode_ok", 0u) == cases,
              "with every case matching HF byte-for-byte");
        check(status_of(t, "quant_codec") == "skipped",
              "and quant_codec is skipped — there is no container",
              status_of(t, "quant_codec"));
    }

    // ── 4. nothing there at all ──────────────────────────────────────────────
    //
    // Every stage skipped means the ladder did not run, and `passed` must be
    // false. "Nothing was checked" reported as a pass is the whole failure mode
    // this file exists to prevent.
    std::cout << "\n4. an empty directory\n";
    const auto empty_dir = fs::temp_directory_path() / "soma_conform_g8_empty";
    fs::create_directories(empty_dir);
    const auto e = conform(exe, empty_dir.string(), rc);
    check(!e.is_null(), "conform still produces JSON");
    if (!e.is_null()) {
        check(e.value("ran", true) == false, "and reports that nothing ran");
        check(e.value("passed", true) == false,
              "so `passed` is FALSE — nothing checked is not a pass");
        bool all_skipped = true;
        for (const auto& s : e.value("stages", json::array())) {
            if (s.value("status", std::string{}) != "skipped") all_skipped = false;
        }
        check(all_skipped, "with every stage skipped");
    }
    std::error_code ec;
    fs::remove_all(empty_dir, ec);

    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
