// Soma G0 — the compiled chat template against the model's own renderer.
//
// The prompt framing is the one part of serving that is invisible when it is
// wrong. A model handed `user: hi\nassistant:` instead of
// `<|user|>hi<|assistant|><think>` still answers, still streams, still reads
// fluently — and is not the model that was trained. Nothing downstream can catch
// that: the weights are right, the tokenizer round-trips, the KL against a
// reference is computed on whatever prompt was actually built.
//
// So this grades the engine against the ONLY authority there is: the ids HF's
// own Jinja renderer produced for the same conversation, tokenized as one
// string. Assembling those ids out of precompiled pieces and landing on the same
// sequence is a real check; comparing against a second copy of this engine's
// opinion would not be.
//
// Usage: chat_template_g0 <fixtures/tokenizers>

#include "soma/tokenizer.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what;
    if (!ok && !detail.empty()) std::cout << " -- " << detail;
    std::cout << "\n";
    if (!ok) ++failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: chat_template_g0 <fixtures/tokenizers>\n";
        return 2;
    }
    const fs::path root(argv[1]);
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        std::cerr << "no tokenizer fixtures at " << root << "\n";
        return 2;
    }

    std::uint32_t with_template = 0;
    std::uint32_t without_template = 0;

    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;
        const auto name = entry.path().filename().string();
        const auto compiled = entry.path() / "tokenizer.soma";
        const auto oracle_path = entry.path() / "chat_oracle.bin";
        if (!fs::is_regular_file(compiled, ec)) continue;

        soma::CompiledTokenizer tokenizer;
        if (const auto st = tokenizer.open(compiled.string()); !st.ok()) {
            check(false, name + ": open", st.message());
            continue;
        }
        const auto& chat = tokenizer.chat_template();

        // A fixture with no oracle must ALSO report no template, and one with an
        // oracle must report one. Without this pairing the loop would quietly
        // skip every fixture and report a clean run over nothing — which is how
        // a gate ends up structurally present and doing nothing.
        if (!fs::is_regular_file(oracle_path, ec)) {
            check(!chat.present,
                  name + ": no chat oracle, so no compiled template either",
                  chat.present ? "a template was compiled with nothing to grade it" : "");
            ++without_template;
            continue;
        }
        ++with_template;
        check(chat.present, name + ": a chat oracle implies a compiled template");
        if (!chat.present) continue;

        std::vector<soma::ChatOracleCase> cases;
        if (const auto st = soma::read_chat_oracle(oracle_path.string(), cases); !st.ok()) {
            check(false, name + ": read chat oracle", st.message());
            continue;
        }

        soma::RoundTripResult result;
        if (const auto st = soma::verify_chat_template(tokenizer, cases, result); !st.ok()) {
            check(false, name + ": verify", st.message());
            continue;
        }
        check(result.clean(),
              name + ": " + std::to_string(result.encode_ok) + "/" +
                  std::to_string(result.cases) + " conversations match HF's ids",
              result.first_failure);

        // The refusals are part of the contract, not decoration. An engine that
        // accepted an option the template cannot honour would return a prompt
        // the caller believes is something it is not.
        std::vector<soma::ChatMessage> convo{
            soma::ChatMessage{soma::MessageRole::User, "hello", {}}};
        std::vector<soma::TokenId> ids;
        if (!chat.has(soma::chat_flag::kEnableThinking)) {
            soma::ChatOptions options;
            options.enable_thinking = false;
            check(!tokenizer.apply_chat_template(convo, options, ids).ok(),
                  name + ": enable_thinking=false is REFUSED, not ignored");
        }
        if (!chat.has(soma::chat_flag::kClearThinkingSettable)) {
            soma::ChatOptions options;
            options.clear_thinking_set = true;
            options.clear_thinking = true;
            check(!tokenizer.apply_chat_template(convo, options, ids).ok(),
                  name + ": clear_thinking is REFUSED, not ignored");
        }
        if (!chat.has(soma::chat_flag::kReasoningEffort)) {
            soma::ChatOptions options;
            options.reasoning_effort = "high";
            check(!tokenizer.apply_chat_template(convo, options, ids).ok(),
                  name + ": reasoning_effort is REFUSED, not ignored");
        } else {
            // An effort the template does not recognize must behave the way the
            // TEMPLATE behaves — GLM renders `medium` as its default rather than
            // erroring — so refusing it here would be stricter than the model.
            soma::ChatOptions options;
            options.reasoning_effort = "banana";
            std::vector<soma::TokenId> fallback;
            soma::ChatOptions plain;
            const bool ok = tokenizer.apply_chat_template(convo, options, ids).ok() &&
                            tokenizer.apply_chat_template(convo, plain, fallback).ok() &&
                            ids == fallback;
            check(ok, name + ": an unrecognized reasoning_effort falls back to the default");
        }
    }

    std::cout << "\n" << with_template << " fixture(s) with a compiled chat template, "
              << without_template << " without\n";
    if (with_template == 0) {
        check(false, "at least one fixture carries a compiled chat template",
              "every fixture skipped; this gate would pass over nothing");
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
