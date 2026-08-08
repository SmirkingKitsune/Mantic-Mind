// Mantic-Mind — G5: the node's side of the subprocess boundary.
//
// Runs against the REAL soma executable, not a stub. A supervisor tested against
// a mock proves the mock; the failure modes that matter here — a child that
// exits during startup, a child that dies after reporting ready — only exist
// with a real process.
//
// The crash watchdog is the point. Today a dead engine stays SlotState::Ready
// until a request happens to fail, so the node advertises capacity it does not
// have and the scheduler keeps placing work on it.
//
// Usage: engine_g5 <soma_exe> <model_dir>

#include "common/engine_client.hpp"
#include "node/engine_descriptor.hpp"
#include "node/engine_process.hpp"
#include "node/engine_supervisor.hpp"
#include "node/kv_checkpoint_backend.hpp"
#include "soma/kv_checkpoint.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(64) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

std::string soma_kv_extension() {
    return soma::kv_checkpoint_extension();
}

/// One chat turn against a live engine. Returns the assistant's reply.
///
/// `messages` is the FULL transcript, the way a real client sends it — that is
/// what makes each turn an extension of the last rather than a new conversation
/// that happens to share a key.
std::string chat(std::uint16_t port,
                 const json& messages,
                 const std::string& conversation,
                 std::uint32_t max_tokens) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(60);
    const json body{
        {"messages", messages}, {"conversation", conversation}, {"max_tokens", max_tokens}};
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    if (!res || res->status != 200) {
        std::cout << "   (chat failed: " << (res ? std::to_string(res->status) : "no response")
                  << (res ? " " + res->body : "") << ")\n";
        return {};
    }
    try {
        return json::parse(res->body)["choices"][0]["message"].value("content", std::string{});
    } catch (const std::exception&) {
        return {};
    }
}

json user_msg(const std::string& text) {
    return json{{"role", "user"}, {"content", text}};
}

/// A greedy turn, non-streaming. `temperature: 0` is argmax and consumes no RNG,
/// which is what makes the streamed and non-streamed forms of the same prompt
/// comparable at all.
std::string chat_greedy(std::uint16_t port, const json& messages, std::uint32_t max_tokens) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(60);
    const json body{{"messages", messages}, {"max_tokens", max_tokens}, {"temperature", 0.0f}};
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    if (!res || res->status != 200) return {};
    try {
        return json::parse(res->body)["choices"][0]["message"].value("content", std::string{});
    } catch (const std::exception&) {
        return {};
    }
}

/// The same turn over SSE, returning every `delta.content` in order.
///
/// Kept separate rather than folded into chat(): the point is the SEQUENCE of
/// deltas, not the text they add up to, and a helper that concatenated them
/// would discard the thing under test.
std::vector<std::string>
chat_stream_deltas(std::uint16_t port, const json& messages, std::uint32_t max_tokens) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(60);
    const json body{{"messages", messages},
                    {"max_tokens", max_tokens},
                    {"temperature", 0.0f},
                    {"stream", true}};

    std::vector<std::string> deltas;
    std::string buf;
    auto res = cli.Post(
        "/v1/chat/completions",
        httplib::Headers{},
        body.dump(),
        "application/json",
        [&](const char* data, std::size_t len) {
            buf.append(data, len);
            for (std::size_t p; (p = buf.find("\n\n")) != std::string::npos;) {
                const std::string frame = buf.substr(0, p);
                buf.erase(0, p + 2);
                if (frame.rfind("data: ", 0) != 0) continue;
                const std::string payload = frame.substr(6);
                if (payload == "[DONE]") continue;
                try {
                    const auto j = json::parse(payload);
                    if (j.contains("choices")) {
                        deltas.push_back(j["choices"][0]["delta"].value("content", std::string{}));
                    }
                } catch (const std::exception&) {
                }
            }
            return true;
        });
    if (!res || res->status != 200) return {};
    return deltas;
}

/// Does `s` end on a codepoint boundary? Malformed bytes count as latin-1, which
/// is how the tokenizer treats them.
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
            if (i + len > s.size()) return false;
            for (std::size_t k = 1; k < len; ++k) {
                if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) {
                    len = 1;
                    break;
                }
            }
        }
        i += len;
    }
    return true;
}

/// GET /internal/sessions — live per-sequence state, straight from the engine.
json sessions(std::uint16_t port) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(10);
    auto res = cli.Get("/internal/sessions");
    if (!res || res->status != 200) return json::object();
    try {
        return json::parse(res->body);
    } catch (const std::exception&) {
        return json::object();
    }
}

/// A DELIBERATELY independent encoder for the checkpoint header.
///
/// Writing these bytes by hand is the point: the header crosses a process
/// boundary, so the decoder deserves a check that does not share its code. If
/// the engine changes the layout, this stops matching — which is the failure a
/// shared helper would hide.
void write_soma_header(const std::string& path,
                       const std::string& arch_hash,
                       std::uint32_t format_id,
                       std::uint32_t length_tokens,
                       std::uint32_t version = soma::kKvCheckpointVersion) {
    std::vector<unsigned char> b;
    const auto u32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i)
            b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFF));
    };
    const auto u64 = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i)
            b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFF));
    };

    for (const char c : std::string("SOMAKV01"))
        b.push_back(static_cast<unsigned char>(c));
    u32(version);
    u32(static_cast<std::uint32_t>(arch_hash.size())); // arch_hash length
    for (const char c : arch_hash)
        b.push_back(static_cast<unsigned char>(c));
    u32(format_id);
    u32(length_tokens);
    u32(64);            // d_model
    u64(4096);          // payload_bytes
    u64(1700000000000); // written_at_ms
    // v2: the token ids. Not written here — stat reads a bounded prefix and
    // computes the offsets arithmetically, which is the property being checked.

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    // A header with no payload is enough: stat() reads a bounded prefix and never
    // touches the payload, which is the property being checked.
    const std::vector<char> pad(64, 0);
    f.write(pad.data(), static_cast<std::streamsize>(pad.size()));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: engine_g5 <soma_exe> <model_dir> [container_dir]"
                     " [llama_server_exe] [gguf]\n";
        return 2;
    }
    const std::string exe = argv[1];
    const std::string model = argv[2];
    // Optional. A converted container is what makes the STREAMING path real: a
    // plain checkpoint is resident, so its expert cache and heat grid are empty
    // by construction and every counter reads zero.
    const std::string container = argc > 3 ? argv[3] : std::string{};

    // The fallback engine, for §12. The GGUF is committed; the llama-server
    // binary is not, because it is provisioned per host — so CMake passes
    // whatever MM_LLAMA_SERVER points at and §12 skips loudly when it is unset.
    // Resolved here rather than in §12 so a path that was given but does not
    // exist is reported as a bad path, not silently downgraded to "not
    // configured".
    std::string llama_exe = argc > 4 ? argv[4] : std::string{};
    std::string gguf = argc > 5 ? argv[5] : std::string{};
    {
        std::error_code ec;
        // Both are checked, and the GGUF especially: it is committed, so a
        // missing one means .gitignore ate it (`*.gguf` nearly did) rather than
        // "not configured on this host". Those want different reactions.
        if (!llama_exe.empty() && !std::filesystem::is_regular_file(llama_exe, ec)) {
            std::cout << "   note: llama-server not found at " << llama_exe << "\n";
            llama_exe.clear();
        }
        if (!gguf.empty() && !std::filesystem::is_regular_file(gguf, ec)) {
            std::cout << "   note: the committed GGUF fixture is MISSING at " << gguf
                      << "\n          (build it with tools/testing/make_tiny_gguf.py)\n";
            gguf.clear();
        }
    }
    int skipped = 0;

    // ── 1. the registry ──────────────────────────────────────────────────────
    std::cout << "1. engines are DATA, not code paths\n";
    {
        auto& reg = mm::EngineRegistry::instance();
        reg.register_engine(mm::make_soma_descriptor(exe));
        reg.register_engine(mm::make_llama_descriptor("llama-server"));

        check(reg.find("soma") != nullptr, "soma is registered");
        check(reg.find("llama-cpp") != nullptr, "llama-cpp is registered");
        check(reg.find("nonesuch") == nullptr, "an unknown id is not");
        check(reg.ids().size() == 2,
              "the registry can enumerate itself",
              "so a 400 body lists real contents, not a literal");

        // The capability that actually differs at this layer, and the reason the
        // scheduler may co-locate agents on one engine but not the other.
        const auto* s = reg.find("soma");
        const auto* l = reg.find("llama-cpp");
        check(s->supports_multi_seq && !l->supports_multi_seq,
              "only Soma advertises real per-sequence state");

        // build_launch is pure, so the argv is testable without spawning.
        mm::EngineLoadRequest req;
        req.model_path = model;
        req.port = 8123;
        const auto spec = s->build_launch(req);
        bool has_serve = false, has_port = false;
        for (const auto& a : spec.args) {
            has_serve |= (a == "serve");
            has_port |= (a == "8123");
        }
        check(has_serve && has_port && spec.executable == exe,
              "build_launch emits a runnable argv");
        check(spec.readiness.kind == mm::ReadinessProbe::Kind::HttpHealth,
              "readiness is an HTTP probe, not a stdout sentinel");

        // The FALLBACK's argv, which nothing asserted — and that is exactly how
        // it came to drop nine settings while its own documentation said it
        // wrapped build_llama_server_args() (roadmap D14). An engine started
        // through EngineSupervisor ran on llama.cpp's defaults regardless of what
        // the operator configured.
        //
        // Every value below is deliberately distinctive, so a match cannot be a
        // coincidence of some other flag carrying the same number.
        mm::EngineLoadRequest lreq;
        lreq.model_path = "/models/m.gguf";
        lreq.port = 8124;
        lreq.settings.ctx_size = 1536;
        lreq.settings.n_gpu_layers = 17;
        lreq.settings.n_threads = 5;
        lreq.settings.batch_size = 384;
        lreq.settings.ubatch_size = 96;
        lreq.settings.parallel = 3;
        lreq.settings.extra_args = {"--no-warmup"};
        const auto lspec = l->build_launch(lreq);

        const auto argv_has = [&](const std::string& flag, const std::string& value) {
            for (std::size_t i = 0; i + 1 < lspec.args.size(); ++i) {
                if (lspec.args[i] == flag && lspec.args[i + 1] == value) return true;
            }
            return false;
        };
        check(argv_has("--gpu-layers", "17"), "n_gpu_layers reaches the process");
        check(argv_has("--threads", "5"), "n_threads reaches the process");
        check(argv_has("--batch-size", "384"), "batch_size reaches the process");
        check(argv_has("--ubatch-size", "96"), "ubatch_size reaches the process");
        check(argv_has("--parallel", "3"), "parallel reaches the process");
        // llama-server hosts ONE shared context of ctx_size*parallel, so the
        // value on the wire is not the value in the settings. Asserted at the
        // product rather than at 1536, because a builder that passed the raw
        // ctx_size would under-provision every slot but three.
        check(argv_has("--ctx-size", "4608"), "ctx_size is scaled by parallel, not passed raw");
        check(std::find(lspec.args.begin(), lspec.args.end(), "--no-warmup") != lspec.args.end(),
              "operator extra_args survive");
    }

    // ── 2. launch, become ready, stop cleanly ────────────────────────────────
    std::cout << "\n2. launch -> ready -> clean stop\n";
    {
        mm::EngineProcess proc;
        std::atomic<int> crashes{0};
        proc.set_crash_callback([&](int, const std::string&) { ++crashes; });

        mm::EngineLoadRequest req;
        req.model_path = model;
        req.port = 8124;
        auto spec = mm::EngineRegistry::instance().find("soma")->build_launch(req);
        spec.readiness.timeout_seconds = 60;

        const auto t0 = std::chrono::steady_clock::now();
        const bool started = proc.start(spec);
        const auto secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        check(started, "start() blocks until the health probe answers", std::to_string(secs) + "s");
        if (!started) {
            std::cout << "   last_error: " << proc.last_error() << "\n";
            return 1;
        }
        check(proc.state() == mm::ProcessState::Ready, "state is Ready");
        check(proc.url() == "http://127.0.0.1:8124", "url()", proc.url());

        proc.stop();
        check(proc.state() == mm::ProcessState::Stopped, "state is Stopped after stop()");
        // A deliberate stop must NOT look like a crash, or the signal is
        // worthless — every shutdown would fire it.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        check(crashes.load() == 0,
              "a clean stop does not fire the crash callback",
              std::to_string(crashes.load()) + " crashes");
    }

    // ── 3. THE WATCHDOG ──────────────────────────────────────────────────────
    //
    // Kill the child behind the supervisor's back and require it to notice.
    std::cout << "\n3. an engine that dies is NOTICED\n";
    {
        mm::EngineProcess proc;
        std::atomic<int> crashes{0};
        std::string detail;
        proc.set_crash_callback([&](int code, const std::string& d) {
            ++crashes;
            detail = d + " (code " + std::to_string(code) + ")";
        });

        mm::EngineLoadRequest req;
        req.model_path = model;
        req.port = 8125;
        auto spec = mm::EngineRegistry::instance().find("soma")->build_launch(req);
        spec.readiness.timeout_seconds = 60;

        if (!proc.start(spec)) {
            check(false, "engine started for the crash test", proc.last_error());
            return 1;
        }
        check(proc.state() == mm::ProcessState::Ready, "engine is Ready before the kill");

        // By pid, never by image name. `taskkill /IM soma.exe` would also kill a
        // developer's own running engine, and a test that reaches outside its own
        // process tree is a test nobody can run on a working machine.
        const std::uint32_t victim = proc.pid();
        check(victim != 0, "pid() identifies the child", std::to_string(victim));
#if defined(_WIN32)
        const std::string cmd = "taskkill /F /PID " + std::to_string(victim) + " >nul 2>&1";
#else
        const std::string cmd = "kill -9 " + std::to_string(victim) + " >/dev/null 2>&1";
#endif
        (void)std::system(cmd.c_str());

        // Bounded wait: a watchdog that eventually notices is not a watchdog.
        bool noticed = false;
        for (int i = 0; i < 50 && !noticed; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            noticed = (crashes.load() > 0);
        }
        check(noticed, "the crash callback fired within 5s", detail);
        check(proc.state() == mm::ProcessState::Crashed,
              "state is Crashed, not Ready",
              std::string("state=") +
                  (proc.state() == mm::ProcessState::Crashed ? "Crashed" : "other"));
        proc.stop();
    }

    // ── 4. a child that never becomes ready ──────────────────────────────────
    std::cout << "\n4. a start-up failure fails fast\n";
    {
        mm::EngineProcess proc;
        mm::EngineLoadRequest req;
        req.model_path = "Z:/definitely/not/a/model";
        req.port = 8126;
        auto spec = mm::EngineRegistry::instance().find("soma")->build_launch(req);
        // Generous budget on purpose: the point is that it returns in about the
        // time the CHILD takes to die, not the time the budget allows. An
        // implementation that only checks liveness at the end would sit here for
        // 30 seconds and still "pass" a boolean check.
        spec.readiness.timeout_seconds = 30;

        const auto t0 = std::chrono::steady_clock::now();
        const bool started = proc.start(spec);
        const auto secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        check(!started, "start() returns false for a bad model");
        check(proc.state() == mm::ProcessState::Error, "state is Error");
        check(secs < 15.0,
              "and fails in child-exit time, not budget time",
              std::to_string(secs) + "s of a 30s budget");
        check(!proc.last_error().empty(), "last_error explains it", proc.last_error());
    }

    // ── 5. the supervisor ────────────────────────────────────────────────────
    std::cout << "\n5. EngineSupervisor: sharing, leases, unknown ids\n";
    {
        mm::EngineSupervisor sup(8200, 8210, /*max_slots=*/2);

        mm::EngineLoadRequest req;
        req.model_path = model;
        req.settings.ctx_size = 4096;
        req.settings.n_threads = 4;

        // An unknown engine id must not reach a hardcoded literal.
        const auto bad = sup.load("vllm", req, "agent-x");
        check(bad.empty(), "an unknown engine id fails");
        const std::string why = sup.last_error();
        check(why.find("soma") != std::string::npos && why.find("llama-cpp") != std::string::npos,
              "and the error lists the REGISTERED engines",
              why);

        const auto slot_a = sup.load("soma", req, "agent-a");
        check(!slot_a.empty(), "load() starts an engine", slot_a);
        if (slot_a.empty()) {
            std::cout << "   last_error: " << sup.last_error() << "\n";
            return 1;
        }

        // Same model, different ctx_size. llama.cpp would need a second process;
        // Soma's KV slot is per-sequence, so this must ATTACH.
        mm::EngineLoadRequest req_b = req;
        req_b.settings.ctx_size = 8192;
        const auto slot_b = sup.load("soma", req_b, "agent-b");
        check(slot_b == slot_a, "a second agent shares the engine despite a different ctx_size");
        check(sup.slots().size() == 1, "so there is still exactly one process");
        check(sup.available_slot_count() == 1, "and one slot free of two");

        const auto info = sup.find(slot_a);
        check(info.has_value() && info->backend == "soma",
              "SlotInfo::backend comes from the descriptor, not a literal",
              info ? info->backend : std::string("(none)"));
        check(info && info->agent_ids.size() == 2, "both agents are attached");
        check(sup.find_by_agent("agent-b").value_or("") == slot_a, "find_by_agent resolves");

        // A leased engine is busy and must not be evicted out from under a
        // request that is already streaming.
        {
            auto lease = sup.acquire(slot_a);
            check(static_cast<bool>(lease), "acquire() yields a client");
            check(lease.get() != nullptr && lease.get()->health_check(),
                  "and that client can reach the engine");
            const auto blocked = sup.unload(slot_a);
            check(blocked.status == mm::EngineOpStatus::Busy,
                  "unload() refuses while a lease is live",
                  blocked.message);
        }
        // Lease destroyed -> the borrow is released.
        const auto detached = sup.detach_agent(slot_a, "agent-a");
        check(detached.ok() && detached.remaining_agents == 1,
              "detaching one agent leaves the engine up",
              std::to_string(detached.remaining_agents) + " remaining");

        const auto last = sup.detach_agent(slot_a, "agent-b");
        check(last.ok() && last.unloaded, "detaching the last agent unloads it");
        check(sup.slots().empty(), "the pool is empty");
        check(sup.available_slot_count() == 2, "and both slots are free again");
    }

    // ── 6. a crash reaches the supervisor ────────────────────────────────────
    //
    // The gap named in engine_process.hpp: without this, a crashed engine is only
    // discovered when a request fails, and the slot advertises Ready throughout.
    std::cout << "\n6. a crashed engine stops advertising Ready\n";
    {
        mm::EngineSupervisor sup(8220, 8230, /*max_slots=*/2);
        mm::EngineLoadRequest req;
        req.model_path = model;

        const auto slot = sup.load("soma", req, "agent-c");
        check(!slot.empty(), "engine loaded", slot);
        if (slot.empty()) {
            std::cout << "   last_error: " << sup.last_error() << "\n";
            return 1;
        }
        const auto before = sup.find(slot);
        check(before && before->state == mm::SlotState::Ready, "state is Ready");

        const auto ports = sup.slots();
        std::uint16_t port = ports.empty() ? 0 : ports.front().port;
        check(port != 0, "engine has a port", std::to_string(port));

        // Kill it the way the OS would: find the child by the port the supervisor
        // assigned, since the supervisor owns the EngineProcess and the test does
        // not.
#if defined(_WIN32)
        const std::string cmd =
            "for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :" + std::to_string(port) +
            " ^| findstr LISTENING') do @taskkill /F /PID %a >nul 2>&1";
        (void)std::system(("cmd /c \"" + cmd + "\"").c_str());
#else
        const std::string cmd =
            "kill -9 $(lsof -ti tcp:" + std::to_string(port) + ") >/dev/null 2>&1";
        (void)std::system(cmd.c_str());
#endif

        bool noticed = false;
        for (int i = 0; i < 60 && !noticed; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const auto now = sup.find(slot);
            noticed = now && now->state == mm::SlotState::Error;
        }
        check(noticed, "the supervisor moved the slot to Error within 6s");
        // Error, not gone: control has to see that this placement died. A removed
        // record reads as an engine that was never there.
        check(sup.find(slot).has_value(), "the slot record survives, so the crash is visible");
        check(!static_cast<bool>(sup.acquire(slot)),
              "and acquire() refuses it, so no further work is placed on a dead engine");
        (void)sup.unload_all(true);
    }

    // ── 7. KV checkpoint backends ────────────────────────────────────────────
    std::cout << "\n7. KV checkpoints: the format is owned in ONE place\n";
    {
        mm::LlamaKvBackend llama;
        mm::SomaKvBackend soma;

        check(std::string(llama.file_extension()) != std::string(soma.file_extension()),
              "each backend owns its own on-disk extension",
              std::string(llama.file_extension()) + " vs " + soma.file_extension());
        // Deliberately NOT the per-session KV extension. Soma's suspend artifact
        // is a manifest naming every session in the engine, which is what
        // supports_multi_sequence() has to mean if it is to mean anything.
        check(std::string(soma.file_extension()) != soma_kv_extension(),
              "and Soma's engine-level artifact is not one session's KV file",
              std::string(soma.file_extension()) + " vs " + soma_kv_extension());

        // The bug this exists to stop repeating: POST /slots/0?action=save
        // hardcodes sequence 0, so a --parallel > 1 slot silently checkpoints one
        // sequence and drops the rest.
        check(!llama.supports_multi_sequence(), "llama.cpp reports it cannot do multi-sequence");
        check(soma.supports_multi_sequence(), "Soma reports it can");

        std::string err;
        check(!llama.save("http://127.0.0.1:1", {}, /*sequence=*/2, "x.kvbin", err),
              "llama.cpp REFUSES a non-zero sequence rather than saving seq 0",
              err);

        const auto dir = fs::temp_directory_path() / "soma_kv_g5";
        fs::remove_all(dir);
        fs::create_directories(dir);

        // ── stat on a real header ────────────────────────────────────────────
        // The bytes below are written by hand on purpose. This is a WIRE FORMAT
        // between two binaries, so an independent encoder checking the shared
        // decoder is the point — if the engine changes the layout, this fails,
        // which is correct. The other half of the chain (the engine's writer
        // against this same decoder) is covered by soma_checkpoint_g3.
        const auto good = (dir / ("slot-1" + std::string(soma.file_extension()))).string();
        write_soma_header(good, "arch-hash-abc", /*format_id=*/7, /*length_tokens=*/128);

        mm::KvCheckpointInfo info;
        err.clear(); // these only WRITE out_error on failure, so a stale one lies
        const bool ok = soma.stat_sequence(good, info, err);
        check(ok, "stat_sequence() reads a Soma checkpoint header", err);
        check(info.arch_hash == "arch-hash-abc", "arch_hash round-trips", info.arch_hash);
        check(info.format_id == 7, "format_id round-trips", std::to_string(info.format_id));
        check(info.length_tokens == 128,
              "length_tokens round-trips",
              std::to_string(info.length_tokens));
        check(info.bytes > 0, "and the size is real", std::to_string(info.bytes));

        // v1 files are REFUSED, not reinterpreted. v2 inserts the token array
        // between the header and the payload, so reading a v1 layout as v2 puts
        // every offset out by 4 x length_tokens — silently.
        const auto v1 = (dir / ("old-v1" + std::string(soma_kv_extension()))).string();
        write_soma_header(v1, "arch-hash-abc", 7, 128, /*version=*/1);
        mm::KvCheckpointInfo old_info;
        err.clear();
        check(!soma.stat_sequence(v1, old_info, err),
              "a v1 checkpoint is refused rather than misread",
              err);

        // arch_hash is REPORTED, not compared, so a caller can reject a
        // cross-architecture resume before an engine is ever spawned. That is the
        // whole reason stat() reads the file rather than asking a running engine.
        check(info.arch_hash != "arch-hash-xyz",
              "a mismatched arch is visible to the caller pre-spawn");

        // ── refusals ─────────────────────────────────────────────────────────
        mm::KvCheckpointInfo junk;
        check(!soma.stat_sequence((dir / "missing.somakv").string(), junk, err),
              "stat fails on a missing file",
              err);

        const auto empty = (dir / "empty.somakv").string();
        { std::ofstream(empty, std::ios::binary); }
        check(!soma.stat_sequence(empty, junk, err),
              "a zero-byte checkpoint is refused, not read",
              err);

        const auto garbage = (dir / "garbage.somakv").string();
        {
            std::ofstream f(garbage, std::ios::binary);
            f << "this is not a checkpoint at all, not even close";
        }
        check(
            !soma.stat_sequence(garbage, junk, err), "and so is a file with the wrong magic", err);

        const auto truncated = (dir / "truncated.somakv").string();
        {
            std::ofstream f(truncated, std::ios::binary);
            f << "SOMAKV01\x01";
        }
        check(!soma.stat_sequence(truncated, junk, err), "and a truncated header", err);

        // llama.cpp's session blob is versioned by llama.cpp and validated on its
        // own restore, so stat() claims only what it can check.
        mm::KvCheckpointInfo linfo;
        check(llama.stat(good, linfo, err) && linfo.arch_hash.empty(),
              "llama.cpp's stat() reports size only, and leaves arch_hash empty");

        // ── remove ───────────────────────────────────────────────────────────
        err.clear();
        check(soma.remove(good, err), "remove() deletes the checkpoint", err);
        check(!fs::exists(good), "the file is gone");
        err.clear();
        check(soma.remove(good, err), "and remove() is idempotent", err);

        fs::remove_all(dir);
    }

    // ── 8. sessions outlive their request ────────────────────────────────────
    //
    // The point of the whole increment. Before this, `soma serve` created a
    // sequence per request and finished it with the response: every turn
    // re-prefilled the entire conversation, and at suspend time there was no live
    // KV for the node to checkpoint at all.
    std::cout << "\n8. a sequence outlives the request that created it\n";
    {
        const auto kvdir = (fs::temp_directory_path() / "soma_kv_g5_sess").string();
        fs::remove_all(kvdir);

        mm::EngineSupervisor sup(8260, 8270, /*max_slots=*/2);
        sup.set_kv_checkpoint_dir(kvdir);

        mm::EngineLoadRequest req;
        req.model_path = model;
        const auto slot = sup.load("soma", req, "agent-e");
        check(!slot.empty(), "engine loaded", slot);
        if (slot.empty()) {
            std::cout << "   last_error: " << sup.last_error() << "\n";
            return 1;
        }
        const auto port = sup.slots().front().port;

        // max_tokens = 1 on the opening turn, so the cache holds exactly the
        // prompt: the single generated token is sampled from the last prefill row
        // and never fed back. That makes turn 2's prompt a genuine extension
        // without needing a tokenizer whose decode-then-encode round-trips, which
        // the tiny fixtures have no way to provide.
        const std::string turn1 = "the quick brown fox";
        const auto reply1 = chat(port, json::array({user_msg(turn1)}), "conv-A", 1);
        auto s1 = sessions(port);
        check(s1.value("sessions", json::array()).size() == 1,
              "one turn creates one session",
              s1.value("sessions", json::array()).dump());
        const auto seq1 = s1["sessions"][0].value("sequence", 0u);
        const auto kv1 = s1["sessions"][0].value("kv_tokens", 0u);
        check(kv1 >= turn1.size(),
              "and its KV holds the prompt",
              std::to_string(kv1) + " tokens for " + std::to_string(turn1.size()) + " bytes");

        // The full transcript, assistant turn included — which is what makes the
        // cached prefix a prefix. Send only the new user text and the engine sees
        // a prompt whose middle has changed, correctly cold-starts, and the
        // session buys nothing.
        const json transcript = json::array({user_msg(turn1),
                                             json{{"role", "assistant"}, {"content", reply1}},
                                             user_msg("jumped over")});
        chat(port, transcript, "conv-A", 1);
        auto s2 = sessions(port);
        const auto seq2 = s2["sessions"][0].value("sequence", 0u);
        const auto kv2 = s2["sessions"][0].value("kv_tokens", 0u);
        check(seq2 == seq1,
              "the second turn REUSES the same sequence",
              "seq " + std::to_string(seq1) + " -> " + std::to_string(seq2));
        check(kv2 > kv1,
              "and extends its cache rather than rebuilding it",
              std::to_string(kv1) + " -> " + std::to_string(kv2) + " tokens");

        // A different key is a different conversation, not a continuation.
        chat(port, json::array({user_msg("something else entirely")}), "conv-B", 1);
        auto s3 = sessions(port);
        check(s3.value("sessions", json::array()).size() == 2,
              "a different conversation key gets its own sequence");

        // And a prompt that is NOT an extension must cold-start rather than
        // attach a cache describing a conversation that no longer exists.
        chat(port, json::array({user_msg("a completely different opening")}), "conv-A", 1);
        auto s4 = sessions(port);
        std::uint32_t conv_a_kv = 0, conv_a_seq = 0;
        for (const auto& s : s4["sessions"]) {
            if (s.value("conversation", std::string{}) == "conv-A") {
                conv_a_kv = s.value("kv_tokens", 0u);
                conv_a_seq = s.value("sequence", 0u);
            }
        }
        check(conv_a_seq != seq1,
              "an edited prompt retires the stale sequence",
              "seq " + std::to_string(seq1) + " -> " + std::to_string(conv_a_seq));
        check(conv_a_kv < kv2,
              "and starts cold instead of attaching a wrong cache",
              std::to_string(kv2) + " -> " + std::to_string(conv_a_kv) + " tokens");

        // The supervisor reports per-sequence state now, asked of the engine
        // rather than synthesised from an agent count.
        const auto seqs = sup.sequences(slot);
        check(seqs.size() == 2,
              "the supervisor reports per-sequence state",
              std::to_string(seqs.size()) + " sequences");
        check(!seqs.empty() && seqs[0].kv_tokens > 0,
              "with real KV depths, not a request count",
              seqs.empty() ? "(none)" : std::to_string(seqs[0].kv_tokens) + " tokens");

        // ── 9. suspend now has something to save ─────────────────────────────
        std::cout << "\n9. suspend writes EVERY session, not sequence 0\n";
        const auto result = sup.suspend(slot);
        check(result.ok(), "suspend succeeds", result.message);
        check(!result.kv_checkpoint_path.empty(),
              "and reports where it went",
              result.kv_checkpoint_path);

        mm::SomaKvBackend soma_kv;
        mm::KvCheckpointInfo info;
        std::string err;
        check(soma_kv.stat(result.kv_checkpoint_path, info, err),
              "the manifest stats without a running engine",
              err);
        // The contrast with llama.cpp, made concrete: two sessions were live and
        // two are in the manifest. The fallback would have saved sequence 0 and
        // silently dropped the other.
        check(info.sequence == 2,
              "both live sessions are in it",
              std::to_string(info.sequence) + " sessions");
        check(info.length_tokens > 0,
              "with their token counts",
              std::to_string(info.length_tokens) + " tokens total");
        check(!info.arch_hash.empty(),
              "and an arch_hash for the pre-spawn check",
              info.arch_hash.substr(0, 16) + "...");

        const auto after = sup.find(slot);
        check(after && after->state == mm::SlotState::Suspended,
              "the slot is Suspended, and the process is stopped");

        (void)sup.unload_all(true);
        fs::remove_all(kvdir);
    }

    // ── 9b. streamed deltas reconstruct the answer ───────────────────────────
    //
    // The seam the Streamer sits in. Its own test proves the class is exact; this
    // proves the WIRING is — that the tail still held when generation ends is
    // flushed, and flushed before the request thread stops listening. A lost
    // flush is invisible in the happy case and eats the last character of every
    // answer that ends mid-codepoint.
    //
    // Greedy on both sides, because otherwise the two turns sample different text
    // and there is nothing to compare.
    std::cout << "\n9b. streamed deltas equal the non-streamed answer\n";
    {
        mm::EngineSupervisor sup(8290, 8300, /*max_slots=*/1);
        mm::EngineLoadRequest req;
        req.model_path = model;
        const auto slot = sup.load("soma", req, "agent-stream");
        check(!slot.empty(), "engine loaded", slot);
        if (!slot.empty()) {
            const auto port = sup.slots().front().port;
            const auto msgs = json::array({user_msg("the quick brown fox")});

            const auto whole = chat_greedy(port, msgs, 24);
            const auto deltas = chat_stream_deltas(port, msgs, 24);

            std::string joined;
            for (const auto& d : deltas)
                joined += d;

            check(!whole.empty(),
                  "the non-streamed turn produced text",
                  std::to_string(whole.size()) + " bytes");
            check(!deltas.empty(),
                  "the streamed turn produced deltas",
                  std::to_string(deltas.size()) + " frames");
            check(joined == whole,
                  "and they agree byte-for-byte",
                  joined == whole ? "identical"
                                  : (std::to_string(joined.size()) + " streamed vs " +
                                     std::to_string(whole.size()) + " whole"));

            // Every frame must be text on its own. This is the property the old
            // re-decode path did NOT have: it sent the difference between two
            // decodes, and that difference is a bare lead byte whenever a
            // codepoint spans two tokens.
            std::size_t bad = 0;
            for (const auto& d : deltas) {
                if (!ends_on_codepoint_boundary(d)) ++bad;
            }
            check(bad == 0,
                  "and no frame ends mid-codepoint",
                  bad == 0 ? "all frames are complete UTF-8"
                           : (std::to_string(bad) + " truncated frames"));

            // Say what was actually exercised. Fewer frames than tokens means a
            // codepoint spanned two tokens and the streamer held one back; equal
            // counts mean this particular generation never hit that case, and the
            // boundary check above passed without being tested. That is a fact
            // about the fixture's random weights, not something to assert — the
            // deterministic version of it lives in soma_tokenizer_g0's probe.
            std::cout << "   " << deltas.size() << " frames for 24 tokens — "
                      << (deltas.size() < 24 ? "a codepoint spanned two tokens"
                                             : "no split this run; see tokenizer_g0's probe")
                      << "\n";
        }
        (void)sup.unload_all(true);
    }

    // ── 10. concurrent turns share one forward ───────────────────────────────
    //
    // The batch union is the mechanism the whole engine is built around, and it
    // was unreachable over HTTP: `generate` held a mutex across the entire turn,
    // so the union only ever had one sequence to union. This is the check that
    // the fix is real rather than structural.
    std::cout << "\n10. concurrent turns land in ONE forward\n";
    {
        mm::EngineSupervisor sup(8280, 8290, /*max_slots=*/1);
        mm::EngineLoadRequest req;
        req.model_path = model;
        const auto slot = sup.load("soma", req, "agent-f");
        check(!slot.empty(), "engine loaded", slot);
        if (slot.empty()) {
            std::cout << "   last_error: " << sup.last_error() << "\n";
            return 1;
        }
        const auto port = sup.slots().front().port;

        const std::string probe = "concurrency probe prompt for the union";
        const std::uint32_t tokens = 48;

        // Alone first. This is the reference the concurrent run must reproduce.
        const auto solo = chat(port, json::array({user_msg(probe)}), "", tokens);
        check(!solo.empty(), "a solo turn produces output", std::to_string(solo.size()) + " chars");

        // Then the same turn alongside three others. A poller samples the
        // engine's own step statistics while they run.
        std::atomic<bool> polling{true};
        std::atomic<std::uint32_t> max_batch{0};
        std::atomic<std::uint32_t> max_unique{0}, max_naive{0};
        std::thread poller([&] {
            while (polling.load()) {
                const auto s = sessions(port);
                const auto b = s.value("current_batch", 0u);
                if (b > max_batch.load()) max_batch.store(b);
                const auto u = s.value("unique_experts_last_step", 0u);
                const auto n = s.value("naive_expert_reads_last_step", 0u);
                if (n > max_naive.load()) {
                    max_naive.store(n);
                    max_unique.store(u);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });

        std::vector<std::string> results(4);
        std::vector<std::thread> callers;
        for (int i = 0; i < 4; ++i) {
            callers.emplace_back([&, i] {
                const auto text = (i == 0) ? probe : probe + " variant " + std::to_string(i);
                results[static_cast<std::size_t>(i)] =
                    chat(port, json::array({user_msg(text)}), "", tokens);
            });
        }
        for (auto& t : callers)
            t.join();
        polling.store(false);
        poller.join();

        // THE gate. Batching sequences together must not change what any of them
        // says — the same property G3 asserts inside the scheduler, now asserted
        // through the HTTP boundary where the batch is actually assembled.
        check(results[0] == solo,
              "a turn's output is IDENTICAL whether it ran alone or in a batch",
              results[0] == solo ? "byte-for-byte" : "DIVERGED");
        check(!results[1].empty() && !results[2].empty() && !results[3].empty(),
              "and every concurrent turn gets its own answer");
        check(results[1] != results[0] && results[2] != results[1],
              "which are not spliced into each other",
              "distinct prompts gave distinct answers");

        check(max_batch.load() >= 2,
              "the engine really did batch them",
              "max observed batch = " + std::to_string(max_batch.load()));
        if (max_naive.load() > 0) {
            const double ratio =
                static_cast<double>(max_naive.load()) / std::max(1u, max_unique.load());
            std::cout << "   union at the widest sampled step: " << max_unique.load()
                      << " unique / " << max_naive.load() << " naive = " << ratio << "x\n";
        }

        (void)sup.unload_all(true);
    }

    // ── 11. telemetry over the wire ──────────────────────────────────────────
    //
    // bucket_heat()'s maths is checked in soma_telemetry_g6 without a model.
    // What needs a LIVE engine is the transport: that frames arrive at the
    // requested rate, that `?hz=` is clamped rather than obeyed or rejected, and
    // that a snapshot answers whether or not anyone is streaming.
    std::cout << "\n11. the telemetry feed ticks, and the rate is clamped\n";
    {
        // A CONTAINER when one is given, not the plain checkpoint. A resident
        // model has no MemoryHierarchy at all, so its grid is legitimately empty
        // and the counters this section reads would all be zero — the feed would
        // be exercised and the thing it reports would not.
        mm::EngineSupervisor sup(8300, 8310, /*max_slots=*/1);
        mm::EngineLoadRequest req;
        req.model_path = container.empty() ? model : container;
        const auto slot = sup.load("soma", req, "agent-t");
        check(!slot.empty(), "engine loaded", slot);
        if (slot.empty()) {
            std::cout << "   last_error: " << sup.last_error() << "\n";
            return 1;
        }
        const auto port = sup.slots().front().port;

        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(10);
            auto res = cli.Get("/internal/heat");
            check(res && res->status == 200, "GET /internal/heat answers");
            if (res) {
                const auto j = json::parse(res->body, nullptr, false);
                check(!j.is_discarded() && j.value("resolution", std::string{}) == "bucketed",
                      "and is BUCKETED by default");
            }
            auto full = cli.Get("/internal/heat?resolution=full");
            check(full && full->status == 200 &&
                      json::parse(full->body, nullptr, false).value("resolution", std::string{}) ==
                          "full",
                  "?resolution=full is an explicit opt-in");

            auto dump = cli.Get("/internal/telemetry/dump");
            check(dump && dump->status == 200 && dump->body.find("sched") != std::string::npos,
                  "and the G3 text dump still reads");

            if (!container.empty()) {
                // Drive one turn so the expert cache is actually touched, then
                // assert the counters MOVED. A telemetry feed reporting
                // structurally-correct zeros is indistinguishable from one wired
                // to nothing, which is the failure this catches.
                auto health = cli.Get("/health");
                check(health && health->body.find("\"streamed\":true") != std::string::npos,
                      "the container is served as STREAMED");

                chat(
                    port, json::array({user_msg("route some experts through this prompt")}), "", 8);

                auto after = cli.Get("/internal/heat");
                check(after && after->status == 200, "heat answers after a turn");
                if (after) {
                    const auto j = json::parse(after->body, nullptr, false);
                    check(!j.is_discarded() && j.value("n_experts", 0u) > 0,
                          "and reports the model's real dimensions",
                          std::to_string(j.value("n_layers", 0u)) + "x" +
                              std::to_string(j.value("n_experts", 0u)));
                    std::uint64_t hottest = 0;
                    for (const auto& c : j.value("counts", json::array())) {
                        hottest = std::max<std::uint64_t>(hottest, c.get<std::uint64_t>());
                    }
                    check(hottest > 0,
                          "with NON-ZERO counts — experts actually fired",
                          "hottest cell = " + std::to_string(hottest));
                }
                auto stats = cli.Get("/internal/telemetry/dump");
                check(stats && stats->body.find("hit_rate") != std::string::npos,
                      "and the cache reports a hit rate, so lookups happened");
            } else {
                std::cout << "   (no container given; streaming counters not exercised)\n";
            }
        }

        // Counted over a fixed window rather than waited on frame-by-frame: the
        // assertion is about RATE, and a per-frame wait would pass at any rate.
        const auto count_frames = [&](const std::string& query, int ms) {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(1, 0);
            int ticks = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
            cli.Get(("/internal/telemetry" + query).c_str(),
                    [&](const char* data, std::size_t len) {
                        const std::string chunk(data, len);
                        std::size_t at = 0;
                        while ((at = chunk.find("event: telemetry", at)) != std::string::npos) {
                            ++ticks;
                            at += 16;
                        }
                        return std::chrono::steady_clock::now() < deadline;
                    });
            return ticks;
        };

        // ~2 Hz over 1.5 s is about 3 frames. Bounded generously on both sides:
        // this is a scheduling assertion on a shared machine, not a stopwatch.
        const int slow = count_frames("", 1500);
        check(slow >= 2 && slow <= 8,
              "the default feed ticks near 2 Hz",
              std::to_string(slow) + " frames in 1.5s");

        const int fast = count_frames("?hz=10", 1500);
        check(fast > slow,
              "?hz=10 is faster than the default",
              std::to_string(fast) + " vs " + std::to_string(slow));

        // The ceiling is the ENGINE's. A client asking for 1000 Hz gets 10 — not
        // an error, because the limit is a property of the engine rather than a
        // mistake by the caller.
        const int clamped = count_frames("?hz=1000", 1500);
        check(clamped > 0, "?hz=1000 is CLAMPED, not refused", std::to_string(clamped) + " frames");
        check(clamped <= 30,
              "and does not exceed the 10 Hz ceiling",
              std::to_string(clamped) + " frames in 1.5s");

        const int garbage = count_frames("?hz=banana", 1200);
        check(garbage > 0,
              "an unparseable hz falls back to the default",
              std::to_string(garbage) + " frames");

        (void)sup.unload_all(true);
    }

    // ── 12. TWO DIFFERENT ENGINES, ONE SUPERVISOR ────────────────────────────
    // Deliberately LAST. llama-server loads the CUDA backend at startup, and
    // that plus its teardown perturbs §10, which asserts the engine actually
    // BATCHED four concurrent turns — a timing window that a busy machine
    // closes. Measured, not guessed: with this section running earlier, §10
    // failed 1 run in 3 with "max observed batch = 1"; with it here, and with
    // it skipped, 4 of 4 passed. Ordering is the fix rather than a sleep,
    // because coexistence has no dependency on running early and a sleep would
    // only move the same race.
    //
    // The G8 criterion: "a Soma agent and a fallback agent run concurrently on
    // the same node." Nothing asserted it. §5 above looks like it does and is the
    // OPPOSITE arrangement — it loads `soma` twice and proves the two agents
    // SHARE one process. That is Soma's per-sequence KV slot, not coexistence,
    // and reading it as this criterion is the mistake that left the peer-engine
    // claim resting on unexercised code (roadmap D13).
    //
    // What only shows up with two DIFFERENT engines: port allocation handing out
    // two ports from one pool, slot accounting counting heterogeneous engines,
    // SlotInfo::backend attributing each to the right descriptor, and one
    // lifecycle not disturbing the other.
    //
    // Deliberately NOT asserted: anything llama.cpp generates. The fixture has
    // random weights, so its logits are noise and its tokens are invalid UTF-8 —
    // llama-server itself errors building a response string from them. That is
    // inherent to a random model, not a defect, and this criterion is about
    // coexistence, which no logit participates in.
    std::cout << "\n12. Soma and llama.cpp run side by side\n";
    if (llama_exe.empty() || gguf.empty()) {
        // Loud, and counted as a skip rather than silently passing. A criterion
        // that reports green because its fixture was absent is worse than one
        // that reports nothing.
        ++skipped;
        std::cout << "   SKIPPED: no llama-server (arg 4) or gguf (arg 5).\n"
                  << "   The G8 coexistence criterion is NOT covered by this run.\n";
    } else {
        auto& reg = mm::EngineRegistry::instance();
        reg.register_engine(mm::make_llama_descriptor(llama_exe));

        mm::EngineSupervisor sup(8240, 8250, /*max_slots=*/2);

        mm::EngineLoadRequest soma_req;
        soma_req.model_path = container;
        const auto soma_slot = sup.load("soma", soma_req, "agent-soma");
        check(!soma_slot.empty(), "the Soma engine loads", sup.last_error());

        mm::EngineLoadRequest llama_req;
        llama_req.model_path = gguf;
        llama_req.settings.ctx_size = 512;
        const auto llama_slot = sup.load("llama-cpp", llama_req, "agent-llama");
        check(!llama_slot.empty(), "the llama.cpp engine loads ALONGSIDE it",
              sup.last_error());

        if (!soma_slot.empty() && !llama_slot.empty()) {
            check(soma_slot != llama_slot,
                  "they are separate slots, not a shared engine");
            check(sup.slots().size() == 2, "the supervisor holds two processes");
            check(sup.available_slot_count() == 0, "and the pool is now full");

            const auto si = sup.find(soma_slot);
            const auto li = sup.find(llama_slot);
            check(si && si->backend == "soma" && li && li->backend == "llama-cpp",
                  "each slot is attributed to the engine that actually serves it",
                  (si ? si->backend : std::string("?")) + " + " +
                      (li ? li->backend : std::string("?")));
            check(si && li && si->port != li->port,
                  "the port pool hands out two distinct ports",
                  std::to_string(si ? si->port : 0) + " vs " +
                      std::to_string(li ? li->port : 0));
            check(si && si->state == mm::SlotState::Ready &&
                      li && li->state == mm::SlotState::Ready,
                  "both are Ready at the same time — which is the criterion");

            // Each descriptor's own client reaches its own engine. A shared or
            // mis-attributed client would still "work" against one of them, so
            // both are checked.
            {
                auto sl = sup.acquire(soma_slot);
                auto ll = sup.acquire(llama_slot);
                check(sl && sl.get() != nullptr && sl.get()->health_check(),
                      "the Soma client reaches the Soma engine");
                check(ll && ll.get() != nullptr && ll.get()->health_check(),
                      "the llama.cpp client reaches the llama.cpp engine");
            }

            // Independent lifecycle: the failure this guards against is a
            // supervisor keyed on something the two engines share, where
            // unloading one takes the other down with it.
            const auto gone = sup.unload(llama_slot);
            check(gone.ok(), "unloading the fallback succeeds", gone.message);
            const auto survivor = sup.find(soma_slot);
            check(survivor && survivor->state == mm::SlotState::Ready,
                  "and the Soma engine is untouched — still Ready",
                  survivor ? std::string("Ready") : std::string("(gone)"));
            check(sup.available_slot_count() == 1, "one slot freed, not two");
        }
        (void)sup.unload_all(true);
    }


    // A skip is reported in the SAME line as the verdict. Buried in scrollback it
    // reads as coverage that ran; here "OK" alone means everything ran.
    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << (skipped > 0 ? " (" + std::to_string(skipped) + " SECTION SKIPPED)" : "")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
