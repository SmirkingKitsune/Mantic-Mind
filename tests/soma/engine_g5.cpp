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
        std::cerr << "usage: engine_g5 <soma_exe> <model_dir>\n";
        return 2;
    }
    const std::string exe = argv[1];
    const std::string model = argv[2];

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

    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
