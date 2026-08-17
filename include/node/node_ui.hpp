#pragma once

#include "common/engine_config.hpp"
#include "common/models.hpp"
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

class NodeState;

/// Which modal owns the node TUI's screen. At most one — that is the type's job.
///
/// This was five independent booleans whose mutual exclusion was maintained by
/// five scattered "if X then Y = false" statements, with the precedence written
/// a SECOND time, in a different order, in the event handler. The two agreed
/// only because the recompute happened to leave at most one true.
enum class NodeModal : std::uint8_t {
    None,
    Progress,      ///< an install/transfer is running; outranks everything
    EngineSwitch,  ///< operator asked to change the llama.cpp variant
    Troubleshoot,  ///< llama.cpp provisioning failed and wants a decision
    Target,        ///< the active build is not the intended one
    Update,        ///< a newer llama.cpp is available
};

/// Everything the ladder depends on, so the decision is a pure function.
///
/// Split into "can" and "unacknowledged" per modal because they are different
/// questions: whether the prompt APPLIES, and whether the operator has already
/// dismissed this particular version/fingerprint/target. Collapsing them is how
/// a dismissed prompt reopens on the next frame.
struct NodeModalInputs {
    bool progress_active = false;

    /// EngineSwitch never auto-opens — nothing about the runtime asks for it,
    /// so it appears only while it is already open (i.e. the operator pressed
    /// the button). There is deliberately no `engine_switch_unacknowledged`.
    bool engine_switch_available = false;
    bool engine_variants_listed = false;

    bool can_troubleshoot = false;
    bool troubleshoot_unacknowledged = false;
    bool can_install_target = false;
    bool target_unacknowledged = false;
    bool can_update = false;
    bool update_unacknowledged = false;
};

/// The whole modal ladder. PURE, and the only place precedence is expressed.
///
/// `current` is the modal showing now, which is what makes a prompt sticky once
/// opened: an auto-opening modal that closed the moment its "unacknowledged"
/// flag cleared would vanish under the operator mid-read.
///
/// Precedence, highest first: Progress, EngineSwitch, Troubleshoot, Target,
/// Update.
NodeModal resolve_node_modal(const NodeModalInputs& in, NodeModal current) noexcept;
const char* to_string(NodeModal modal) noexcept;

// FTXUI-based terminal UI for mantic-mind.
//
// Waiting state:  centered panel with spinner, API key, and listening port.
// Connected state: status grid + health bars + API key table + log panel.
//
// Run on the main thread; NodeState updates are delivered via callbacks that
// post to the FTXUI event loop.
class NodeUI {
public:
    using ForgetPairingCallback = std::function<bool(std::string* out_message)>;
    // Empty accelerator approves the assessed current-backend action. A value
    // such as vulkan/cpu selects an official release alternative.
    using RequestLlamaUpdateCallback = std::function<void(std::string accelerator)>;
    // Change the active managed llama.cpp execution variant independently of
    // update availability (for example cuda-12, vulkan, or cpu).
    using RequestLlamaSwitchCallback = std::function<void(std::string variant)>;
    // Runtime/wizard action: diagnose | retry | target | compile-anyway | release.
    // `variant` is populated only for release and is a report variant id.
    using RequestLlamaRecoveryCallback =
        std::function<void(std::string action, std::string variant)>;

    /// What this node was told to run, and whether it is running it.
    ///
    /// A PROVIDER rather than fields mirrored into NodeState: conformance is
    /// derived from live provisioner status, so a copy stored on state change
    /// would report the moment it last succeeded rather than the current one.
    /// A runtime that fails after a good apply is drift, and drift the TUI
    /// cannot see is the whole class of problem this work exists to close.
    struct EngineView {
        ClusterEngineConfig config;
        EngineConformance conformance;
        std::vector<RuntimeStatus> runtimes;
    };
    using EngineViewProvider = std::function<EngineView()>;

    NodeUI(NodeState& state, uint16_t listen_port,
           ForgetPairingCallback forget_pairing_cb = {},
           RequestLlamaUpdateCallback request_llama_update_cb = {},
           RequestLlamaSwitchCallback request_llama_switch_cb = {},
           RequestLlamaRecoveryCallback request_llama_recovery_cb = {},
           EngineViewProvider engine_view_provider = {});
    ~NodeUI();

    // Append a log line from the runtime engine (thread-safe, posts to UI event loop).
    void append_log(const std::string& line);

    // Blocks until the user quits (ESC / q / window close).
    void run();

    // Call from any thread to trigger graceful UI exit.
    void quit();

private:
    NodeState& state_;
    uint16_t   listen_port_;
    ForgetPairingCallback forget_pairing_cb_;
    RequestLlamaUpdateCallback request_llama_update_cb_;
    RequestLlamaSwitchCallback request_llama_switch_cb_;
    RequestLlamaRecoveryCallback request_llama_recovery_cb_;
    EngineViewProvider engine_view_provider_;

    static constexpr size_t kMaxLogLines = 4000;
    static constexpr int    kLogScrollPage = 8;

    mutable std::mutex       log_mutex_;
    std::deque<std::string>  log_lines_;
    int                      log_scroll_from_bottom_ = 0;
    std::string              log_file_path_;
    std::ofstream            log_file_;

    // Metric history for the health sparkline/braille graphs. Sampled once per
    // second from inside the render lambda (which only ever runs on the FTXUI
    // loop thread), so these need no separate lock.
    static constexpr size_t  kHistLen = 60;   // ~60s window at 1 sample/s
    std::deque<float>        hist_cpu_, hist_ram_, hist_gpu_, hist_vram_;
    int64_t                  last_hist_ms_ = 0;
    int64_t                  started_ms_   = 0;   // process/UI start, for uptime

    std::mutex            screen_mutex_;
    std::function<void()> quit_fn_;
    std::function<void()> refresh_fn_;
};

} // namespace mm
