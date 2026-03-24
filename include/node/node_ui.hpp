#pragma once

#include "common/models.hpp"
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

class NodeState;

// FTXUI-based terminal UI for mantic-mind.
//
// Waiting state:  centered panel with spinner, API key, and listening port.
// Connected state: status grid + health bars + API key table + log panel.
//
// Run on the main thread; NodeState updates are delivered via callbacks that
// post to the FTXUI event loop.
class NodeUI {
public:
    using UpdateRequestCallback = std::function<void()>;
    using ModelPullCallback = std::function<bool(
        const std::string& model_filename,
        std::string* out_message)>;
    using ModelDeleteCallback = std::function<bool(
        const std::string& model_filename,
        std::string* out_message)>;

    NodeUI(NodeState& state, uint16_t listen_port,
           UpdateRequestCallback update_request_cb = {},
           ModelPullCallback pull_cb = {},
           ModelDeleteCallback delete_cb = {});
    ~NodeUI();

    // Append a log line from llama-server (thread-safe, posts to UI event loop).
    void append_log(const std::string& line);

    // Blocks until the user quits (ESC / q / window close).
    void run();

    // Call from any thread to trigger graceful UI exit.
    void quit();

private:
    NodeState& state_;
    uint16_t   listen_port_;
    UpdateRequestCallback update_request_cb_;
    ModelPullCallback pull_cb_;
    ModelDeleteCallback delete_cb_;

    static constexpr size_t kMaxLogLines = 4000;
    static constexpr int    kLogScrollPage = 8;

    mutable std::mutex       log_mutex_;
    std::vector<std::string> log_lines_;
    int                      log_scroll_from_bottom_ = 0;
    std::string              log_file_path_;
    std::ofstream            log_file_;

    std::mutex            screen_mutex_;
    std::function<void()> quit_fn_;
    std::function<void()> refresh_fn_;
};

} // namespace mm
