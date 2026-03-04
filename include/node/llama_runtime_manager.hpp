#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

class NodeState;

struct LlamaRuntimeStatus {
    bool        running = false;
    std::string status  = "idle"; // idle|running|succeeded|failed
    std::string message;
    int64_t     started_ms  = 0;
    int64_t     finished_ms = 0;

    std::string repo_url;
    std::string install_root;
    std::string repo_dir;
    std::string build_dir;
    std::string binary_path;

    std::string installed_commit;
    std::string remote_commit;
    std::string remote_error;
    int64_t     remote_checked_ms = 0;
    bool        update_available  = false;
    std::string update_reason     = "unknown";

    std::string              last_job_id;
    std::string              last_log_path;
    std::vector<std::string> last_log_tail;
};

struct LlamaRuntimeLogChunk {
    bool                     found = false;
    std::string              job_id;
    std::string              log_path;
    size_t                   offset = 0;
    size_t                   next_offset = 0;
    std::vector<std::string> lines;
};

class LlamaRuntimeManager {
public:
    struct Options {
        std::string            repo_url = "https://github.com/ggml-org/llama.cpp.git";
        std::filesystem::path  install_root = std::filesystem::path("data") / "llama.cpp";
        std::filesystem::path  metadata_file = std::filesystem::path("data") / "llama_runtime.json";
        std::filesystem::path  log_dir = std::filesystem::path("logs") / "llama-updater";
        std::filesystem::path  updater_script;
    };

    using LogCallback = std::function<void(const std::string&)>;
    using BinaryReadyCallback = std::function<void(const std::string&)>;

    LlamaRuntimeManager(NodeState& state, Options opts);
    ~LlamaRuntimeManager();

    void set_log_callback(LogCallback cb);
    void set_binary_ready_callback(BinaryReadyCallback cb);

    bool check_update(std::string* out_message = nullptr, bool force_remote = false);
    bool start_update(bool build,
                      bool force,
                      std::string* out_job_id = nullptr,
                      std::string* out_message = nullptr);

    LlamaRuntimeStatus get_status(size_t log_tail_lines = 80) const;
    LlamaRuntimeLogChunk read_job_log(const std::string& job_id, size_t offset, size_t limit) const;

private:
    NodeState& state_;
    Options    opts_;

    mutable std::mutex mutex_;
    LlamaRuntimeStatus status_;
    std::unordered_map<std::string, std::filesystem::path> job_logs_;
    LogCallback        log_cb_;
    BinaryReadyCallback binary_ready_cb_;

    std::filesystem::path find_script_path() const;
    bool refresh_versions_locked(bool force_remote, std::string* out_message);
    bool persist_metadata_locked() const;
    bool load_metadata_locked();
    void sync_state_locked() const;

    void run_update_worker(std::string job_id, bool build, bool force);
};

} // namespace mm
