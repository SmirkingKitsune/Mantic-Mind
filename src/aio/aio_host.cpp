#include "aio/aio_host.hpp"

#include "aio/local_node_operations.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/agent_scheduler.hpp"
#include "control/control_api_server.hpp"
#include "control/control_host.hpp"
#include "control/control_ui.hpp"
#include "control/node_registry.hpp"
#include "node/llama_cpp_provisioner.hpp"
#include "node/llama_runtime.hpp"
#include "node/node_host.hpp"
#include "node/node_service.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#  include <Windows.h>
#  include <conio.h>
#  include <io.h>
#else
#  include <cerrno>
#  include <poll.h>
#  include <unistd.h>
#endif

namespace mm {

namespace {

constexpr std::size_t kMaxRuntimeLogLines = 4000;

std::string control_client_base_url(std::string bind_host, uint16_t port) {
    bind_host = util::trim(bind_host);
    if (bind_host == "0.0.0.0") bind_host = "127.0.0.1";
    if (bind_host == "::") bind_host = "::1";
    if (bind_host.find(':') != std::string::npos && bind_host.front() != '[') {
        bind_host = "[" + bind_host + "]";
    }
    return "http://" + bind_host + ":" + std::to_string(port);
}

NodeCapabilities initial_capabilities(const NodeConfig& config,
                                      const NodeHealthMetrics& metrics,
                                      int detected_nvidia_count) {
    NodeCapabilities capabilities;
    capabilities.arch = current_runtime_arch();
    capabilities.gpu_count = resolve_node_gpu_count(
        config.node_gpu_count,
        detected_nvidia_count,
        metrics.gpu_backend_available || metrics.gpu_vram_total_mb > 0);
    return capabilities;
}

std::string response_error(const HttpResponse& response) {
    if (response.body.empty()) return "HTTP " + std::to_string(response.status);
    try {
        const auto body = nlohmann::json::parse(response.body);
        return body.value("error", response.body);
    } catch (...) {
        return response.body;
    }
}

void print_cli_help() {
    std::cout
        << "Commands:\n"
        << "  nodes list|discovered|add|remove|forget|pair ...\n"
        << "  models list\n"
        << "  agents list|show|create|update|delete ...\n"
        << "  chat <agent-id> <message>\n"
        << "  chat send <agent-id> <message> [conversation-id]\n"
        << "  curation conv list|create|activate|delete|compact ...\n"
        << "  curation mem list|delete|extract ...\n"
        << "  activity tail [n] [level]\n"
        << "  node status [node-id]\n"
        << "  node slots [node-id]\n"
        << "  node logs [node-id] [tail]\n"
        << "  node cancel [node-id]\n"
        << "  node runtime status [node-id]\n"
        << "  node runtime provision [node-id]\n"
        << "  node runtime update [node-id] [--accelerator NAME]\n"
        << "  node runtime check-update [node-id]\n"
        << "  node runtime switch <variant> [node-id]\n"
        << "  node runtime diagnose [node-id]\n"
        << "  node runtime recover <retry|target|compile-anyway> [node-id]\n"
        << "  node runtime recover release <variant> [node-id]\n"
        << "  help | quit\n";
}

// std::getline cannot be interrupted when stdin remains open without a full
// line.  That makes a headless `--cli` process ignore SIGINT/SIGTERM until its
// parent writes or closes the pipe.  Read readiness in short intervals so the
// host's quit flag remains authoritative during shutdown.
bool read_cli_line_interruptibly(
    std::string* line,
    const std::function<bool()>& quit_requested) {
    if (!line) return false;
    line->clear();

#ifdef _WIN32
    const int stdin_fd = _fileno(stdin);
    if (stdin_fd >= 0 && _isatty(stdin_fd)) {
        std::wstring wide_line;
        const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
        auto write_console = [&](const wchar_t* text, DWORD length) {
            DWORD written = 0;
            if (output != nullptr && output != INVALID_HANDLE_VALUE) {
                WriteConsoleW(output, text, length, &written, nullptr);
            }
        };

        while (!quit_requested()) {
            if (!_kbhit()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                continue;
            }
            const wchar_t ch = static_cast<wchar_t>(_getwch());
            if (ch == 0 || ch == 0xE0) {
                // Consume the scan code for navigation/function keys.  The
                // compact AIO CLI intentionally keeps only basic line editing.
                (void)_getwch();
                continue;
            }
            if (ch == L'\r' || ch == L'\n') {
                write_console(L"\r\n", 2);
                if (wide_line.empty()) return true;
                const int bytes = WideCharToMultiByte(
                    CP_UTF8, 0, wide_line.data(),
                    static_cast<int>(wide_line.size()), nullptr, 0, nullptr,
                    nullptr);
                if (bytes <= 0) return false;
                line->resize(static_cast<std::size_t>(bytes));
                (void)WideCharToMultiByte(
                    CP_UTF8, 0, wide_line.data(),
                    static_cast<int>(wide_line.size()), line->data(), bytes,
                    nullptr, nullptr);
                return true;
            }
            if (ch == 0x1A && wide_line.empty()) return false; // Ctrl+Z
            if (ch == L'\b') {
                if (!wide_line.empty()) {
                    wide_line.pop_back();
                    write_console(L"\b \b", 3);
                }
                continue;
            }
            if (ch >= L' ') {
                wide_line.push_back(ch);
                write_console(&ch, 1);
            }
        }
        return false;
    }

    const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    if (input == nullptr || input == INVALID_HANDLE_VALUE) return false;
    const DWORD input_type = GetFileType(input);
    for (;;) {
        if (quit_requested()) return false;

        if (input_type == FILE_TYPE_PIPE) {
            DWORD available = 0;
            if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) {
                const DWORD code = GetLastError();
                if (code == ERROR_BROKEN_PIPE || code == ERROR_HANDLE_EOF) {
                    return !line->empty();
                }
                return false;
            }
            if (available == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                continue;
            }
        }

        char ch = 0;
        DWORD read = 0;
        if (!ReadFile(input, &ch, 1, &read, nullptr) || read == 0) {
            return !line->empty();
        }
        if (ch == '\n') return true;
        if (ch != '\r') line->push_back(ch);
    }
#else
    pollfd input{};
    input.fd = STDIN_FILENO;
    input.events = POLLIN | POLLHUP;
    for (;;) {
        if (quit_requested()) return false;
        input.revents = 0;
        const int ready = ::poll(&input, 1, 50);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (ready == 0) continue;
        if ((input.revents & (POLLERR | POLLNVAL)) != 0) return false;

        char ch = 0;
        const ssize_t count = ::read(STDIN_FILENO, &ch, 1);
        if (count == 0) return !line->empty();
        if (count < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return false;
        }
        if (ch == '\n') return true;
        if (ch != '\r') line->push_back(ch);
    }
#endif
}

} // namespace

struct AioHost::Impl {
    explicit Impl(AioConfig value, bool network_allowed)
        : config(std::move(value)), allow_network(network_allowed) {}

    AioConfig config;
    bool allow_network = false;
    bool tui_mode = true;
    std::atomic<bool> started{false};
    std::atomic<bool> stopping{false};
    std::atomic<bool> quit_requested{false};
    std::atomic<bool> listener_failed{false};

    ControlConfig control_config;
    NodeConfig node_config;

    std::unique_ptr<ControlHost> control_host;
    std::unique_ptr<NodeHost> node_host;
    NodeState* state = nullptr;
    SlotManager* slots = nullptr;
    std::unique_ptr<LlamaCppProvisioner> provisioner;
    NodeService* node_service = nullptr;
    std::shared_ptr<LocalNodeOperations> local_operations;
    std::unique_ptr<ControlUI> ui;

    std::mutex runtime_operation_mutex;
    std::mutex runtime_log_mutex;
    std::deque<std::string> runtime_logs;
    std::atomic<bool> stop_workers{false};
    std::thread runtime_resolve_thread;
    std::thread runtime_update_thread;
    std::thread local_refresh_thread;

    void append_runtime_log(std::string line) {
        std::lock_guard<std::mutex> lock(runtime_log_mutex);
        if (runtime_logs.size() >= kMaxRuntimeLogLines) runtime_logs.pop_front();
        runtime_logs.push_back(std::move(line));
    }

    std::vector<std::string> tail_runtime_logs(int tail) {
        tail = std::max(1, tail);
        std::lock_guard<std::mutex> lock(runtime_log_mutex);
        const std::size_t count = std::min(
            runtime_logs.size(), static_cast<std::size_t>(tail));
        return std::vector<std::string>(
            runtime_logs.end() - static_cast<std::ptrdiff_t>(count),
            runtime_logs.end());
    }

    void report_worker_failure(const char* worker,
                               const std::string& detail,
                               bool fatal) noexcept {
        try {
            const std::string message = std::string("AIO ") + worker +
                                        " worker failed: " + detail;
            append_runtime_log(message);
            MM_ERROR("{}", message);
            if (state) state->set_last_error(message);
        } catch (...) {
            // A failure reporter must never escape a std::thread entry point.
        }
        if (!fatal || stopping) return;
        listener_failed = true;
        quit_requested = true;
        try {
            request_work_cancellation();
            if (ui) ui->quit();
        } catch (...) {
        }
    }

    template <typename Work>
    void run_worker(const char* name, bool fatal, Work&& work) noexcept {
        try {
            std::forward<Work>(work)();
        } catch (const std::exception& exception) {
            report_worker_failure(name, exception.what(), fatal);
        } catch (...) {
            report_worker_failure(name, "unknown exception", fatal);
        }
    }

    LlamaRuntimeStatus apply_runtime(LlamaRuntimeStatus runtime) {
        if (llama_runtime_usable(runtime)) {
            slots->set_llama_server_path(runtime.executable_path);
            auto capabilities = state->get_capabilities();
            capabilities.llama_cpp_version = runtime.version;
            state->set_capabilities(capabilities);
            state->set_last_error("");
        } else if (!runtime.last_error.empty()) {
            state->set_last_error(runtime.last_error);
        }
        state->set_llama_runtime(runtime);
        if (control_host && control_host->initialized()) {
            control_host->registry().refresh_node("local");
        }
        return runtime;
    }

    LlamaRuntimeStatus resolve_runtime() {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->ensure_runtime());
    }

    LlamaRuntimeStatus provision_runtime() {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        auto current = provisioner->ensure_runtime();
        if (llama_runtime_usable(current)) return apply_runtime(std::move(current));
        if (stop_workers) return current;
        return apply_runtime(provisioner->update_runtime());
    }

    LlamaRuntimeStatus update_runtime(const std::string& accelerator) {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->update_runtime(accelerator));
    }

    LlamaRuntimeStatus switch_runtime(const std::string& variant) {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->switch_runtime(variant));
    }

    LlamaRuntimeStatus check_runtime_update() {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->check_for_update());
    }

    LlamaRuntimeStatus diagnose_runtime() {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->diagnose_environment(false));
    }

    LlamaRuntimeStatus recover_runtime(const std::string& action,
                                       const std::string& variant) {
        std::lock_guard<std::mutex> lock(runtime_operation_mutex);
        if (stop_workers) return provisioner->status();
        return apply_runtime(provisioner->recover_runtime(action, variant));
    }

    void start_background_workers() {
        runtime_resolve_thread = std::thread([this] {
            run_worker("runtime resolve", false, [this] {
                const auto runtime = resolve_runtime();
                if (llama_runtime_usable(runtime)) {
                    MM_INFO("AIO embedded node resolved llama-server at {}",
                            runtime.executable_path);
                } else {
                    MM_WARN("AIO started without an available llama runtime: {}",
                            runtime.last_error.empty()
                                ? "not installed" : runtime.last_error);
                }
            });
        });

        if (node_config.llama_update_check &&
            node_config.llama_update_policy != "manual") {
            runtime_update_thread = std::thread([this] {
                run_worker("runtime update", false, [this] {
                    for (int i = 0; i < 50 && !stop_workers; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    while (!stop_workers) {
                        const auto status = check_runtime_update();
                        if (!stop_workers && status.update_available &&
                            node_config.llama_update_policy == "auto") {
                            update_runtime({});
                        }
                        const int64_t interval_ms = std::max<int64_t>(
                            1, node_config.llama_update_check_interval_hours) *
                            60 * 60 * 1000;
                        for (int64_t waited = 0;
                             waited < interval_ms && !stop_workers;
                             waited += 200) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(200));
                        }
                    }
                });
            });
        }

        local_refresh_thread = std::thread([this] {
            run_worker("local refresh", true, [this] {
                while (!stop_workers) {
                    if (control_host && control_host->initialized()) {
                        control_host->registry().refresh_node("local");
                    }
                    for (int i = 0; i < 20 && !stop_workers; ++i) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(100));
                    }
                }
            });
        });

    }

    void request_work_cancellation() {
        stop_workers = true;
        if (control_host) control_host->request_shutdown();
        if (local_operations) local_operations->request_shutdown();
        if (node_host) node_host->request_shutdown();
    }

    void stop_all() {
        if (!started.exchange(false)) return;
        if (stopping.exchange(true)) return;

        // Stop ingress first, then drain the control queue while the embedded
        // node and its llama-server children remain available.
        request_work_cancellation();
        if (local_refresh_thread.joinable()) local_refresh_thread.join();
        if (runtime_update_thread.joinable()) runtime_update_thread.join();
        if (runtime_resolve_thread.joinable()) runtime_resolve_thread.join();

        // ControlHost drains AgentQueue before releasing its registry. The
        // embedded NodeService remains alive until that drain completes.
        if (control_host) control_host->stop();

        ui.reset();
        control_host.reset();
        local_operations.reset();
        node_service = nullptr;
        provisioner.reset();
        if (node_host) node_host->stop();
        node_host.reset();
        slots = nullptr;
        state = nullptr;
        stopping = false;
        MM_INFO("mantic-mind-aio stopped");
        if (auto logger = spdlog::get("mm-aio")) logger->flush();
        spdlog::drop("mm-aio");
    }
};

AioHost::AioHost(AioConfig config, bool allow_network)
    : impl_(std::make_unique<Impl>(std::move(config), allow_network)) {}

AioHost::~AioHost() { stop(); }

bool AioHost::start(bool tui_mode, std::string* error) {
    if (error) error->clear();
    if (impl_->started) {
        if (error) *error = "AIO host is already started";
        return false;
    }

    const auto config_issues = validate_aio_config(impl_->config);
    if (!config_issues.empty()) {
        if (error) {
            *error = "invalid AIO configuration";
            for (const auto& issue : config_issues) {
                *error += "; ";
                if (!issue.key.empty()) *error += issue.key + ": ";
                *error += issue.message;
            }
        }
        return false;
    }

    impl_->tui_mode = tui_mode;
    impl_->stopping = false;
    impl_->quit_requested = false;
    impl_->listener_failed = false;
    impl_->stop_workers = false;
    {
        std::lock_guard<std::mutex> lock(impl_->runtime_log_mutex);
        impl_->runtime_logs.clear();
    }
    impl_->control_config = make_control_config(impl_->config);
    impl_->node_config = make_node_config(impl_->config);

    ControlHost::Options control_options;
    control_options.config = impl_->control_config;
    control_options.bind_host = util::trim(impl_->config.control.bind_host);
    control_options.enable_remote_nodes = impl_->config.cluster.enabled;
    control_options.enable_discovery =
        impl_->config.cluster.enabled &&
        impl_->config.cluster.discovery_enabled;
    control_options.allow_legacy_environment = false;
    impl_->control_host =
        std::make_unique<ControlHost>(std::move(control_options));
    if (!impl_->control_host->acquire_singleton_lock(error)) {
        impl_->control_host.reset();
        return false;
    }
    NodeHost::Options node_options;
    node_options.config = impl_->node_config;
    node_options.node_id = "local";
    node_options.registered = true;
    node_options.mark_control_contact = true;
    node_options.manage_model_cache = false;
    impl_->node_host = std::make_unique<NodeHost>(std::move(node_options));
    if (!impl_->node_host->acquire_singleton_lock(error)) {
        impl_->node_host.reset();
        impl_->control_host->stop();
        impl_->control_host.reset();
        return false;
    }

    impl_->started = true;
    try {
        auto ensure_directory = [](const std::string& path) {
            std::error_code ec;
            std::filesystem::create_directories(path, ec);
            if (ec) {
                throw std::runtime_error(
                    "failed to create AIO directory '" + path + "': " + ec.message());
            }
        };
        ensure_directory(impl_->control_config.data_dir);
        ensure_directory(impl_->control_config.models_dir);

        init_logger(impl_->control_config.log_file, "mm-aio",
                    tui_mode ? spdlog::level::off : spdlog::level::info,
                    spdlog::level::trace);
        MM_INFO("mantic-mind-aio starting (cluster={}, runtime_network_policy={})",
                impl_->config.cluster.enabled,
                to_string(impl_->config.node.runtime_network_policy));

        std::string node_error;
        if (!impl_->node_host->initialize(&node_error)) {
            throw std::runtime_error(node_error.empty()
                ? "failed to initialize embedded node host"
                : node_error);
        }
        impl_->state = &impl_->node_host->state();
        impl_->slots = &impl_->node_host->slots();
        impl_->node_service = &impl_->node_host->service();
        if (!impl_->node_host->initial_metrics_ready()) {
            MM_WARN("Timed out waiting for the initial AIO hardware metrics sample; "
                    "falling back to direct host GPU probes");
        }
        const auto initial_host_metrics = impl_->state->get_metrics();
        const int detected_nvidia_gpu_count = detect_nvidia_gpu_count();
        impl_->state->set_capabilities(initial_capabilities(
            impl_->node_config,
            initial_host_metrics,
            detected_nvidia_gpu_count));

        LlamaProvisionConfig provision_config;
        provision_config.requested_executable = impl_->node_config.llama_server_path;
        provision_config.provision_dir = impl_->node_config.llama_provision_dir;
        provision_config.auto_provision = impl_->node_config.llama_auto_provision;
        provision_config.install_method = impl_->node_config.llama_install_method;
        provision_config.version = impl_->node_config.llama_version;
        provision_config.accelerator = impl_->node_config.llama_accelerator.empty()
            ? detect_llama_accelerator(current_runtime_platform(), current_runtime_arch(),
                                       initial_host_metrics.gpu_backend_available ||
                                           detected_nvidia_gpu_count > 0,
                                       detect_rocm_present())
            : impl_->node_config.llama_accelerator;
        provision_config.accelerator_explicit =
            !impl_->node_config.llama_accelerator.empty();
        provision_config.cuda_arch = impl_->node_config.llama_cuda_arch;
        if (provision_config.accelerator == "cuda" &&
            provision_config.cuda_arch.empty()) {
            provision_config.cuda_arch = detect_cuda_architectures();
            if (!provision_config.cuda_arch.empty()) {
                MM_INFO("Detected CUDA compute architecture(s) for AIO llama.cpp: {}",
                        provision_config.cuda_arch);
            }
        }
        provision_config.cmake_args = impl_->node_config.llama_cmake_args;
        provision_config.build_jobs = impl_->node_config.llama_build_jobs;
        impl_->provisioner =
            std::make_unique<LlamaCppProvisioner>(std::move(provision_config));
        impl_->state->set_llama_runtime(impl_->provisioner->status());

        impl_->provisioner->set_log_sink([owner = impl_.get()](
            const std::string& line, bool is_stderr) {
            owner->append_runtime_log(is_stderr ? "[stderr] " + line : line);
            if (is_stderr) MM_WARN("[llama] {}", line);
            else MM_INFO("[llama] {}", line);
        });
        impl_->provisioner->set_progress_sink([owner = impl_.get()](
            const RuntimeInstallProgress& progress) {
            if (!progress.active) {
                owner->state->clear_action_progress("llama-runtime");
                return;
            }
            NodeActionProgress action;
            action.active = true;
            action.operation_id = "llama-runtime";
            action.kind = "runtime";
            action.action = "Provisioning llama.cpp runtime";
            action.target = "llama-server";
            action.stage = progress.stage;
            action.detail = progress.last_line;
            action.step = progress.step;
            action.total_steps = progress.total_steps;
            action.fraction = progress.fraction;
            action.cancelable = true;
            owner->state->set_action_progress(action);
        });
        impl_->provisioner->set_cancel_check([owner = impl_.get()] {
            return owner->stop_workers.load() ||
                   owner->state->action_cancel_requested("llama-runtime");
        });

        impl_->slots->set_log_callback([owner = impl_.get()](
            const std::string& line, bool is_stderr) {
            owner->append_runtime_log(is_stderr ? "[stderr] " + line : line);
        });

        impl_->node_service->set_runtime_logs_provider([owner = impl_.get()](int tail) {
            return owner->tail_runtime_logs(tail);
        });
        impl_->node_service->set_llama_provision_callback([owner = impl_.get()] {
            return owner->provision_runtime();
        });
        impl_->node_service->set_llama_update_callback([owner = impl_.get()](
            const std::string& accelerator) {
            return owner->update_runtime(accelerator);
        });
        impl_->node_service->set_llama_switch_callback([owner = impl_.get()](
            const std::string& variant) {
            return owner->switch_runtime(variant);
        });
        impl_->node_service->set_llama_check_update_callback([owner = impl_.get()] {
            return owner->check_runtime_update();
        });
        impl_->node_service->set_llama_diagnose_callback([owner = impl_.get()] {
            return owner->diagnose_runtime();
        });
        impl_->node_service->set_llama_recovery_callback([owner = impl_.get()](
            const std::string& action, const std::string& variant) {
            return owner->recover_runtime(action, variant);
        });

        impl_->local_operations = std::make_shared<LocalNodeOperations>(
            *impl_->node_service, impl_->control_config.models_dir,
            std::string(to_string(impl_->config.node.runtime_network_policy)));

        std::string control_error;
        if (!impl_->control_host->initialize(&control_error)) {
            throw std::runtime_error(control_error.empty()
                ? "failed to initialize embedded control host"
                : control_error);
        }
        auto& agents = impl_->control_host->agents();
        auto& registry = impl_->control_host->registry();
        auto& scheduler = impl_->control_host->scheduler();
        auto& api = impl_->control_host->api();

        registry.add_embedded_node(
            impl_->local_operations, current_runtime_platform(), util::hostname());
        impl_->local_operations->set_mutation_callback([owner = impl_.get()] {
            if (owner->control_host && owner->control_host->initialized()) {
                owner->control_host->registry().refresh_node("local");
            }
        });
        impl_->ui = std::make_unique<ControlUI>(
            registry, agents, scheduler,
            impl_->control_config.models_dir,
            control_client_base_url(impl_->config.control.bind_host,
                                    impl_->control_config.listen_port),
            impl_->control_config.external_api_token,
            [owner = impl_.get()](const std::string& agent_id,
                                  const std::string& message,
                                  std::string* text,
                                  std::string* conversation_id,
                                  std::string* chat_error) {
                const auto result = owner->control_host->api().chat_local(
                    agent_id, message);
                std::string combined;
                for (const auto& chunk : result.chunks) {
                    combined += chunk.delta_content;
                }
                if (text) *text = std::move(combined);
                if (conversation_id) *conversation_id = result.conv_id;
                if (chat_error) *chat_error = result.error;
                return result.success;
            },
            [owner = impl_.get()] { owner->request_work_cancellation(); });
        impl_->ui->set_pairing_key(impl_->control_config.pairing_key);
        api.set_log_callback([owner = impl_.get()](
            int level, const std::string& message) {
            const auto ui_level = level >= 2 ? ControlUI::LogLevel::Error
                : level == 1 ? ControlUI::LogLevel::Warn
                             : ControlUI::LogLevel::Info;
            owner->ui->log(ui_level, message);
        });
        registry.set_update_callback([owner = impl_.get()](const NodeInfo&) {
            if (owner->ui) owner->ui->refresh();
        });
        impl_->control_host->set_failure_callback(
            [owner = impl_.get()](const std::string& detail) {
                owner->report_worker_failure("control", detail, true);
            });

        // Establish both control-facing listeners before starting any worker
        // that can contact remote nodes or perform runtime network work.  A
        // bind failure therefore rolls back without discovery, polling, or
        // provisioning traffic.
        if (!impl_->control_host->start(error)) {
            impl_->stop_all();
            return false;
        }
        impl_->start_background_workers();
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        impl_->stop_all();
        return false;
    }

    MM_INFO("AIO control API listening on {}:{}",
            util::trim(impl_->config.control.bind_host),
            impl_->control_config.listen_port);
    return true;
}

int AioHost::run_tui() {
    if (!impl_->started || !impl_->ui) return 1;
    impl_->ui->run();
    return impl_->listener_failed ? 1 : 0;
}

int AioHost::run_cli(bool json_output) {
    if (!impl_->started) return 1;

    HttpClient client(control_client_base_url(
        impl_->config.control.bind_host, impl_->control_config.listen_port));
    if (!impl_->control_config.external_api_token.empty()) {
        client.set_bearer_token(impl_->control_config.external_api_token);
    }
    client.set_timeouts(10, 3600, 3600);

    std::unique_ptr<HttpClient> openai_client;
    if (impl_->control_config.openai_compat_port != 0) {
        openai_client = std::make_unique<HttpClient>(control_client_base_url(
            impl_->config.control.bind_host,
            impl_->control_config.openai_compat_port));
        if (!impl_->control_config.external_api_token.empty()) {
            openai_client->set_bearer_token(
                impl_->control_config.external_api_token);
        }
        openai_client->set_timeouts(10, 3600, 3600);
    }

    auto request_chat = [&](const std::string& agent_id,
                            const std::string& message,
                            const std::string& conversation_id = std::string{}) {
        HttpResponse response;
        if (openai_client) {
            nlohmann::json messages = nlohmann::json::array();
            messages.push_back({{"role", "user"}, {"content", message}});
            nlohmann::json body{
                {"model", "agent:" + agent_id},
                {"messages", std::move(messages)},
                {"stream", false},
            };
            if (!conversation_id.empty()) body["conversation_id"] = conversation_id;
            return openai_client->post("/v1/chat/completions", body);
        }

        const auto result = impl_->control_host->api().chat_local(
            agent_id, message, conversation_id);
        if (!result.success) {
            response.status = 500;
            response.body = nlohmann::json{{
                "error", result.error.empty()
                    ? "chat completion failed" : result.error}}.dump();
            return response;
        }

        std::string content;
        int completion_tokens = 0;
        for (const auto& chunk : result.chunks) {
            content += chunk.delta_content;
            completion_tokens = std::max(completion_tokens, chunk.tokens_used);
        }
        response.status = 200;
        response.body = nlohmann::json{
            {"id", "chatcmpl-" + util::generate_uuid()},
            {"object", "chat.completion"},
            {"created", util::now_ms() / 1000},
            {"model", "agent:" + agent_id},
            {"choices", nlohmann::json::array({{
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", content},
                }},
                {"finish_reason", "stop"},
            }})},
            {"usage", {
                {"prompt_tokens", 0},
                {"completion_tokens", completion_tokens},
                {"total_tokens", completion_tokens},
            }},
            {"mantic_conversation_id", result.conv_id},
        }.dump();
        return response;
    };

    auto command_error = [](const std::string& message) {
        return HttpResponse{400, nlohmann::json{{"error", message}}.dump()};
    };

    std::cout << "mantic-mind-aio CLI. Type 'help' for commands.\n";
    std::string line;
    while (!impl_->quit_requested) {
        std::cout << "aio> " << std::flush;
        if (!read_cli_line_interruptibly(
                &line, [this] { return impl_->quit_requested.load(); })) {
            break;
        }
        std::vector<std::string> tokens;
        std::string parse_error;
        if (!cli::tokenize_command_line(line, &tokens, &parse_error)) {
            if (json_output) {
                std::cout << nlohmann::json{
                    {"ok", false}, {"error", parse_error}}.dump() << '\n';
            } else {
                std::cerr << "ERROR: " << parse_error << '\n';
            }
            continue;
        }
        if (tokens.empty()) continue;
        const std::string command = util::to_lower(tokens.front());
        if (command == "quit" || command == "exit") break;
        if (command == "help" || command == "?") {
            print_cli_help();
            continue;
        }

        HttpResponse response;
        bool have_response = true;
        bool chat_response = false;
        if (command == "nodes" && tokens.size() == 1) {
            response = client.get("/v1/nodes");
        } else if (command == "nodes" && tokens.size() >= 2) {
            const std::string subcommand = util::to_lower(tokens[1]);
            if (subcommand == "list") {
                response = client.get("/v1/nodes");
            } else if (subcommand == "discovered") {
                response = client.get("/v1/nodes/discovered");
            } else if (subcommand == "add" && tokens.size() >= 4) {
                bool remember = false;
                if (tokens.size() >= 6 &&
                    !cli::parse_bool_token(tokens[5], &remember)) {
                    response = command_error("remember must be true|false");
                } else {
                    response = client.post("/v1/nodes", nlohmann::json{
                        {"url", tokens[2]},
                        {"api_key", tokens[3]},
                        {"platform", tokens.size() >= 5 ? tokens[4] : ""},
                        {"remember", remember},
                    });
                }
            } else if (subcommand == "remove" && tokens.size() == 3) {
                response = client.del("/v1/nodes/" + tokens[2]);
            } else if (subcommand == "forget" && tokens.size() == 3) {
                response = client.post("/v1/nodes/" + tokens[2] + "/forget",
                                       nlohmann::json::object());
            } else if (subcommand == "pair" && tokens.size() >= 4) {
                const std::string pair_action = util::to_lower(tokens[2]);
                if (pair_action == "start" && tokens.size() == 4) {
                    response = client.post("/v1/nodes/pair/start",
                                           nlohmann::json{{"url", tokens[3]}});
                } else if (pair_action == "complete" && tokens.size() >= 6) {
                    bool remember = false;
                    if (tokens.size() >= 7 &&
                        !cli::parse_bool_token(tokens[6], &remember)) {
                        response = command_error("remember must be true|false");
                    } else {
                        response = client.post("/v1/nodes/pair/complete",
                            nlohmann::json{
                                {"url", tokens[3]},
                                {"nonce", tokens[4]},
                                {"pin_or_psk", tokens[5]},
                                {"remember", remember},
                            });
                    }
                } else if (pair_action == "psk") {
                    bool remember = false;
                    if (tokens.size() >= 6 &&
                        !cli::parse_bool_token(tokens[5], &remember)) {
                        response = command_error("remember must be true|false");
                    } else {
                        nlohmann::json body{
                            {"url", tokens[3]}, {"remember", remember}};
                        if (tokens.size() >= 5) {
                            body["psk"] = tokens[4];
                        } else if (!impl_->control_config.pairing_key.empty()) {
                            // The configured cluster key is already available
                            // to the TUI; the AIO CLI uses the same default so
                            // users do not have to retype a secret.
                            body["psk"] = impl_->control_config.pairing_key;
                        }
                        response = client.post("/v1/nodes/pair/psk", body);
                    }
                } else {
                    have_response = false;
                }
            } else {
                have_response = false;
            }
        } else if (command == "models" &&
                   (tokens.size() == 1 ||
                    (tokens.size() == 2 && util::to_lower(tokens[1]) == "list"))) {
            response = client.get("/v1/models");
        } else if (command == "agents" && tokens.size() == 1) {
            response = client.get("/v1/agents");
        } else if (command == "agents" && tokens.size() >= 2) {
            const std::string subcommand = util::to_lower(tokens[1]);
            if (subcommand == "list") {
                response = client.get("/v1/agents");
            } else if (subcommand == "show" && tokens.size() == 3) {
                response = client.get("/v1/agents/" + tokens[2]);
            } else if (subcommand == "create" && tokens.size() >= 3) {
                try {
                    response = client.post(
                        "/v1/agents",
                        nlohmann::json::parse(cli::join_tokens(tokens, 2)));
                } catch (const std::exception& exception) {
                    response = command_error(
                        std::string("invalid JSON: ") + exception.what());
                }
            } else if (subcommand == "update" && tokens.size() >= 4) {
                try {
                    response = client.put(
                        "/v1/agents/" + tokens[2],
                        nlohmann::json::parse(cli::join_tokens(tokens, 3)));
                } catch (const std::exception& exception) {
                    response = command_error(
                        std::string("invalid JSON: ") + exception.what());
                }
            } else if (subcommand == "delete" && tokens.size() == 3) {
                response = client.del("/v1/agents/" + tokens[2]);
            } else {
                have_response = false;
            }
        } else if (command == "chat" && tokens.size() >= 4 &&
                   util::to_lower(tokens[1]) == "send") {
            chat_response = true;
            response = request_chat(
                tokens[2], tokens[3], tokens.size() >= 5 ? tokens[4] : "");
        } else if (command == "chat" && tokens.size() >= 3) {
            chat_response = true;
            response = request_chat(tokens[1], cli::join_tokens(tokens, 2));
        } else if (command == "curation" && tokens.size() >= 3) {
            const std::string group = util::to_lower(tokens[1]);
            const std::string subcommand = util::to_lower(tokens[2]);
            if (group == "conv" && subcommand == "list" && tokens.size() == 4) {
                response = client.get(
                    "/v1/agents/" + tokens[3] + "/conversations");
            } else if (group == "conv" && subcommand == "create" &&
                       tokens.size() >= 5) {
                try {
                    response = client.post(
                        "/v1/agents/" + tokens[3] + "/conversations",
                        nlohmann::json::parse(cli::join_tokens(tokens, 4)));
                } catch (const std::exception& exception) {
                    response = command_error(
                        std::string("invalid JSON: ") + exception.what());
                }
            } else if (group == "conv" &&
                       (subcommand == "activate" || subcommand == "delete" ||
                        subcommand == "compact") && tokens.size() == 5) {
                const std::string path = "/v1/agents/" + tokens[3] +
                    "/conversations/" + tokens[4];
                if (subcommand == "delete") {
                    response = client.del(path);
                } else {
                    response = client.post(path + "/" + subcommand,
                                           nlohmann::json::object());
                }
            } else if (group == "mem" && subcommand == "list" &&
                       tokens.size() == 4) {
                response = client.get(
                    "/v1/agents/" + tokens[3] + "/memories");
            } else if (group == "mem" && subcommand == "delete" &&
                       tokens.size() == 5) {
                response = client.del(
                    "/v1/agents/" + tokens[3] + "/memories/" + tokens[4]);
            } else if (group == "mem" && subcommand == "extract" &&
                       tokens.size() >= 5) {
                try {
                    response = client.post(
                        "/v1/agents/" + tokens[3] + "/memories/extract",
                        nlohmann::json::parse(cli::join_tokens(tokens, 4)));
                } catch (const std::exception& exception) {
                    response = command_error(
                        std::string("invalid JSON: ") + exception.what());
                }
            } else {
                have_response = false;
            }
        } else if (command == "activity" && tokens.size() >= 2 &&
                   util::to_lower(tokens[1]) == "tail") {
            int tail = 20;
            bool valid = true;
            if (tokens.size() >= 3) {
                try {
                    tail = std::max(1, std::stoi(tokens[2]));
                } catch (...) {
                    valid = false;
                }
            }
            if (!valid || tokens.size() > 4) {
                have_response = false;
            } else {
                std::string path = "/v1/activity?tail=" + std::to_string(tail);
                if (tokens.size() == 4) path += "&level=" + tokens[3];
                response = client.get(path);
            }
        } else if (command == "node" && tokens.size() >= 2) {
            const std::string action = util::to_lower(tokens[1]);
            if (action == "status" || action == "slots") {
                const std::string node_id = tokens.size() >= 3 ? tokens[2] : "local";
                response = client.get("/v1/nodes/" + node_id + "/status");
                if (response.ok() && action == "slots") {
                    try {
                        const auto status = nlohmann::json::parse(response.body);
                        response.body = status.value("slots", nlohmann::json::array()).dump();
                    } catch (...) {}
                }
            } else if (action == "logs") {
                const std::string node_id = tokens.size() >= 3 ? tokens[2] : "local";
                const std::string tail = tokens.size() >= 4 ? tokens[3] : "100";
                response = client.get("/v1/nodes/" + node_id + "/logs?tail=" + tail);
            } else if (action == "cancel") {
                const std::string node_id = tokens.size() >= 3 ? tokens[2] : "local";
                response = client.post("/v1/nodes/" + node_id + "/actions/cancel",
                                       nlohmann::json::object());
            } else if (action == "runtime" && tokens.size() >= 3) {
                const std::string runtime_action = util::to_lower(tokens[2]);
                nlohmann::json body{{"allow_network", impl_->allow_network}};
                if (runtime_action == "status") {
                    const std::string node_id = tokens.size() >= 4 ? tokens[3] : "local";
                    response = client.get("/v1/nodes/" + node_id + "/runtime/llama");
                } else if (runtime_action == "provision" ||
                           runtime_action == "check-update" ||
                           runtime_action == "diagnose") {
                    const std::string node_id = tokens.size() >= 4 ? tokens[3] : "local";
                    response = client.post("/v1/nodes/" + node_id +
                        "/runtime/llama/" + runtime_action, body);
                } else if (runtime_action == "update") {
                    std::string node_id = "local";
                    std::string accelerator;
                    bool node_id_set = false;
                    bool valid = true;
                    for (std::size_t index = 3; index < tokens.size(); ++index) {
                        if (tokens[index] == "--accelerator") {
                            if (++index >= tokens.size() || !accelerator.empty()) {
                                valid = false;
                                break;
                            }
                            accelerator = tokens[index];
                        } else if (tokens[index].rfind("--accelerator=", 0) == 0) {
                            if (!accelerator.empty()) {
                                valid = false;
                                break;
                            }
                            accelerator = tokens[index].substr(14);
                            if (accelerator.empty()) {
                                valid = false;
                                break;
                            }
                        } else if (!node_id_set) {
                            node_id = tokens[index];
                            node_id_set = true;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) {
                        have_response = false;
                    } else {
                        if (!accelerator.empty()) body["accelerator"] = accelerator;
                        response = client.post("/v1/nodes/" + node_id +
                                               "/runtime/llama/update", body);
                    }
                } else if (runtime_action == "switch" && tokens.size() >= 4) {
                    body["variant"] = tokens[3];
                    const std::string node_id = tokens.size() >= 5 ? tokens[4] : "local";
                    response = client.post("/v1/nodes/" + node_id +
                                           "/runtime/llama/switch", body);
                } else if (runtime_action == "recover" && tokens.size() >= 4) {
                    const std::string recovery_action = util::to_lower(tokens[3]);
                    std::string node_id = "local";
                    if (recovery_action == "release") {
                        if (tokens.size() < 5 || tokens.size() > 6) {
                            have_response = false;
                        } else {
                            body["action"] = recovery_action;
                            body["variant"] = tokens[4];
                            if (tokens.size() == 6) node_id = tokens[5];
                        }
                    } else if (recovery_action == "retry" ||
                               recovery_action == "target" ||
                               recovery_action == "compile-anyway") {
                        if (tokens.size() > 5) {
                            have_response = false;
                        } else {
                            body["action"] = recovery_action;
                            if (tokens.size() == 5) node_id = tokens[4];
                        }
                    } else {
                        have_response = false;
                    }
                    if (have_response) {
                        response = client.post("/v1/nodes/" + node_id +
                                               "/runtime/llama/recover", body);
                    }
                } else {
                    have_response = false;
                }
            } else {
                have_response = false;
            }
        } else {
            have_response = false;
        }

        if (!have_response) {
            if (json_output) {
                std::cout << nlohmann::json{
                    {"ok", false},
                    {"error", "unknown or incomplete command"},
                }.dump() << '\n';
            } else {
                std::cerr << "Unknown or incomplete command. Type 'help'.\n";
            }
            continue;
        }
        if (!response.ok()) {
            if (json_output) {
                std::cout << nlohmann::json{
                    {"ok", false},
                    {"status", response.status},
                    {"error", response_error(response)},
                }.dump() << '\n';
            } else {
                std::cerr << "ERROR: " << response_error(response) << '\n';
            }
            continue;
        }
        if (chat_response && !json_output) {
            try {
                const auto result = nlohmann::json::parse(response.body);
                const auto& content = result.at("choices").at(0)
                    .at("message").at("content");
                std::cout << (content.is_string() ? content.get<std::string>()
                                                  : content.dump())
                          << '\n';
            } catch (...) {
                std::cout << response.body << '\n';
            }
        } else if (json_output) {
            try {
                std::cout << nlohmann::json::parse(response.body).dump() << '\n';
            } catch (...) {
                std::cout << nlohmann::json{{"raw", response.body}}.dump() << '\n';
            }
        } else {
            try {
                std::cout << nlohmann::json::parse(response.body).dump(2) << '\n';
            } catch (...) {
                std::cout << response.body << '\n';
            }
        }
    }
    return impl_->listener_failed ? 1 : 0;
}

void AioHost::request_quit() {
    impl_->quit_requested = true;
    impl_->request_work_cancellation();
    if (impl_->ui) impl_->ui->quit();
}

void AioHost::stop() { impl_->stop_all(); }

} // namespace mm
