#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/logger.hpp"

#include <filesystem>

namespace mm {

Agent::Agent(const AgentConfig& config, const std::string& data_dir)
    : config_(config)
    , data_dir_(std::filesystem::path(data_dir) / "agents" / config.id)
    , db_(std::make_unique<AgentDB>(config.id, data_dir))
{}

Agent::~Agent() {
    db_.reset();

    if (!remove_data_dir_on_destroy_) return;

    std::error_code ec;
    std::filesystem::remove_all(data_dir_, ec);
    if (ec) {
        MM_WARN("Could not remove agent data dir {}: {}", data_dir_.string(), ec.message());
    }
}

AgentId Agent::get_id() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.id;
}

std::string Agent::get_name() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.name;
}

AgentConfig Agent::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

void Agent::update_config(const AgentConfig& cfg) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = cfg;
}
AgentDB& Agent::db() { return *db_; }
const AgentDB& Agent::db() const { return *db_; }

void Agent::remove_data_dir_on_destroy() {
    remove_data_dir_on_destroy_ = true;
}

} // namespace mm
