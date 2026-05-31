#include "common/agent.hpp"
#include "common/agent_db.hpp"

namespace mm {

Agent::Agent(const AgentConfig& config, const std::string& data_dir)
    : config_(config)
    , db_(std::make_unique<AgentDB>(config.id, data_dir))
{}

Agent::~Agent() = default;

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

} // namespace mm
