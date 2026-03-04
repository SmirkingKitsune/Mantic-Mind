#include "common/agent.hpp"
#include "common/agent_db.hpp"

namespace mm {

Agent::Agent(const AgentConfig& config, const std::string& data_dir)
    : config_(config)
    , db_(std::make_unique<AgentDB>(config.id, data_dir))
{}

Agent::~Agent() = default;

const AgentId& Agent::get_id() const { return config_.id; }
const std::string& Agent::get_name() const { return config_.name; }
AgentConfig Agent::get_config() const { return config_; }
void Agent::update_config(const AgentConfig& cfg) { config_ = cfg; }
AgentDB& Agent::db() { return *db_; }
const AgentDB& Agent::db() const { return *db_; }

} // namespace mm
