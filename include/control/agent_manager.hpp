#pragma once

#include "common/models.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <string>

namespace mm {

class Agent;

// Owns all Agent instances for the control process.
// Manages creation, loading, persistence, and deletion.
class AgentManager {
public:
    explicit AgentManager(std::string data_dir = "data");
    ~AgentManager();

    // Load all persisted agents from data_dir on startup.
    void load_all();

    AgentId     create_agent(const AgentConfig& config);
    Agent*      get_agent(const AgentId& id);
    const Agent* get_agent(const AgentId& id) const;
    bool        delete_agent(const AgentId& id);
    std::vector<AgentConfig> list_agents() const;

    // Update config and persist.  If config.id differs from id the agent is
    // renamed (directory moved, map re-keyed).  Returns the final agent ID.
    // Throws std::invalid_argument on bad new ID or std::runtime_error on
    // filesystem failure.  Returns empty string if id is not found.
    AgentId update_agent(const AgentId& id, const AgentConfig& config);

private:
    std::string data_dir_;
    mutable std::mutex mutex_;
    std::unordered_map<AgentId, std::unique_ptr<Agent>> agents_;
};

} // namespace mm
