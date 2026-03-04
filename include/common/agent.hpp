#pragma once

#include "common/models.hpp"
#include <memory>
#include <string>

namespace mm {

class AgentDB;  // forward declare

// Runtime agent handle — combines AgentConfig with its backing database.
// Owned by AgentManager (control) or loaded transiently in request workers.
class Agent {
public:
    explicit Agent(const AgentConfig& config, const std::string& data_dir = "data");
    ~Agent();

    const AgentId&   get_id()   const;
    const std::string& get_name() const;
    AgentConfig      get_config() const;
    void             update_config(const AgentConfig& cfg);

    AgentDB&       db();
    const AgentDB& db() const;

private:
    AgentConfig              config_;
    std::unique_ptr<AgentDB> db_;
};

} // namespace mm
