#pragma once

#include "common/models.hpp"
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace mm {

class AgentDB;  // forward declare

// Runtime agent handle — combines AgentConfig with its backing database.
// Owned by AgentManager (control) or loaded transiently in request workers.
class Agent {
public:
    explicit Agent(const AgentConfig& config, const std::string& data_dir = "data");
    ~Agent();

    AgentId          get_id()   const;
    std::string      get_name() const;
    AgentConfig      get_config() const;
    void             update_config(const AgentConfig& cfg);

    AgentDB&       db();
    const AgentDB& db() const;

    void remove_data_dir_on_destroy();

private:
    AgentConfig              config_;
    std::filesystem::path    data_dir_;
    std::unique_ptr<AgentDB> db_;
    bool                     remove_data_dir_on_destroy_ = false;
    mutable std::mutex       config_mutex_;
};

} // namespace mm
