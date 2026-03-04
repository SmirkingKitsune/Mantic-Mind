#include "control/agent_manager.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"

#include <filesystem>
#include <stdexcept>

namespace mm {

AgentManager::AgentManager(std::string data_dir)
    : data_dir_(std::move(data_dir)) {}

AgentManager::~AgentManager() = default;

void AgentManager::load_all() {
    namespace fs = std::filesystem;
    fs::path agents_dir = fs::path(data_dir_) / "agents";
    if (!fs::exists(agents_dir)) return;

    for (auto& entry : fs::directory_iterator(agents_dir)) {
        if (!entry.is_directory()) continue;
        if (!fs::exists(entry.path() / "agent.db")) continue;

        AgentId agent_id = entry.path().filename().string();
        try {
            AgentDB temp(agent_id, data_dir_);
            AgentConfig cfg = temp.load_config();
            if (cfg.id.empty() || cfg.name.empty()) continue;

            auto agent = std::make_unique<Agent>(cfg, data_dir_);
            std::lock_guard g(mutex_);
            agents_[cfg.id] = std::move(agent);
            MM_INFO("Loaded agent: {} ({})", cfg.name, cfg.id);
        } catch (const std::exception& e) {
            MM_WARN("Failed to load agent from {}: {}", agent_id, e.what());
        }
    }
}

AgentId AgentManager::create_agent(const AgentConfig& config) {
    AgentConfig cfg = config;
    if (cfg.id.empty()) {
        cfg.id = util::generate_uuid();
    } else if (!util::is_valid_agent_id(cfg.id)) {
        throw std::invalid_argument(
            "Invalid agent ID '" + cfg.id + "': must be 1–128 characters, "
            "start with alphanumeric, and contain only [a-zA-Z0-9_-]");
    }

    auto agent = std::make_unique<Agent>(cfg, data_dir_);
    agent->db().save_config(cfg);

    std::lock_guard g(mutex_);
    AgentId id = cfg.id;
    agents_[id] = std::move(agent);
    MM_INFO("Created agent: {} ({})", cfg.name, id);
    return id;
}

Agent* AgentManager::get_agent(const AgentId& id) {
    std::lock_guard g(mutex_);
    auto it = agents_.find(id);
    return it != agents_.end() ? it->second.get() : nullptr;
}

const Agent* AgentManager::get_agent(const AgentId& id) const {
    std::lock_guard g(mutex_);
    auto it = agents_.find(id);
    return it != agents_.end() ? it->second.get() : nullptr;
}

bool AgentManager::delete_agent(const AgentId& id) {
    namespace fs = std::filesystem;
    std::lock_guard g(mutex_);
    if (!agents_.erase(id)) return false;

    // Remove database directory.
    fs::path dir = fs::path(data_dir_) / "agents" / id;
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) MM_WARN("Could not remove agent data dir {}: {}", dir.string(), ec.message());
    return true;
}

std::vector<AgentConfig> AgentManager::list_agents() const {
    std::lock_guard g(mutex_);
    std::vector<AgentConfig> out;
    out.reserve(agents_.size());
    for (auto& [_, a] : agents_) out.push_back(a->get_config());
    return out;
}

AgentId AgentManager::update_agent(const AgentId& id, const AgentConfig& config) {
    namespace fs = std::filesystem;

    // ── Simple update (no ID change) ─────────────────────────────────────────
    if (config.id == id || config.id.empty()) {
        AgentConfig cfg = config;
        cfg.id = id;
        std::lock_guard g(mutex_);
        auto it = agents_.find(id);
        if (it == agents_.end()) return {};
        it->second->update_config(cfg);
        it->second->db().save_config(cfg);
        return id;
    }

    // ── Rename ────────────────────────────────────────────────────────────────
    const AgentId& new_id = config.id;

    if (!util::is_valid_agent_id(new_id))
        throw std::invalid_argument(
            "Invalid agent ID '" + new_id + "': must be 1–128 characters, "
            "start with alphanumeric, and contain only [a-zA-Z0-9_-]");

    std::lock_guard g(mutex_);

    if (agents_.count(new_id))
        throw std::invalid_argument("Agent ID '" + new_id + "' is already in use");

    auto it = agents_.find(id);
    if (it == agents_.end()) return {};

    // Destroy the old Agent (closes its SQLite connection) before renaming.
    agents_.erase(it);

    fs::path old_dir = fs::path(data_dir_) / "agents" / id;
    fs::path new_dir = fs::path(data_dir_) / "agents" / new_id;
    std::error_code ec;
    fs::rename(old_dir, new_dir, ec);
    if (ec)
        throw std::runtime_error("Failed to rename agent directory: " + ec.message());

    auto agent = std::make_unique<Agent>(config, data_dir_);
    agent->db().save_config(config);
    agents_[new_id] = std::move(agent);
    MM_INFO("Renamed agent {} → {}", id, new_id);
    return new_id;
}

} // namespace mm
