#include "control/user_manager.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <system_error>

namespace mm {

namespace fs = std::filesystem;

// ── User ──────────────────────────────────────────────────────────────────────

User::User(UserProfile profile, const std::string& data_dir)
    : id_(profile.id),
      dir_(fs::path(data_dir) / "users" / profile.id),
      db_(std::make_unique<UserDB>(profile.id, data_dir)),
      profile_(std::move(profile)) {}

User::~User() {
    db_.reset(); // SQLite closed before the files go: Windows will not delete open files
    if (!remove_on_destroy_) return;
    std::error_code ec;
    fs::remove_all(dir_, ec);
    if (ec) MM_WARN("Could not remove the data of deleted user {}: {}", id_, ec.message());
}

UserProfile User::profile() const {
    std::lock_guard<std::mutex> lk(profile_mu_);
    return profile_;
}

void User::set_profile(UserProfile profile) {
    std::lock_guard<std::mutex> lk(profile_mu_);
    profile_ = std::move(profile);
}

// ── UserManager ───────────────────────────────────────────────────────────────

const char* to_string(UserManager::CreateResult result) {
    switch (result) {
        case UserManager::CreateResult::Created: return "created";
        case UserManager::CreateResult::Invalid: return "invalid";
        case UserManager::CreateResult::Exists: return "exists";
    }
    return "unknown";
}

UserManager::UserManager(std::string data_dir) : data_dir_(std::move(data_dir)) {}

UserManager::~UserManager() = default;

void UserManager::load_all() {
    std::error_code ec;
    if (fs::is_directory(fs::path(data_dir_) / "agents", ec)) {
        legacy_agent_data_ = true;
        MM_INFO("{} holds agent data from before users replaced agents; it is not read, "
                "and nothing in it is changed or removed",
                (fs::path(data_dir_) / "agents").string());
    }

    const fs::path root = fs::path(data_dir_) / "users";
    if (!fs::is_directory(root, ec)) return;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        std::error_code entry_ec;
        if (!entry.is_directory(entry_ec)) continue;
        const std::string name = entry.path().filename().string();
        if (!util::is_valid_user_id(name)) {
            MM_WARN("Skipping {}: '{}' is not a valid user id", entry.path().string(), name);
            continue;
        }
        if (!fs::exists(entry.path() / "user.db", entry_ec)) continue;
        try {
            UserProfile profile;
            {
                UserDB store(name, data_dir_);
                if (auto stored = store.load_profile()) {
                    profile = std::move(*stored);
                } else {
                    // Created and interrupted before its profile was written.
                    profile.id = name;
                    profile.display_name = name;
                    store.save_profile(profile);
                    profile = store.load_profile().value_or(profile);
                }
            } // closed before the long-lived handle opens its own connection
            auto user = std::make_shared<User>(std::move(profile), data_dir_);
            std::lock_guard<std::mutex> lk(mu_);
            users_[user->id()] = std::move(user);
        } catch (const std::exception& e) {
            MM_WARN("Failed to load user {}: {}", name, e.what());
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    MM_INFO("Loaded {} user(s)", users_.size());
}

UserManager::CreateResult UserManager::create_user(UserProfile profile, std::string& out_error) {
    if (!util::is_valid_user_id(profile.id)) {
        out_error = "invalid user id '" + profile.id +
                    "': 1-64 characters of [a-z0-9._-], starting and ending with a lowercase "
                    "letter or digit, and not a Windows device name";
        return CreateResult::Invalid;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (users_.count(profile.id)) {
        out_error = "user '" + profile.id + "' already exists";
        return CreateResult::Exists;
    }
    // A directory without a live user is one still being deleted (or left by
    // hand). Opening it would hand the new user the old one's conversations —
    // or have them removed from under it when the last old request finishes.
    std::error_code ec;
    if (fs::exists(fs::path(data_dir_) / "users" / profile.id, ec)) {
        out_error = "the data of a previous user '" + profile.id +
                    "' is still being removed; try again shortly";
        return CreateResult::Exists;
    }
    const std::int64_t now = util::now_ms();
    profile.created_at_ms = now;
    profile.updated_at_ms = now;
    if (profile.display_name.empty()) profile.display_name = profile.id;
    try {
        auto user = std::make_shared<User>(profile, data_dir_);
        user->db().save_profile(profile);
        user->set_profile(user->db().load_profile().value_or(profile));
        users_[profile.id] = std::move(user);
    } catch (const std::exception& e) {
        out_error = std::string("could not create the user's store: ") + e.what();
        return CreateResult::Invalid;
    }
    MM_INFO("Created user {}", profile.id);
    return CreateResult::Created;
}

std::shared_ptr<User> UserManager::get_user(const UserId& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = users_.find(id);
    return it != users_.end() ? it->second : nullptr;
}

std::vector<UserProfile> UserManager::list_users() const {
    std::vector<UserProfile> out;
    {
        std::lock_guard<std::mutex> lk(mu_);
        out.reserve(users_.size());
        for (const auto& [id, user] : users_) out.push_back(user->profile());
    }
    std::sort(out.begin(), out.end(),
              [](const UserProfile& a, const UserProfile& b) { return a.id < b.id; });
    return out;
}

bool UserManager::update_profile(const UserId& id, UserProfile profile) {
    if (profile.id != id) return false;
    std::shared_ptr<User> user = get_user(id);
    if (!user) return false;
    profile.created_at_ms = user->profile().created_at_ms;
    user->db().save_profile(profile);
    user->set_profile(user->db().load_profile().value_or(profile));
    return true;
}

bool UserManager::delete_user(const UserId& id) {
    std::shared_ptr<User> user;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = users_.find(id);
        if (it == users_.end()) return false;
        user = std::move(it->second);
        users_.erase(it);
    }
    user->remove_data_dir_on_destroy();
    MM_INFO("Deleted user {}", id);
    return true;
}

} // namespace mm
