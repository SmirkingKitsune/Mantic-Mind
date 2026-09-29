#pragma once

// Mantic-Mind — the users control knows about, each with its own store under
// {data_dir}/users/{id}.
//
// Ids are immutable and lowercase (util::is_valid_user_id). There is no rename:
// the id names the directory AND keys token bindings, and moving one without
// the other would hand a user's access to whoever took the old name.
//
// Deleting a user is deferred the way deleting an agent was: requests already
// holding the user keep a working store, and the directory is removed when the
// last of them lets go — closing SQLite first, which Windows requires before
// the files can be deleted. Until then the id cannot be reused, so a new user
// can never open a store that is about to be removed from under it.
//
// data/agents/ is never read. If it exists, load_all() says so once.

#include "common/user.hpp"
#include "common/user_db.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

/// A loaded user: the profile, and the store it lives in.
class User {
public:
    User(UserProfile profile, const std::string& data_dir);
    ~User();

    User(const User&) = delete;
    User& operator=(const User&) = delete;

    const UserId& id() const noexcept { return id_; }
    UserProfile profile() const;
    UserDB& db() { return *db_; }
    const UserDB& db() const { return *db_; }

private:
    friend class UserManager;
    void set_profile(UserProfile profile);
    void remove_data_dir_on_destroy() { remove_on_destroy_ = true; }

    UserId id_;
    std::filesystem::path dir_;
    std::unique_ptr<UserDB> db_;
    mutable std::mutex profile_mu_;
    UserProfile profile_;
    bool remove_on_destroy_ = false;
};

class UserManager {
public:
    explicit UserManager(std::string data_dir = "data");
    ~UserManager();

    UserManager(const UserManager&) = delete;
    UserManager& operator=(const UserManager&) = delete;

    /// Load every user under {data_dir}/users. A directory whose name is not a
    /// valid id is skipped with a warning; one whose store has no profile row
    /// (created, then interrupted) is adopted with a bare profile.
    void load_all();

    enum class CreateResult { Created, Invalid, Exists };
    /// `profile.id` must be a valid id not in use — including by a user whose
    /// deletion is still waiting for its last request. Timestamps are set here.
    CreateResult create_user(UserProfile profile, std::string& out_error);

    std::shared_ptr<User> get_user(const UserId& id) const;
    /// Sorted by id.
    std::vector<UserProfile> list_users() const;

    /// Replace the profile; the id cannot change. False when there is no such
    /// user, or `profile.id` is not `id`.
    bool update_profile(const UserId& id, UserProfile profile);

    /// Forget the user now; remove its directory when the last handle drops.
    bool delete_user(const UserId& id);

    /// Whether load_all() found agent data it will not read.
    bool legacy_agent_data_present() const noexcept { return legacy_agent_data_; }

private:
    std::string data_dir_;
    mutable std::mutex mu_;
    std::unordered_map<UserId, std::shared_ptr<User>> users_;
    bool legacy_agent_data_ = false;
};

const char* to_string(UserManager::CreateResult result);

} // namespace mm
