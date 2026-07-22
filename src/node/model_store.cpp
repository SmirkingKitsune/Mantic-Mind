#include "node/model_store.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>

namespace mm {
namespace fs = std::filesystem;

namespace {
constexpr int64_t kBytesPerMiB = 1024 * 1024;
constexpr const char* kJournalName = ".mm-model-store.json";
}  // namespace

ModelStore::ModelStore(std::string root_dir, int64_t min_free_mb)
    : min_free_mb_(min_free_mb) {
    std::error_code ec;
    fs::path abs = fs::absolute(root_dir, ec);
    root_ = ec ? root_dir : abs.lexically_normal().string();
    fs::create_directories(root_, ec);

    std::lock_guard<std::mutex> lock(mutex_);
    load_locked();
}

std::string ModelStore::root() const { return root_; }
int64_t     ModelStore::min_free_mb() const { return min_free_mb_; }

std::string ModelStore::load_path(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return load_path_locked(id);
}

std::optional<ManagedModel> ModelStore::get(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& m : models_)
        if (m.id == id) return m;
    return std::nullopt;
}

std::vector<ManagedModel> ModelStore::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_;
}

int64_t ModelStore::free_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_bytes_locked();
}

int64_t ModelStore::free_bytes_locked() const {
    std::error_code ec;
    const auto info = fs::space(root_, ec);
    // On a transient stat failure report "plenty free" so eviction is SKIPPED
    // rather than deleting the whole cache on a momentary hiccup (a network /
    // removable models_dir that briefly disconnects).
    if (ec) return (std::numeric_limits<int64_t>::max)();
    return static_cast<int64_t>(info.available);
}

std::string ModelStore::journal_path_locked() const {
    return (fs::path(root_) / kJournalName).string();
}

std::string ModelStore::model_dir_locked(const std::string& id) const {
    return (fs::path(root_) / id).lexically_normal().string();
}

std::vector<ManagedModel>::iterator ModelStore::find_locked(const std::string& id) {
    return std::find_if(models_.begin(), models_.end(),
                        [&](const ManagedModel& m) { return m.id == id; });
}

void ModelStore::load_locked() {
    std::ifstream f(journal_path_locked());
    if (!f) return;
    try {
        nlohmann::json j;
        f >> j;
        for (const auto& e : j.value("models", nlohmann::json::array())) {
            ManagedModel m;
            m.id           = e.value("id", std::string{});
            m.size_bytes   = e.value("size_bytes", static_cast<int64_t>(0));
            m.last_used_ms = e.value("last_used_ms", static_cast<int64_t>(0));
            m.pinned       = e.value("pinned", false);
            if (m.id.empty()) continue;
            // Drop journal entries whose files vanished (e.g. manual cleanup).
            std::error_code ec;
            if (!fs::exists(model_dir_locked(m.id), ec)) continue;
            models_.push_back(std::move(m));
        }
    } catch (const std::exception& e) {
        MM_WARN("ModelStore: ignoring unreadable journal {}: {}",
                journal_path_locked(), e.what());
    }
}

void ModelStore::save_locked() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& m : models_) {
        arr.push_back({{"id", m.id},
                       {"size_bytes", m.size_bytes},
                       {"last_used_ms", m.last_used_ms},
                       {"pinned", m.pinned}});
    }
    nlohmann::json j = {{"models", arr}};
    std::ofstream f(journal_path_locked(), std::ios::trunc);
    if (f) f << j.dump(2);
}

int64_t ModelStore::recompute_size_locked(const std::string& id) const {
    int64_t total = 0;
    std::error_code ec;
    const fs::path dir = model_dir_locked(id);
    if (!fs::exists(dir, ec)) return 0;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code fec;
        if (it->is_regular_file(fec))
            total += static_cast<int64_t>(it->file_size(fec));
    }
    return total;
}

std::string ModelStore::load_path_locked(const std::string& id) const {
    std::error_code ec;
    const fs::path dir = model_dir_locked(id);
    if (!fs::exists(dir, ec)) return {};

    fs::path only_file;
    int files = 0;
    int dirs = 0;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::error_code fec;
        if (entry.is_directory(fec)) {
            ++dirs;
        } else if (entry.is_regular_file(fec)) {
            ++files;
            only_file = entry.path();
        }
    }
    // A single-file artifact loads from that file; a multi-file model loads
    // from its directory.
    if (dirs == 0 && files == 1) return only_file.lexically_normal().string();
    return dir.string();
}

bool ModelStore::model_in_use_locked(const std::string& id,
                                     const std::set<std::string>& in_use_paths) const {
    if (in_use_paths.empty()) return false;
    const fs::path mroot = fs::path(model_dir_locked(id));
    for (const auto& raw : in_use_paths) {
        if (raw.empty()) continue;
        const fs::path p = fs::path(raw).lexically_normal();
        if (p == mroot) return true;
        // p under mroot/ ?
        auto mit = mroot.begin();
        auto pit = p.begin();
        bool under = true;
        for (; mit != mroot.end(); ++mit, ++pit) {
            if (pit == p.end() || *pit != *mit) { under = false; break; }
        }
        if (under) return true;
    }
    return false;
}

bool ModelStore::is_reserved_locked(const std::string& id) const {
    const auto it = reserved_.find(id);
    return it != reserved_.end() && it->second > util::now_ms();
}

void ModelStore::reserve(const std::string& id, int64_t grace_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t now = util::now_ms();
    reserved_[id] = now + std::max<int64_t>(0, grace_ms);
    // Opportunistically drop expired reservations so the map cannot grow
    // unbounded across many transfers.
    for (auto it = reserved_.begin(); it != reserved_.end();) {
        if (it->first != id && it->second <= now) it = reserved_.erase(it);
        else ++it;
    }
}

void ModelStore::delete_model_files_locked(const std::string& id) const {
    std::error_code ec;
    fs::remove_all(model_dir_locked(id), ec);
    if (ec)
        MM_WARN("ModelStore: failed to delete model dir for {}: {}", id, ec.message());
}

std::string ModelStore::evict_lru_locked(const std::set<std::string>& in_use_paths) {
    for (auto it = models_.begin(); it != models_.end(); ++it) {
        if (it->pinned) continue;
        if (is_reserved_locked(it->id)) continue;
        if (model_in_use_locked(it->id, in_use_paths)) continue;
        const std::string id = it->id;
        delete_model_files_locked(id);
        models_.erase(it);
        return id;
    }
    return {};
}

std::vector<std::string> ModelStore::make_room_for(
    int64_t incoming_bytes, const std::set<std::string>& in_use_paths) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> evicted;
    const int64_t margin = min_free_mb_ > 0 ? min_free_mb_ * kBytesPerMiB : 0;
    const int64_t target = margin + std::max<int64_t>(0, incoming_bytes);
    while (free_bytes_locked() < target) {
        const std::string id = evict_lru_locked(in_use_paths);
        if (id.empty()) break;
        evicted.push_back(id);
    }
    if (!evicted.empty()) save_locked();
    return evicted;
}

ModelStore::FileSlot ModelStore::begin_file(const std::string& id,
                                            const std::string& rel_within) {
    std::lock_guard<std::mutex> lock(mutex_);
    FileSlot slot;

    if (id.empty() || rel_within.empty()) {
        slot.error = "model id and file path required";
        return slot;
    }

    const fs::path mdir = fs::path(model_dir_locked(id));
    const fs::path final = (mdir / rel_within).lexically_normal();

    // Refuse a rel path that escapes the model directory.
    {
        auto mit = mdir.begin();
        auto fit = final.begin();
        bool under = true;
        for (; mit != mdir.end(); ++mit, ++fit) {
            if (fit == final.end() || *fit != *mit) { under = false; break; }
        }
        if (!under || final == mdir) {
            slot.error = "invalid destination path: " + rel_within;
            return slot;
        }
    }

    std::error_code ec;
    fs::create_directories(final.parent_path(), ec);
    if (ec) {
        slot.error = "failed to create directory: " + ec.message();
        return slot;
    }

    slot.final_path = final.string();
    slot.temp_path  = final.string() + ".part";
    slot.ok = true;
    return slot;
}

bool ModelStore::commit_file(const FileSlot& slot, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    fs::remove(slot.final_path, ec);        // replace any stale copy
    fs::rename(slot.temp_path, slot.final_path, ec);
    if (ec) {
        // Cross-device or racy rename: fall back to copy + remove.
        std::error_code cec;
        fs::copy_file(slot.temp_path, slot.final_path,
                      fs::copy_options::overwrite_existing, cec);
        fs::remove(slot.temp_path, ec);
        if (cec) {
            if (error) *error = "failed to place file: " + cec.message();
            return false;
        }
    }
    return true;
}

void ModelStore::register_model(const std::string& id, bool pinned) {
    std::lock_guard<std::mutex> lock(mutex_);
    ManagedModel m;
    if (auto it = find_locked(id); it != models_.end()) {
        m = *it;
        models_.erase(it);   // move to back (most-recently-used)
    }
    m.id           = id;
    m.pinned       = m.pinned || pinned;   // pin is sticky; never cleared here
    m.size_bytes   = recompute_size_locked(id);
    m.last_used_ms = util::now_ms();
    models_.push_back(std::move(m));
    save_locked();
}

void ModelStore::touch(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = find_locked(id);
    if (it == models_.end()) return;
    ManagedModel m = *it;
    models_.erase(it);
    m.last_used_ms = util::now_ms();
    models_.push_back(std::move(m));
    save_locked();
}

void ModelStore::set_pinned(const std::string& id, bool pinned) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = find_locked(id);
    if (it == models_.end() || it->pinned == pinned) return;
    it->pinned = pinned;
    save_locked();
}

std::vector<std::string> ModelStore::enforce_min_free(
    const std::set<std::string>& in_use_paths) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> evicted;
    if (min_free_mb_ <= 0) return evicted;
    const int64_t target = min_free_mb_ * kBytesPerMiB;
    while (free_bytes_locked() < target) {
        const std::string id = evict_lru_locked(in_use_paths);
        if (id.empty()) break;
        evicted.push_back(id);
    }
    if (!evicted.empty()) save_locked();
    return evicted;
}

std::vector<std::string> ModelStore::clear_unpinned(
    const std::set<std::string>& in_use_paths) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> evicted;
    for (auto it = models_.begin(); it != models_.end();) {
        if (it->pinned || is_reserved_locked(it->id) ||
            model_in_use_locked(it->id, in_use_paths)) {
            ++it;
            continue;
        }
        const std::string id = it->id;
        delete_model_files_locked(id);
        it = models_.erase(it);
        evicted.push_back(id);
    }
    if (!evicted.empty()) save_locked();
    return evicted;
}

}  // namespace mm
