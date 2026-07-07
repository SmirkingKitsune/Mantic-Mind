#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mm {

// One model held in the node-local cache.
struct ManagedModel {
    std::string id;              // stable id (mm::util::model_id_from_ref)
    int64_t     size_bytes  = 0; // total on-disk footprint
    int64_t     last_used_ms = 0;// use-queue recency; larger = more recent
    bool        pinned      = false; // kept across shutdown / disk eviction
};

// Per-node local model cache rooted at models_dir.
//
// Models transferred from control land here as `root/<id>/<files...>`. The
// store keeps a persisted LRU "use queue" (front = least-recently-used); a used
// model moves to the back. Under disk pressure it evicts unpinned models from
// the front, and on shutdown it clears every unpinned model. Pinned models —
// those control marked preferred for this node — are never auto-evicted.
//
// All public methods are thread-safe.
class ModelStore {
public:
    // root_dir is made absolute and created. min_free_mb is the low-water mark
    // on the store's filesystem that triggers eviction (<= 0 disables the
    // proactive threshold; a transfer still evicts enough to fit).
    ModelStore(std::string root_dir, int64_t min_free_mb);

    std::string root() const;            // absolute store root
    int64_t     min_free_mb() const;

    // On-disk path a runtime should load for `id`: the single contained file
    // when the model is exactly one file, otherwise the model directory. ""
    // when the model is not present on disk.
    std::string load_path(const std::string& id) const;

    std::optional<ManagedModel> get(const std::string& id) const;
    std::vector<ManagedModel>   list() const;

    // Free bytes on the store's filesystem right now.
    int64_t free_bytes() const;

    // Evict LRU unpinned models (never those whose files back an in_use load
    // path) until `incoming_bytes` plus the min-free margin fits, or nothing
    // evictable remains. Returns the ids evicted. Call before a transfer.
    std::vector<std::string> make_room_for(int64_t incoming_bytes,
                                           const std::set<std::string>& in_use_paths);

    // Protect `id` from all eviction (make_room_for, the background
    // enforce_min_free, and shutdown clear) for grace_ms from now. Refreshed on
    // every received file so an in-progress multi-file/multi-request transfer is
    // never evicted between files; self-expires if the transfer is abandoned.
    void reserve(const std::string& id, int64_t grace_ms);

    // Destination paths for one incoming file `rel_within` of model `id`.
    // Creates parent directories. Write to temp_path, then commit_file().
    struct FileSlot {
        std::string temp_path;   // write target (root/<id>/<rel>.part)
        std::string final_path;  // rename target (root/<id>/<rel>)
        bool        ok = false;
        std::string error;
    };
    FileSlot begin_file(const std::string& id, const std::string& rel_within);

    // Atomically move a fully-written temp file to its final path.
    bool commit_file(const FileSlot& slot, std::string* error);

    // Record/refresh the model after its files are in place: recompute size
    // from disk, set the pinned flag, move to the back of the use queue, persist.
    void register_model(const std::string& id, bool pinned);

    // Move a model to the back of the use queue (most-recently-used) + persist.
    void touch(const std::string& id);

    // Set/clear the pinned flag + persist.
    void set_pinned(const std::string& id, bool pinned);

    // Evict LRU unpinned, not-in-use models until free space >= min-free.
    // Returns evicted ids. Safe to call periodically.
    std::vector<std::string> enforce_min_free(const std::set<std::string>& in_use_paths);

    // Delete every unpinned, not-in-use model from disk + journal (shutdown
    // cleanup). Returns evicted ids.
    std::vector<std::string> clear_unpinned(const std::set<std::string>& in_use_paths);

private:
    std::string journal_path_locked() const;
    void        load_locked();
    void        save_locked() const;

    std::vector<ManagedModel>::iterator find_locked(const std::string& id);
    std::string model_dir_locked(const std::string& id) const;   // root/<id>
    int64_t     recompute_size_locked(const std::string& id) const;
    std::string load_path_locked(const std::string& id) const;
    bool        model_in_use_locked(const std::string& id,
                                    const std::set<std::string>& in_use_paths) const;
    bool        is_reserved_locked(const std::string& id) const;
    void        delete_model_files_locked(const std::string& id) const;
    int64_t     free_bytes_locked() const;
    // Evict the single least-recently-used unpinned, not-in-use model.
    // Returns its id, or "" when none is evictable.
    std::string evict_lru_locked(const std::set<std::string>& in_use_paths);

    mutable std::mutex        mutex_;
    std::string               root_;
    int64_t                   min_free_mb_ = 0;
    // Ordered use queue: front() = least-recently-used, back() = most-recent.
    std::vector<ManagedModel> models_;
    // id -> epoch-ms until which the model is reserved from eviction. Covers
    // in-flight transfers that span multiple requests (before load-model runs).
    std::map<std::string, int64_t> reserved_;
};

} // namespace mm
