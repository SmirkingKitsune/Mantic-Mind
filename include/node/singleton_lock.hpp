#pragma once

#include <memory>
#include <string>

namespace mm {

/// Prevents multiple mantic-mind node instances from running simultaneously.
/// Windows: named kernel mutex.  Linux: flock() on a lock file.
class SingletonLock {
public:
    ~SingletonLock();

    SingletonLock(const SingletonLock&) = delete;
    SingletonLock& operator=(const SingletonLock&) = delete;

    /// Returns a held lock, or nullptr if another instance already owns it.
    static std::unique_ptr<SingletonLock> try_acquire(
        const std::string& lock_name = "mantic-mind-node");

private:
    SingletonLock() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
