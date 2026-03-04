#include "node/singleton_lock.hpp"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/file.h>
#  include <unistd.h>
#  include <fcntl.h>
#endif

namespace mm {

// ── Platform-specific impl ──────────────────────────────────────────────────

struct SingletonLock::Impl {
#ifdef _WIN32
    HANDLE mutex_handle = nullptr;
#else
    int    lock_fd      = -1;
    std::string lock_path;
#endif

    ~Impl() {
#ifdef _WIN32
        if (mutex_handle) {
            ReleaseMutex(mutex_handle);
            CloseHandle(mutex_handle);
        }
#else
        if (lock_fd >= 0) {
            flock(lock_fd, LOCK_UN);
            close(lock_fd);
        }
#endif
    }
};

SingletonLock::~SingletonLock() = default;

std::unique_ptr<SingletonLock> SingletonLock::try_acquire(
    const std::string& lock_name)
{
    auto impl = std::make_unique<Impl>();

#ifdef _WIN32
    std::string mutex_name = "Global\\" + lock_name;
    impl->mutex_handle = CreateMutexA(nullptr, TRUE, mutex_name.c_str());
    if (!impl->mutex_handle || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (impl->mutex_handle) {
            CloseHandle(impl->mutex_handle);
            impl->mutex_handle = nullptr;
        }
        return nullptr;
    }
#else
    impl->lock_path = "/tmp/" + lock_name + ".lock";
    impl->lock_fd = open(impl->lock_path.c_str(),
                         O_CREAT | O_RDWR, 0600);
    if (impl->lock_fd < 0)
        return nullptr;

    if (flock(impl->lock_fd, LOCK_EX | LOCK_NB) != 0) {
        close(impl->lock_fd);
        impl->lock_fd = -1;
        return nullptr;
    }
#endif

    auto lock = std::unique_ptr<SingletonLock>(new SingletonLock());
    lock->impl_ = std::move(impl);
    return lock;
}

} // namespace mm
