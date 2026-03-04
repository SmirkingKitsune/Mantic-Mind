#include "common/node_discovery.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
   using SockType = SOCKET;
   static constexpr SockType kInvalidSock = INVALID_SOCKET;
   inline static int  sock_close(SockType s) { return closesocket(s); }
   inline static bool sock_valid(SockType s) { return s != INVALID_SOCKET; }
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
   using SockType = int;
   static constexpr SockType kInvalidSock = -1;
   inline static int  sock_close(SockType s) { return ::close(s); }
   inline static bool sock_valid(SockType s) { return s >= 0; }
#endif

namespace mm {

static constexpr int64_t kNodeStalenessMs = 30000; // 30 s

// ── NodeDiscoveryBroadcaster ──────────────────────────────────────────────────

void NodeDiscoveryBroadcaster::start(const std::string& url,
                                      const std::string& node_id,
                                      uint16_t port,
                                      int interval_s) {
    if (running_.exchange(true)) return;

    thread_ = std::thread([this, url, node_id, port, interval_s]() {
#ifdef _WIN32
        WSADATA wsa{};
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        nlohmann::json beacon = {
            {"service", "mantic-mind"},
            {"version", 1},
            {"url",     url},
            {"node_id", node_id}
        };
        std::string payload = beacon.dump();

        while (running_) {
            SockType sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (sock_valid(sock)) {
                int bcast = 1;
                setsockopt(sock, SOL_SOCKET, SO_BROADCAST,
                           reinterpret_cast<const char*>(&bcast), sizeof(bcast));

                sockaddr_in dest{};
                dest.sin_family      = AF_INET;
                dest.sin_port        = htons(port);
                dest.sin_addr.s_addr = INADDR_BROADCAST;

                sendto(sock, payload.data(), static_cast<int>(payload.size()), 0,
                       reinterpret_cast<sockaddr*>(&dest), sizeof(dest));

                sock_close(sock);
            }

            // Sleep interval_s seconds, checking running_ every 100 ms.
            for (int i = 0; i < interval_s * 10 && running_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
#ifdef _WIN32
        WSACleanup();
#endif
    });
}

void NodeDiscoveryBroadcaster::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

// ── NodeDiscoveryListener ─────────────────────────────────────────────────────

void NodeDiscoveryListener::set_callback(Callback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    callback_ = std::move(cb);
}

std::vector<DiscoveredNode> NodeDiscoveryListener::get_nodes() const {
    int64_t cutoff = mm::util::now_ms() - kNodeStalenessMs;
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<DiscoveredNode> out;
    for (auto& [id, node] : nodes_)
        if (node.last_seen_ms >= cutoff)
            out.push_back(node);
    return out;
}

void NodeDiscoveryListener::start(uint16_t port) {
    if (running_.exchange(true)) return;

    thread_ = std::thread([this, port]() {
#ifdef _WIN32
        WSADATA wsa{};
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        SockType sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (!sock_valid(sock)) {
            running_ = false;
#ifdef _WIN32
            WSACleanup();
#endif
            return;
        }

        int reuse = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            sock_close(sock);
            running_ = false;
#ifdef _WIN32
            WSACleanup();
#endif
            return;
        }

        // 1-second receive timeout so we can check running_ periodically.
#ifdef _WIN32
        DWORD tv_ms = 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&tv_ms), sizeof(tv_ms));
#else
        struct timeval tv{1, 0};
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&tv), sizeof(tv));
#endif

        char buf[2048];
        while (running_) {
            sockaddr_in sender{};
#ifdef _WIN32
            int sender_len = sizeof(sender);
#else
            socklen_t sender_len = sizeof(sender);
#endif
            int n = recvfrom(sock, buf, static_cast<int>(sizeof(buf) - 1), 0,
                             reinterpret_cast<sockaddr*>(&sender), &sender_len);
            if (n <= 0) continue;
            buf[n] = '\0';

            try {
                auto j = nlohmann::json::parse(buf, buf + n);
                if (j.value("service", std::string{}) != "mantic-mind") continue;

                DiscoveredNode dn;
                dn.url          = j.value("url",     std::string{});
                dn.node_id      = j.value("node_id", std::string{});
                dn.last_seen_ms = mm::util::now_ms();

                if (dn.url.empty() || dn.node_id.empty()) continue;

                Callback cb;
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    nodes_[dn.node_id] = dn;
                    cb = callback_;
                }
                if (cb) cb(dn);
            } catch (const std::exception& e) {
                MM_DEBUG("NodeDiscoveryListener: parse error: {}", e.what());
            }
        }

        sock_close(sock);
#ifdef _WIN32
        WSACleanup();
#endif
    });
}

void NodeDiscoveryListener::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

} // namespace mm
