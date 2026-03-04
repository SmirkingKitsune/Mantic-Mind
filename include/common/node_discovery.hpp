#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mm {

struct DiscoveredNode {
    std::string url;
    std::string node_id;
    int64_t     last_seen_ms = 0;
};

// Broadcasts a JSON UDP beacon to 255.255.255.255:port every interval_s seconds.
// Payload: {"service":"mantic-mind","version":1,"url":"...","node_id":"..."}
class NodeDiscoveryBroadcaster {
public:
    NodeDiscoveryBroadcaster() = default;
    ~NodeDiscoveryBroadcaster() { stop(); }

    void start(const std::string& url, const std::string& node_id,
               uint16_t port = 7072, int interval_s = 5);
    void stop();

private:
    std::atomic<bool> running_{false};
    std::thread       thread_;
};

// Listens on a UDP port for NodeDiscoveryBroadcaster beacons.
// Maintains a map of recently-seen nodes (keyed by node_id).
class NodeDiscoveryListener {
public:
    using Callback = std::function<void(const DiscoveredNode&)>;

    NodeDiscoveryListener() = default;
    ~NodeDiscoveryListener() { stop(); }

    void start(uint16_t port = 7072);
    void stop();

    // Returns entries whose last_seen_ms is within the last 30 seconds.
    std::vector<DiscoveredNode> get_nodes() const;

    void set_callback(Callback cb);

private:
    mutable std::mutex                              mutex_;
    std::unordered_map<std::string, DiscoveredNode> nodes_; // keyed by node_id
    Callback                                        callback_;
    std::atomic<bool>                               running_{false};
    std::thread                                     thread_;
};

} // namespace mm
