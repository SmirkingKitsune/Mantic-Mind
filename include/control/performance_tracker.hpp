#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace mm {

struct PerformanceSample {
    std::string request_id;
    std::string agent_id;
    std::string node_id;
    std::string backend;
    std::string model;
    int64_t started_at_ms = 0;
    int64_t queue_ms = 0;
    int64_t time_to_first_token_ms = -1;
    int64_t total_ms = 0;
    int input_tokens = 0;
    int output_tokens = 0;
    int image_count = 0;
    int64_t decoded_image_bytes = 0;
    bool vision_routing = false;
    std::string projector_basename;
    bool input_tokens_estimated = true;
    double output_tokens_per_second = 0.0;
    bool success = false;
    std::string error;
};

void to_json(nlohmann::json& json, const PerformanceSample& sample);

// Process-session rolling performance history. It is deliberately bounded and
// non-persistent so metrics remain useful without growing the control database.
class PerformanceTracker {
public:
    explicit PerformanceTracker(std::size_t capacity = 2000);

    void record(PerformanceSample sample);
    nlohmann::json snapshot(std::size_t tail = 200) const;
    void clear();

private:
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<PerformanceSample> samples_;
};

}  // namespace mm
