#include "control/performance_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mm {

void to_json(nlohmann::json& json, const PerformanceSample& sample) {
    json = {
        {"request_id", sample.request_id},
        {"agent_id", sample.agent_id},
        {"node_id", sample.node_id},
        {"backend", sample.backend},
        {"model", sample.model},
        {"started_at_ms", sample.started_at_ms},
        {"queue_ms", sample.queue_ms},
        {"time_to_first_token_ms", sample.time_to_first_token_ms},
        {"total_ms", sample.total_ms},
        {"input_tokens", sample.input_tokens},
        {"output_tokens", sample.output_tokens},
        {"input_tokens_estimated", sample.input_tokens_estimated},
        {"output_tokens_per_second", sample.output_tokens_per_second},
        {"success", sample.success},
        {"error", sample.error},
    };
}

PerformanceTracker::PerformanceTracker(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void PerformanceTracker::record(PerformanceSample sample) {
    if (sample.output_tokens_per_second <= 0.0 && sample.output_tokens > 0 &&
        sample.total_ms > std::max<int64_t>(0, sample.time_to_first_token_ms)) {
        const int64_t generation_ms = sample.time_to_first_token_ms >= 0
            ? sample.total_ms - sample.time_to_first_token_ms : sample.total_ms;
        if (generation_ms > 0) {
            sample.output_tokens_per_second =
                static_cast<double>(sample.output_tokens) * 1000.0 /
                static_cast<double>(generation_ms);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    while (samples_.size() >= capacity_) samples_.pop_front();
    samples_.push_back(std::move(sample));
}

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = p * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    if (low == high) return values[low];
    const double fraction = position - static_cast<double>(low);
    return values[low] + (values[high] - values[low]) * fraction;
}

nlohmann::json aggregate(const std::vector<PerformanceSample>& samples) {
    std::vector<double> total_ms;
    std::vector<double> ttft_ms;
    std::vector<double> token_rate;
    int success = 0;
    int input_tokens = 0;
    int output_tokens = 0;
    for (const auto& sample : samples) {
        if (sample.success) ++success;
        if (sample.total_ms >= 0) total_ms.push_back(static_cast<double>(sample.total_ms));
        if (sample.time_to_first_token_ms >= 0)
            ttft_ms.push_back(static_cast<double>(sample.time_to_first_token_ms));
        if (sample.output_tokens_per_second > 0.0)
            token_rate.push_back(sample.output_tokens_per_second);
        input_tokens += sample.input_tokens;
        output_tokens += sample.output_tokens;
    }
    auto stats = [](const std::vector<double>& values) {
        double sum = 0.0;
        for (const double value : values) sum += value;
        return nlohmann::json{
            {"average", values.empty() ? 0.0 : sum / static_cast<double>(values.size())},
            {"p50", percentile(values, 0.50)},
            {"p95", percentile(values, 0.95)},
        };
    };
    return {
        {"requests", samples.size()},
        {"successful", success},
        {"failed", static_cast<int>(samples.size()) - success},
        {"input_tokens", input_tokens},
        {"output_tokens", output_tokens},
        {"total_ms", stats(total_ms)},
        {"time_to_first_token_ms", stats(ttft_ms)},
        {"output_tokens_per_second", stats(token_rate)},
    };
}

}  // namespace

nlohmann::json PerformanceTracker::snapshot(std::size_t tail) const {
    std::vector<PerformanceSample> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t count = std::min(tail, samples_.size());
        copy.assign(samples_.end() - static_cast<std::ptrdiff_t>(count), samples_.end());
    }
    return {{"session", true}, {"aggregate", aggregate(copy)}, {"samples", copy}};
}

void PerformanceTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
}

}  // namespace mm
