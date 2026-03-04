#include "common/sse_emitter.hpp"
#include <string>

namespace mm {

SseEmitter::SseEmitter(WriteFn write_fn) : write_fn_(std::move(write_fn)) {}

bool SseEmitter::send(const std::string& json_payload) {
    return send_raw(json_payload);
}

bool SseEmitter::send_raw(const std::string& data_line) {
    std::string frame = "data: " + data_line + "\n\n";
    return write_fn_(frame.data(), frame.size());
}

bool SseEmitter::done() {
    static const std::string kDone = "data: [DONE]\n\n";
    return write_fn_(kDone.data(), kDone.size());
}

} // namespace mm
