#pragma once

#include <string>
#include <functional>

namespace mm {

// Helper for writing SSE events into a cpp-httplib content-provider callback.
// Usage:
//   SseEmitter sse(sink_ref);
//   sse.send(R"({"content":"hello"})");
//   sse.done();
class SseEmitter {
public:
    // sink must remain valid for the lifetime of SseEmitter.
    // write_fn is called with each serialised SSE frame.
    using WriteFn = std::function<bool(const char* data, size_t size)>;
    explicit SseEmitter(WriteFn write_fn);

    // Send:  data: <json_payload>\n\n
    bool send(const std::string& json_payload);

    // Send raw SSE data line (already formatted JSON).
    bool send_raw(const std::string& data_line);

    // Send "data: [DONE]\n\n"
    bool done();

private:
    WriteFn write_fn_;
};

} // namespace mm
