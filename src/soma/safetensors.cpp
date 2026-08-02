#include "soma/safetensors.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace soma {

namespace {

constexpr std::uint64_t kMaxHeaderBytes = 256ull * 1024 * 1024;

Status read_whole_file(const fs::path& path, std::vector<std::byte>& out) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) {
        return {StatusCode::IoError, "cannot stat " + path.string() + ": " + ec.message()};
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {StatusCode::IoError, "cannot open " + path.string()};
    }
    out.resize(static_cast<std::size_t>(size));
    if (size > 0 &&
        !in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size))) {
        return {StatusCode::IoError, "short read on " + path.string()};
    }
    return {};
}

} // namespace

std::size_t TensorView::elements() const noexcept {
    std::size_t n = 1;
    for (const auto d : shape) {
        if (d < 0) return 0;
        n *= static_cast<std::size_t>(d);
    }
    return shape.empty() ? 0 : n;
}

std::int64_t TensorView::dim(std::size_t i) const noexcept {
    return i < shape.size() ? shape[i] : 0;
}

std::span<const float> TensorView::f32() const noexcept {
    if (dtype != DType::F32) return {};
    return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
}

Status parse_safetensors_dtype(std::string_view text, DType& out) {
    if (text == "F32") {
        out = DType::F32;
        return {};
    }
    if (text == "F16") {
        out = DType::F16;
        return {};
    }
    if (text == "BF16") {
        out = DType::BF16;
        return {};
    }
    return {StatusCode::Unsupported, "unsupported safetensors dtype '" + std::string(text) + "'"};
}

struct SafeTensors::Impl {
    // One buffer per shard, kept alive because TensorView::bytes points into it.
    std::vector<std::vector<std::byte>> buffers;
    std::unordered_map<std::string, TensorView> tensors;
    std::uint64_t bytes_loaded = 0;

    Status ingest(const fs::path& path) {
        buffers.emplace_back();
        auto& buf = buffers.back();
        if (auto st = read_whole_file(path, buf); !st.ok()) return st;

        if (buf.size() < sizeof(std::uint64_t)) {
            return {StatusCode::InvalidArgument, path.string() + ": too small for a header"};
        }

        std::uint64_t header_len = 0;
        std::memcpy(&header_len, buf.data(), sizeof(header_len));
        if (header_len == 0 || header_len > kMaxHeaderBytes ||
            header_len + sizeof(std::uint64_t) > buf.size()) {
            return {StatusCode::InvalidArgument,
                    path.string() + ": implausible header length " + std::to_string(header_len)};
        }

        const char* header_start =
            reinterpret_cast<const char*>(buf.data()) + sizeof(std::uint64_t);
        json header;
        try {
            header = json::parse(header_start, header_start + header_len);
        } catch (const std::exception& e) {
            return {StatusCode::InvalidArgument,
                    path.string() + ": header is not valid JSON: " + e.what()};
        }
        if (!header.is_object()) {
            return {StatusCode::InvalidArgument, path.string() + ": header is not a JSON object"};
        }

        const std::size_t data_base = sizeof(std::uint64_t) + static_cast<std::size_t>(header_len);
        const std::size_t data_size = buf.size() - data_base;

        for (const auto& [name, entry] : header.items()) {
            if (name == "__metadata__") continue;
            if (!entry.is_object()) {
                return {StatusCode::InvalidArgument, path.string() + ": bad entry for " + name};
            }

            TensorView tv;
            tv.name = name;

            const auto dtype_str = entry.value("dtype", std::string{});
            if (auto st = parse_safetensors_dtype(dtype_str, tv.dtype); !st.ok()) {
                return {st.code(), name + ": " + st.message()};
            }

            if (!entry.contains("shape") || !entry["shape"].is_array()) {
                return {StatusCode::InvalidArgument, name + ": missing shape"};
            }
            for (const auto& d : entry["shape"])
                tv.shape.push_back(d.get<std::int64_t>());

            if (!entry.contains("data_offsets") || !entry["data_offsets"].is_array() ||
                entry["data_offsets"].size() != 2) {
                return {StatusCode::InvalidArgument, name + ": missing data_offsets"};
            }
            const auto begin = entry["data_offsets"][0].get<std::uint64_t>();
            const auto end = entry["data_offsets"][1].get<std::uint64_t>();

            if (end < begin || end > data_size) {
                return {StatusCode::InvalidArgument,
                        name + ": data_offsets [" + std::to_string(begin) + "," +
                            std::to_string(end) + ") outside the " + std::to_string(data_size) +
                            "-byte data section"};
            }

            const std::size_t span_bytes = static_cast<std::size_t>(end - begin);
            const std::size_t want = tv.elements() * (dtype_bits(tv.dtype) / 8);
            if (span_bytes != want) {
                return {StatusCode::InvalidArgument,
                        name + ": " + std::to_string(span_bytes) + " bytes for a shape needing " +
                            std::to_string(want)};
            }

            tv.bytes = CByteSpan(buf.data() + data_base + begin, span_bytes);
            bytes_loaded += span_bytes;
            tensors.insert_or_assign(name, std::move(tv));
        }
        return {};
    }
};

SafeTensors::SafeTensors() : impl_(std::make_unique<Impl>()) {}

SafeTensors::~SafeTensors() = default;
SafeTensors::SafeTensors(SafeTensors&&) noexcept = default;
SafeTensors& SafeTensors::operator=(SafeTensors&&) noexcept = default;

Status SafeTensors::open(const std::string& path) {
    close();
    return impl_->ingest(fs::path(path));
}

Status SafeTensors::open_dir(const std::string& dir) {
    close();
    const fs::path root(dir);
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return {StatusCode::NotFound, dir + " is not a directory"};
    }

    const fs::path index = root / "model.safetensors.index.json";
    if (fs::exists(index, ec)) {
        std::vector<std::byte> raw;
        if (auto st = read_whole_file(index, raw); !st.ok()) return st;
        json j;
        try {
            j = json::parse(reinterpret_cast<const char*>(raw.data()),
                            reinterpret_cast<const char*>(raw.data()) + raw.size());
        } catch (const std::exception& e) {
            return {StatusCode::InvalidArgument, index.string() + ": invalid JSON: " + e.what()};
        }
        if (!j.contains("weight_map")) {
            return {StatusCode::InvalidArgument, index.string() + ": no weight_map"};
        }
        // Unique shard names, in sorted order so ingestion is reproducible.
        std::vector<std::string> shards;
        for (const auto& [_, file] : j["weight_map"].items()) {
            auto name = file.get<std::string>();
            if (std::find(shards.begin(), shards.end(), name) == shards.end()) {
                shards.push_back(std::move(name));
            }
        }
        std::sort(shards.begin(), shards.end());
        for (const auto& shard : shards) {
            if (auto st = impl_->ingest(root / shard); !st.ok()) return st;
        }
        return {};
    }

    if (const fs::path single = root / "model.safetensors"; fs::exists(single, ec)) {
        return impl_->ingest(single);
    }

    // A Soma container names its resident half `dense.safetensors` — routed
    // experts live in experts-*.bin, so there is no model.safetensors to find.
    // Accepting both names here is what lets one loader read either an upstream
    // checkpoint or a container (schemas/container.md).
    if (const fs::path dense = root / "dense.safetensors"; fs::exists(dense, ec)) {
        return impl_->ingest(dense);
    }

    return {StatusCode::NotFound,
            dir + ": none of model.safetensors, model.safetensors.index.json, "
                  "or dense.safetensors"};
}

void SafeTensors::close() {
    impl_ = std::make_unique<Impl>();
}

const TensorView* SafeTensors::find(std::string_view name) const noexcept {
    const auto it = impl_->tensors.find(std::string(name));
    return it == impl_->tensors.end() ? nullptr : &it->second;
}

Status SafeTensors::require(std::string_view name, const TensorView*& out) const {
    out = find(name);
    if (out != nullptr) return {};
    return {StatusCode::NotFound, "tensor '" + std::string(name) + "' not in checkpoint"};
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(impl_->tensors.size());
    for (const auto& [name, _] : impl_->tensors)
        out.push_back(name);
    std::sort(out.begin(), out.end());
    return out;
}

std::size_t SafeTensors::size() const noexcept {
    return impl_->tensors.size();
}

std::uint64_t SafeTensors::bytes_loaded() const noexcept {
    return impl_->bytes_loaded;
}

} // namespace soma
