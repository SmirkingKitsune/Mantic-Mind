#include "soma/safetensors.hpp"

#include "soma/quant_format.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace soma {

namespace {

constexpr std::uint64_t kMaxHeaderBytes = 256ull * 1024 * 1024;

/// A read-only view of a file's bytes, WITHOUT committing them to the process.
///
/// This used to be `read_whole_file` into a `std::vector<std::byte>`, and for the
/// four-megabyte fixtures that was indistinguishable from what it is now. On
/// GLM-5.2 it was the difference between loading and not: the F32 dense half is
/// 69.3 GiB, `conform` was measured peaking at 73.54 GiB of PRIVATE memory, and
/// the plan for that same container promises a 24 GiB host (roadmap D26).
///
/// Committing was never necessary. Quantized roles read each tensor ONCE, convert
/// it, and never look at the source again; roles that stay F32 — the router, by
/// deliberate design — keep a `WeightRef` pointing at these bytes for the model's
/// lifetime. A mapping serves both: pages arrive on demand, the pages consumed by
/// quantization are clean and reclaimable the moment the OS wants them back, and
/// the handful still referenced stay because they are still referenced.
///
/// What this does NOT do is change what a TensorView is. `bytes` is still a span
/// over a stable address range for the reader's lifetime, so every caller is
/// unaffected — which is the reason this fix is a file-mapping swap rather than a
/// redesign of the loader.
class MappedFile {
public:
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& o) noexcept { swap(o); }

    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) {
            reset();
            swap(o);
        }
        return *this;
    }

    ~MappedFile() { reset(); }

    Status open(const fs::path& path) {
        reset();
        std::error_code ec;
        const auto file_size = fs::file_size(path, ec);
        if (ec) {
            return {StatusCode::IoError, "cannot stat " + path.string() + ": " + ec.message()};
        }
        size_ = static_cast<std::size_t>(file_size);
        if (size_ == 0) return {};

#ifdef _WIN32
        file_ = ::CreateFileW(path.wstring().c_str(),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            return {StatusCode::IoError,
                    "cannot open " + path.string() + ": error " + std::to_string(::GetLastError())};
        }
        mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            const auto err = ::GetLastError();
            reset();
            return {StatusCode::IoError,
                    "cannot map " + path.string() + ": error " + std::to_string(err)};
        }
        data_ = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (data_ == nullptr) {
            const auto err = ::GetLastError();
            reset();
            return {StatusCode::IoError,
                    "cannot view " + path.string() + ": error " + std::to_string(err)};
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return {StatusCode::IoError, "cannot open " + path.string()};
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            reset();
            return {StatusCode::IoError, "cannot map " + path.string()};
        }
        data_ = p;
#endif
        return {};
    }

    const std::byte* data() const noexcept { return static_cast<const std::byte*>(data_); }

    std::size_t size() const noexcept { return size_; }

private:
    void swap(MappedFile& o) noexcept {
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
#ifdef _WIN32
        std::swap(file_, o.file_);
        std::swap(mapping_, o.mapping_);
#else
        std::swap(fd_, o.fd_);
#endif
    }

    void reset() noexcept {
#ifdef _WIN32
        if (data_ != nullptr) ::UnmapViewOfFile(data_);
        if (mapping_ != nullptr) ::CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
        mapping_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data_ != nullptr && size_ > 0) ::munmap(const_cast<void*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }

    const void* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

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
    // One mapping per shard, kept alive because TensorView::bytes points into it.
    // Reallocation is safe: moving a MappedFile transfers the mapping's ADDRESS
    // rather than the bytes, so spans handed out earlier stay valid — the same
    // property the vector-of-vector this replaced relied on.
    std::vector<MappedFile> buffers;
    std::unordered_map<std::string, TensorView> tensors;
    std::uint64_t bytes_loaded = 0;

    Status ingest(const fs::path& path) {
        auto& buf = buffers.emplace_back();
        if (auto st = buf.open(path); !st.ok()) return st;

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

    Status ingest_index(const fs::path& root, const fs::path& index) {
        std::vector<std::byte> raw;
        if (auto st = read_whole_file(index, raw); !st.ok()) return st;
        json j;
        try {
            j = json::parse(reinterpret_cast<const char*>(raw.data()),
                            reinterpret_cast<const char*>(raw.data()) + raw.size());
        } catch (const std::exception& e) {
            return {StatusCode::InvalidArgument, index.string() + ": invalid JSON: " + e.what()};
        }
        if (!j.contains("weight_map") || !j["weight_map"].is_object()) {
            return {StatusCode::InvalidArgument, index.string() + ": no weight_map"};
        }
        std::vector<std::string> shards;
        for (const auto& [_, file] : j["weight_map"].items()) {
            auto name = file.get<std::string>();
            if (std::find(shards.begin(), shards.end(), name) == shards.end())
                shards.push_back(std::move(name));
        }
        std::sort(shards.begin(), shards.end());
        for (const auto& shard : shards)
            if (auto st = ingest(root / shard); !st.ok()) return st;
        return {};
    }

    Status ingest_quant_index(const fs::path& root, const fs::path& index) {
        std::vector<std::byte> raw;
        if (auto st = read_whole_file(index, raw); !st.ok()) return st;
        json j;
        try {
            j = json::parse(reinterpret_cast<const char*>(raw.data()),
                            reinterpret_cast<const char*>(raw.data()) + raw.size());
        } catch (const std::exception& e) {
            return {StatusCode::InvalidArgument, index.string() + ": invalid JSON: " + e.what()};
        }
        if (j.value("format", 0) != 1 || !j.contains("weight_map") ||
            !j["weight_map"].is_object()) {
            return {StatusCode::InvalidArgument,
                    index.string() + ": expected quantized sidecar format 1 and weight_map"};
        }

        std::vector<std::string> files;
        for (const auto& [name, entry] : j["weight_map"].items()) {
            if (!entry.is_object())
                return {StatusCode::InvalidArgument, name + ": quantized entry is not an object"};
            const auto file = entry.value("file", std::string{});
            const fs::path rel(file);
            if (file.empty() || rel.is_absolute() || rel.has_parent_path()) {
                return {StatusCode::InvalidArgument,
                        name + ": quantized shard must be a flat relative filename"};
            }
            if (std::find(files.begin(), files.end(), file) == files.end()) files.push_back(file);
        }
        std::sort(files.begin(), files.end());

        std::unordered_map<std::string, std::size_t> mapped;
        for (const auto& file : files) {
            const auto slot = buffers.size();
            auto& buf = buffers.emplace_back();
            if (auto st = buf.open(root / file); !st.ok()) return st;
            mapped.emplace(file, slot);
        }

        for (const auto& [name, entry] : j["weight_map"].items()) {
            TensorView tv;
            tv.name = name;
            const auto dtype = entry.value("dtype", std::string{});
            if (dtype == "f32")
                tv.dtype = DType::F32;
            else if (dtype == "q8_0")
                tv.dtype = DType::Q8_0;
            else if (dtype == "q4_0")
                tv.dtype = DType::Q4_0;
            else if (dtype == "q4_g")
                tv.dtype = DType::Q4_G;
            else if (dtype == "q6_g")
                tv.dtype = DType::Q6_G;
            else
                return {StatusCode::Unsupported,
                        name + ": unsupported qweight dtype '" + dtype + "'"};
            tv.group = entry.value("group", 0u);
            if (!entry.contains("shape") || !entry["shape"].is_array() ||
                entry["shape"].size() != 2) {
                return {StatusCode::InvalidArgument, name + ": qweight shape must be rank 2"};
            }
            for (const auto& d : entry["shape"])
                tv.shape.push_back(d.get<std::int64_t>());
            if (tv.dim(0) <= 0 || tv.dim(1) <= 0 || tv.group == 0) {
                return {StatusCode::InvalidArgument, name + ": invalid qweight shape/group"};
            }
            const auto file = entry.value("file", std::string{});
            const auto off = entry.value("offset", std::uint64_t{0});
            const auto length = entry.value("length", std::uint64_t{0});
            const auto bit = mapped.find(file);
            if (bit == mapped.end()) return {StatusCode::InvalidArgument, name + ": unknown shard"};
            const auto& buf = buffers[bit->second];
            if (off > buf.size() || length > buf.size() - static_cast<std::size_t>(off)) {
                return {StatusCode::InvalidArgument, name + ": qweight range outside shard"};
            }
            const auto want = quantized_tensor_bytes(tv.dtype,
                                                     static_cast<std::uint32_t>(tv.dim(0)),
                                                     static_cast<std::uint32_t>(tv.dim(1)),
                                                     tv.group);
            if (length != want) {
                return {StatusCode::InvalidArgument,
                        name + ": qweight has " + std::to_string(length) + " bytes; expected " +
                            std::to_string(want)};
            }
            tv.bytes = CByteSpan(buf.data() + off, static_cast<std::size_t>(length));
            bytes_loaded += length;
            if (!tensors.emplace(name, std::move(tv)).second) {
                return {StatusCode::InvalidArgument, name + ": duplicate resident tensor"};
            }
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

    if (const fs::path index = root / "model.safetensors.index.json"; fs::exists(index, ec)) {
        return impl_->ingest_index(root, index);
    }

    if (const fs::path single = root / "model.safetensors"; fs::exists(single, ec)) {
        return impl_->ingest(single);
    }

    // Container residents can be split between lossless metadata/norm tensors
    // and pre-quantized projections. Both indexes are optional independently so
    // v1 dense.safetensors containers remain readable unchanged.
    bool loaded = false;
    if (const fs::path index = root / "dense.safetensors.index.json"; fs::exists(index, ec)) {
        if (auto st = impl_->ingest_index(root, index); !st.ok()) return st;
        loaded = true;
    } else if (const fs::path dense = root / "dense.safetensors"; fs::exists(dense, ec)) {
        if (auto st = impl_->ingest(dense); !st.ok()) return st;
        loaded = true;
    }
    if (const fs::path qindex = root / "dense.qweights.index.json"; fs::exists(qindex, ec)) {
        if (auto st = impl_->ingest_quant_index(root, qindex); !st.ok()) return st;
        loaded = true;
    }
    if (loaded) {
        // Optional DSpark augmentation. These indexes deliberately remain
        // separate on disk so an interrupted augmentation cannot damage the
        // base container; once metadata advertises the capability the loader
        // unions both namespaces into the same immutable tensor table.
        if (const fs::path index = root / "dspark.safetensors.index.json"; fs::exists(index, ec)) {
            if (auto st = impl_->ingest_index(root, index); !st.ok()) return st;
        }
        if (const fs::path qindex = root / "dspark.qweights.index.json"; fs::exists(qindex, ec)) {
            if (auto st = impl_->ingest_quant_index(root, qindex); !st.ok()) return st;
        }
        return {};
    }

    return {StatusCode::NotFound,
            dir + ": none of model.safetensors, model.safetensors.index.json, "
                  "dense.safetensors.index.json, dense.safetensors, or "
                  "dense.qweights.index.json"};
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
