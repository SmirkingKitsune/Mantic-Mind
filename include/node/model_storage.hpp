#pragma once

#include "common/models.hpp"
#include <string>
#include <vector>
#include <functional>

namespace mm {

/// Manages model files on a node's disk: listing, hashing, upload support.
class ModelStorage {
public:
    explicit ModelStorage(std::string models_dir);

    /// Scan models_dir for .gguf files and return metadata with SHA-256 hashes.
    std::vector<StoredModel> list_models() const;

    /// Check if a model file exists, optionally verifying its hash.
    bool has_model(const std::string& filename,
                   const std::string& expected_sha256 = {}) const;

    /// Full filesystem path for a model filename.
    std::string model_path(const std::string& filename) const;

    /// Delete a model file (and all shards if sharded).
    bool delete_model(const std::string& filename);

    /// Available disk space on the models_dir volume.
    int64_t free_space_mb() const;

    /// Compute SHA-256 of a file using OpenSSL (reads in 1 MB chunks).
    static std::string compute_sha256(const std::string& filepath);

    /// Write raw data at an offset into a file (for chunked uploads).
    bool write_chunk(const std::string& filename,
                     const char* data, size_t size, int64_t offset);

    /// Verify the hash of a fully-uploaded file.
    bool verify_hash(const std::string& filename,
                     const std::string& expected_sha256) const;

private:
    std::string models_dir_;
};

} // namespace mm
