#pragma once

#include "common/models.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mm {

// Return basename(model_ref) with surrounding quotes/whitespace removed.
std::string canonical_model_filename(const std::string& model_ref);

// Restrictive filename validation for model file operations.
// Accepts only plain .gguf filenames (no path separators / traversal).
bool is_safe_model_filename(const std::string& filename);

// Expand shard pattern:
//   model-00001-of-00003.gguf -> { ...00001..., ...00002..., ...00003... }
// Non-sharded inputs return {filename}.
std::vector<std::string> expand_model_shards(const std::string& filename);

// List GGUF files under models_dir recursively with hashes/sizes/shard metadata.
std::vector<StoredModel> list_models_in_dir(const std::string& models_dir);

// Inspect one GGUF file path and return catalog metadata keyed by basename.
// Set include_hash=false for lightweight inventory scans.
std::optional<StoredModel> inspect_model_file(const std::string& model_file_path,
                                              bool include_hash = true);

// Lookup one model by canonical filename in models_dir.
std::optional<StoredModel> find_model_in_dir(const std::string& models_dir,
                                             const std::string& filename);

} // namespace mm
