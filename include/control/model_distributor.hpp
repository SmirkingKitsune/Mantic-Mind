#pragma once

#include "common/models.hpp"
#include <functional>
#include <string>
#include <vector>

namespace mm {

class NodeRegistry;

/// Uploads GGUF model files from the control's models_dir to remote nodes.
class ModelDistributor {
public:
    ModelDistributor(NodeRegistry& registry, std::string models_dir);

    using ProgressCallback = std::function<void(int64_t bytes_sent, int64_t total_bytes)>;

    /// Upload a model file to a remote node. Reads from local models_dir,
    /// sends in 4 MB chunks via POST /api/node/upload-model, verifies SHA-256.
    /// For sharded models, uploads each shard sequentially.
    /// Returns true on success (all shards transferred and verified).
    bool upload_model(const NodeId& node_id,
                      const std::string& model_filename,
                      ProgressCallback progress_cb = {});

    /// Check if a node already has a model stored (by filename match).
    bool node_has_model(const NodeId& node_id,
                        const std::string& model_filename) const;

    /// Delete a model from a remote node.
    bool delete_model_on_node(const NodeId& node_id,
                              const std::string& model_filename);

    /// Detect shard filenames for a sharded model.
    /// E.g. "model-00001-of-00003.gguf" → returns all 3 shard filenames.
    /// For non-sharded models, returns a vector with just the input filename.
    static std::vector<std::string> detect_shards(const std::string& filename);

private:
    NodeRegistry& registry_;
    std::string   models_dir_;

    /// Upload a single file to a node.
    bool upload_single_file(const NodeId& node_id,
                            const std::string& filename,
                            ProgressCallback progress_cb);
};

} // namespace mm
