#pragma once

#include "common/models.hpp"

#include <nlohmann/json.hpp>

#include <string>

namespace mm {

class ModelStorage;
class NodeState;

// Pull GGUF model files from control onto node-local models_dir.
class ModelPuller {
public:
    ModelPuller(ModelStorage& storage,
                NodeState& state,
                std::string control_url);

    // Pull model_filename (or all shards if sharded) from control.
    // out_result includes detailed transfer/verification status.
    bool pull_model(const std::string& model_filename,
                    bool force,
                    nlohmann::json* out_result = nullptr);

private:
    ModelStorage& storage_;
    NodeState&    state_;
    std::string   control_url_;
};

} // namespace mm
