#pragma once

#include "common/models.hpp"

#include <string>

namespace mm {

// Resolve a model path for metadata inspection.
// Tries the provided path first, then models_dir / basename(model_path).
std::string resolve_model_path_for_metadata(const std::string& model_path,
                                            const std::string& models_dir = {});

// Inspect GGUF metadata and filename hints to infer context and capabilities.
// Reads only the GGUF header (no weights), so it is cheap. When the file cannot
// be resolved or is not GGUF, capability fields fall back to filename heuristics.
ModelCapabilityInfo inspect_model_capabilities(const std::string& model_path,
                                               const std::string& models_dir = {});

} // namespace mm
