#pragma once

// Mantic-Mind — what makes a served model servable, checked before it is saved.
//
// Checks only what can be known without placing the model: required fields,
// ranges, the vision projector contract, and whether the model reference looks
// like something the engine that will serve it can load. Placement is where the
// rest is learned.

#include "common/models.hpp"
#include "common/served_model.hpp"

#include <string>
#include <vector>

namespace mm {

class NodeRegistry;
class ControlModelRegistry;

/// Validate a served model.
///
/// `admissions` is optional. When it resolves `model_path` to an admission
/// record, the reference is an admitted container rather than a GGUF file, and
/// the GGUF-shaped warnings do not apply — a container directory "not ending in
/// .gguf" is not a problem, it is the point.
///
/// An empty `id` is not checked here: POST generates one before validating.
ValidationResult validate_served_model(const ServedModel& model,
                                       const NodeRegistry* registry,
                                       const std::string& models_dir,
                                       const ControlModelRegistry* admissions = nullptr,
                                       const ModelCapabilityInfo* precomputed_model_info = nullptr);


// Case-insensitive adjacent mmproj-*.gguf candidates for profile-editor hints.
// Candidates are suggestions only; callers must never auto-select one.
std::vector<std::string> suggest_mmproj_files(const std::string& model_path);

} // namespace mm
