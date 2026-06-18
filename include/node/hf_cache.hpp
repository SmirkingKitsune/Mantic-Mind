#pragma once

#include <string>
#include <vector>

namespace mm {

// Hugging Face model-cache awareness for vLLM nodes.
//
// vLLM resolves a model reference as an HF repo id (downloaded to the local HF
// hub cache on first use), a cache hit, or a local directory. These helpers let
// the cluster (a) see which repos a node already has cached, (b) point every
// node at one shared cache (e.g. an NFS export), and (c) pre-fetch a model
// out-of-band so the multi-GB download does not happen inside the load-model
// health-timeout window.
//
// The classification, dirname mapping, cache scan, and argument builder are
// pure (filesystem only for the scan) and unit-tested. The live download exec
// is POSIX-gated; on Windows it refuses cleanly and vLLM downloads at load time.

// Map an HF hub cache directory name ("models--org--name") to its repo id
// ("org/name"). Returns "" when dirname is not a hub model directory.
std::string hf_repo_id_from_cache_dirname(const std::string& dirname);

// List repo ids present in an HF hub cache directory (the "hub" dir holding the
// "models--*" entries). Cheap: a single directory listing, no size walk.
std::vector<std::string> scan_hf_cache_models(const std::string& hub_dir);

// Resolve the HF hub cache directory from explicit config and environment,
// applying HF's own precedence: configured override → HF_HUB_CACHE →
// HF_HOME/hub → default_home/.cache/huggingface/hub. Pure (env values passed
// in) for testability.
std::string resolve_hf_hub_cache_dir(const std::string& configured,
                                     const std::string& hf_hub_cache_env,
                                     const std::string& hf_home_env,
                                     const std::string& home_dir);

// Build the `hf download` argument vector (excludes the executable), targeting
// an explicit cache dir when one is given. Pure.
std::vector<std::string> build_hf_download_args(const std::string& repo,
                                                const std::string& cache_dir);

// True when this platform can run an out-of-band HF pre-fetch (POSIX only).
bool hf_prefetch_supported();

// Pre-fetch a repo into cache_dir via the `hf` CLI. Blocks until done. Returns
// false and sets *error_out on failure or when hf_prefetch_supported() is false.
bool hf_download(const std::string& hf_cli_path,
                 const std::string& repo,
                 const std::string& cache_dir,
                 std::string* error_out);

} // namespace mm
