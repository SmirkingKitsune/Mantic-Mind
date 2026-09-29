#pragma once

// Mantic-Mind — the served-model catalog: what users may pick.
//
// Two sources, one list:
//
//   explicit   rows an operator wrote (control.db `served_model`): llama.cpp
//              GGUF models, remote-API models, and overrides of admitted ones
//   admitted   every admission whose verdict Soma can serve, COMPUTED on each
//              read from the admission registry and never stored
//
// Computed rather than copied because a copy has to be kept in sync, and every
// way of keeping it in sync has a failure mode: an admission deleted while its
// row survives, a reprofile that recreates a row the operator deleted. A list
// derived on read cannot disagree with what it is derived from.
//
// ── The rules (compute_catalog, pure) ─────────────────────────────────────────
//
//   1. Every explicit row is an entry, disabled ones included (a conversation
//      that names a disabled model must fail as `model_unavailable`, not as an
//      unknown id).
//   2. An admitted model is an entry when its verdict selects Soma and no
//      explicit row resolves to it. An explicit row that resolves to it is the
//      operator's override and SHADOWS it.
//   3. An admitted entry's model_path is the admission's arch_hash, because
//      resolve() matches a hash exactly while a name is ambiguous across
//      quantizations; and its backend_override is `soma`, because under a
//      cluster order of [llama-cpp, soma] `auto` would try llama.cpp first and
//      hand it a container directory, which llama.cpp cannot load.
//   4. Explicit rows win id collisions; an admitted entry that collides gets
//      `-<arch_hash[:8]>` appended.
//
// Editing an admitted entry (update()) writes an explicit row under the SAME
// id, so conversations that named it keep working. An admitted entry cannot be
// removed — the admission is its source; disable it instead.

#include "common/served_model.hpp"
#include "control/model_registry.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

enum class CatalogSource : std::uint8_t { Explicit, Admitted };
const char* to_string(CatalogSource source) noexcept;

struct CatalogEntry {
    ServedModel model;
    CatalogSource source = CatalogSource::Explicit;

    /// Whether a request naming this entry can be served, as far as control
    /// can tell WITHOUT placing it. False for disabled entries and for admitted
    /// entries on a cluster whose engine policy has no Soma.
    bool available = true;
    std::string unavailable_reason;

    /// The admission this entry resolves to, 0 for none. Always set for an
    /// admitted entry; set for an explicit one whose model_path resolves.
    std::int64_t admission_id = 0;
    std::string arch_hash;
};

using AdmissionResolver =
    std::function<std::optional<AdmittedModel>(const std::string& model_ref)>;

/// The rules above, and nothing else. Sorted by id.
std::vector<CatalogEntry> compute_catalog(const std::vector<ServedModel>& explicit_rows,
                                          const std::vector<AdmittedModel>& admitted,
                                          const AdmissionResolver& resolve,
                                          bool soma_configured);

/// The id an admitted model is listed under: a slug of its container
/// directory's name. That name is admission's own collision key
/// (admission_variant()), so two admissions never share it.
std::string admitted_entry_id(const AdmittedModel& model);

/// The served model an admitted entry presents — Soma-pinned, addressed by
/// hash, default settings, no vision and no tools (Soma supports neither).
ServedModel admitted_served_model(const AdmittedModel& model, const ServedModelId& id);

class ServedModelCatalog {
public:
    explicit ServedModelCatalog(ControlModelRegistry& registry);

    /// Whether the cluster's engine policy includes Soma. Unset answers true:
    /// an embedding with no engine policy has no reason to hide admissions.
    using SomaConfiguredFn = std::function<bool()>;
    void set_soma_configured(SomaConfiguredFn fn);

    std::vector<CatalogEntry> list() const;
    std::optional<CatalogEntry> find(const ServedModelId& id) const;

    enum class PutResult : std::uint8_t { Created, Updated, NotFound, Conflict, Failed };

    /// A new explicit row. `Conflict` when any entry — explicit or admitted —
    /// already has the id. The caller validates first.
    PutResult create(ServedModel model, std::string& out_error);

    /// Replace entry `id`. An explicit row is replaced; an admitted entry gains
    /// an explicit row under the same id. `NotFound` for an unknown id.
    /// `model.id` is forced to `id`.
    PutResult update(const ServedModelId& id, ServedModel model, std::string& out_error);

    enum class RemoveResult : std::uint8_t { Removed, NotFound, NotExplicit, Failed };
    RemoveResult remove(const ServedModelId& id, std::string& out_error);

    /// A free id derived from `hint`: its slug, or the slug with `-2`, `-3`, …
    /// Empty when `hint` has nothing usable in it.
    std::string unique_id(const std::string& hint) const;

private:
    ControlModelRegistry& registry_;
    SomaConfiguredFn soma_configured_;

    /// API keys, in process memory only — the same contract agents had. An
    /// update that carries no key keeps the one already held, because every
    /// editor that shows a model (TUI, CLI, UI) is shown one without its key.
    mutable std::mutex keys_mu_;
    std::unordered_map<ServedModelId, std::string> api_keys_;

    void fill_key(ServedModel& model) const;
    void remember_key(const ServedModel& model);
};

// ── Which model answers a turn ────────────────────────────────────────────────

/// The model a user's turn runs on, chosen in this order: the model the
/// request names, else the conversation's, else the user's default. A model
/// named by the REQUEST sticks — the caller saves it on the conversation when
/// `sticky` — so switching models is one request, not a separate update.
struct TurnModel {
    enum class Problem : std::uint8_t {
        None,
        Required,    ///< nothing names a model: 400 model_required
        NotFound,    ///< no catalog entry has the id: 404 model_not_found
        Unavailable, ///< disabled, or its engine is not configured: 409 model_unavailable
    };
    Problem problem = Problem::None;
    /// The id chosen, even when it is the problem.
    std::string model_id;
    std::optional<CatalogEntry> entry;
    /// The request named a model the conversation does not already have.
    bool sticky = false;
    std::string error;
};

using CatalogLookup = std::function<std::optional<CatalogEntry>(const std::string& id)>;

/// Pure given `find`. Ids are compared after trimming; the catalog's ids are
/// lowercase, so a request's is lowercased before the lookup.
TurnModel resolve_turn_model(const std::string& request_model,
                             const std::string& conversation_model,
                             const std::string& user_default,
                             const CatalogLookup& find);

/// "model_required", "model_not_found", "model_unavailable"; "" for None.
const char* to_string(TurnModel::Problem problem);

} // namespace mm
