#include "control/served_model_catalog.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <filesystem>
#include <set>

namespace mm {

const char* to_string(CatalogSource source) noexcept {
    switch (source) {
    case CatalogSource::Explicit:
        return "explicit";
    case CatalogSource::Admitted:
        return "admitted";
    }
    return "unknown";
}

namespace {

std::string container_basename(const AdmittedModel& model) {
    if (model.model_dir.empty()) return {};
    return std::filesystem::path(model.model_dir).filename().string();
}

} // namespace

std::string admitted_entry_id(const AdmittedModel& model) {
    std::string base = container_basename(model);
    if (base.empty()) base = model.name;
    return util::slugify(base);
}

ServedModel admitted_served_model(const AdmittedModel& model, const ServedModelId& id) {
    ServedModel m;
    m.id = id;
    // "Qwen3-30B-A3B (q4_g-q6_g-g128)" reads better in a picker than the
    // directory name, and still says which quantization this is — two
    // admissions of one model at two quants are two entries.
    const std::string base = container_basename(model);
    if (!model.name.empty() && base.size() > model.name.size() + 1 &&
        base.compare(0, model.name.size() + 1, model.name + "-") == 0) {
        m.display_name = model.name + " (" + base.substr(model.name.size() + 1) + ")";
    } else {
        m.display_name = !model.name.empty() ? model.name : base;
    }
    m.model_path = model.arch_hash;
    m.inference_backend = "node";
    m.backend_override = "soma";
    m.tool_calling = false;          // Soma refuses tools (serve.cpp)
    m.vision_settings.enabled = false; // Soma has no vision tower
    m.created_at_ms = model.admitted_at_ms;
    m.updated_at_ms = model.profiled_at_ms;
    return m;
}

std::vector<CatalogEntry> compute_catalog(const std::vector<ServedModel>& explicit_rows,
                                          const std::vector<AdmittedModel>& admitted,
                                          const AdmissionResolver& resolve,
                                          bool soma_configured) {
    std::vector<CatalogEntry> out;
    std::set<std::string> taken;
    std::set<std::int64_t> shadowed;

    for (const auto& row : explicit_rows) {
        CatalogEntry e;
        e.model = row;
        e.source = CatalogSource::Explicit;
        if (resolve && row.node_local()) {
            if (const auto a = resolve(row.model_path)) {
                e.admission_id = a->id;
                e.arch_hash = a->arch_hash;
                shadowed.insert(a->id);
            }
        }
        if (!row.enabled) {
            e.available = false;
            e.unavailable_reason = "disabled by the operator";
        }
        taken.insert(row.id);
        out.push_back(std::move(e));
    }

    for (const auto& a : admitted) {
        if (!verdict_selects_soma(a.verdict)) continue;
        if (shadowed.count(a.id) != 0) continue;

        std::string id = admitted_entry_id(a);
        if (id.empty() || taken.count(id) != 0) {
            const std::string suffix = a.arch_hash.substr(0, std::min<std::size_t>(8, a.arch_hash.size()));
            id = util::slugify((id.empty() ? std::string("admitted") : id) + "-" + suffix);
        }
        if (!util::is_valid_served_model_id(id) || taken.count(id) != 0) {
            // Two admissions whose names AND hash prefixes collide. Listing one
            // under an id the other already answers to would route requests to
            // the wrong weights, so the second is left unlisted and said so.
            MM_WARN("served-model catalog: admission {} ('{}') has no free id; not listed",
                    a.id, a.name);
            continue;
        }

        CatalogEntry e;
        e.model = admitted_served_model(a, id);
        e.source = CatalogSource::Admitted;
        e.admission_id = a.id;
        e.arch_hash = a.arch_hash;
        if (!soma_configured) {
            e.available = false;
            e.unavailable_reason = "the cluster engine policy does not include Soma";
        }
        taken.insert(id);
        out.push_back(std::move(e));
    }

    std::sort(out.begin(), out.end(), [](const CatalogEntry& a, const CatalogEntry& b) {
        return a.model.id < b.model.id;
    });
    return out;
}

ServedModelCatalog::ServedModelCatalog(ControlModelRegistry& registry) : registry_(registry) {}

void ServedModelCatalog::set_soma_configured(SomaConfiguredFn fn) {
    soma_configured_ = std::move(fn);
}

void ServedModelCatalog::fill_key(ServedModel& model) const {
    std::lock_guard<std::mutex> lk(keys_mu_);
    const auto it = api_keys_.find(model.id);
    if (it != api_keys_.end()) model.api_settings.api_key = it->second;
}

void ServedModelCatalog::remember_key(const ServedModel& model) {
    if (model.api_settings.api_key.empty()) return;
    std::lock_guard<std::mutex> lk(keys_mu_);
    api_keys_[model.id] = model.api_settings.api_key;
}

std::vector<CatalogEntry> ServedModelCatalog::list() const {
    const bool soma = soma_configured_ ? soma_configured_() : true;
    auto entries = compute_catalog(
        registry_.list_served_models(), registry_.list(),
        [this](const std::string& ref) { return registry_.resolve(ref); }, soma);
    for (auto& e : entries) fill_key(e.model);
    return entries;
}

std::optional<CatalogEntry> ServedModelCatalog::find(const ServedModelId& id) const {
    if (id.empty()) return std::nullopt;
    // The whole catalog, not just the row: whether `id` names an admitted
    // entry, and whether an explicit row shadows it, are properties of the
    // list rather than of any one row.
    for (auto& e : list()) {
        if (e.model.id == id) return std::move(e);
    }
    return std::nullopt;
}

std::string ServedModelCatalog::unique_id(const std::string& hint) const {
    const std::string base = util::slugify(hint, 120);
    if (base.empty() || !util::is_valid_served_model_id(base)) return {};
    std::set<std::string> taken;
    for (const auto& e : list()) taken.insert(e.model.id);
    if (taken.count(base) == 0) return base;
    for (int n = 2; n < 10000; ++n) {
        const std::string candidate = base + "-" + std::to_string(n);
        if (taken.count(candidate) == 0) return candidate;
    }
    return {};
}

ServedModelCatalog::PutResult ServedModelCatalog::create(ServedModel model,
                                                         std::string& out_error) {
    if (!util::is_valid_served_model_id(model.id)) {
        out_error = "invalid served-model id '" + model.id + "'";
        return PutResult::Failed;
    }
    if (find(model.id)) {
        out_error = "a served model with id '" + model.id + "' already exists";
        return PutResult::Conflict;
    }
    model.created_at_ms = 0; // stamped by the registry
    if (!registry_.upsert_served_model(model, out_error)) return PutResult::Failed;
    remember_key(model);
    return PutResult::Created;
}

ServedModelCatalog::PutResult ServedModelCatalog::update(const ServedModelId& id,
                                                         ServedModel model,
                                                         std::string& out_error) {
    const auto existing = find(id);
    if (!existing) {
        out_error = "no served model with id '" + id + "'";
        return PutResult::NotFound;
    }
    model.id = id;
    // Keep the original creation time for an explicit row; an admitted entry
    // becoming explicit is a new row and gets a new one.
    model.created_at_ms =
        existing->source == CatalogSource::Explicit ? existing->model.created_at_ms : 0;
    if (!registry_.upsert_served_model(model, out_error)) return PutResult::Failed;
    remember_key(model);
    return PutResult::Updated;
}

ServedModelCatalog::RemoveResult ServedModelCatalog::remove(const ServedModelId& id,
                                                            std::string& out_error) {
    const auto existing = find(id);
    if (!existing) {
        out_error = "no served model with id '" + id + "'";
        return RemoveResult::NotFound;
    }
    if (existing->source != CatalogSource::Explicit) {
        out_error = "'" + id +
                    "' is listed because it is admitted, not because a row exists; disable it "
                    "with PUT enabled=false, or remove the admission";
        return RemoveResult::NotExplicit;
    }
    if (!registry_.delete_served_model(id, out_error)) return RemoveResult::Failed;
    std::lock_guard<std::mutex> lk(keys_mu_);
    api_keys_.erase(id);
    return RemoveResult::Removed;
}

// ── Which model answers a turn ────────────────────────────────────────────────

const char* to_string(TurnModel::Problem problem) {
    switch (problem) {
        case TurnModel::Problem::None: return "";
        case TurnModel::Problem::Required: return "model_required";
        case TurnModel::Problem::NotFound: return "model_not_found";
        case TurnModel::Problem::Unavailable: return "model_unavailable";
    }
    return "";
}

TurnModel resolve_turn_model(const std::string& request_model,
                             const std::string& conversation_model,
                             const std::string& user_default,
                             const CatalogLookup& find) {
    TurnModel out;
    const std::string requested = util::to_lower(util::trim(request_model));
    const std::string current = util::trim(conversation_model);
    const std::string fallback = util::trim(user_default);

    if (!requested.empty()) {
        out.model_id = requested;
        out.sticky = requested != current;
    } else if (!current.empty()) {
        out.model_id = current;
    } else if (!fallback.empty()) {
        out.model_id = fallback;
    } else {
        out.problem = TurnModel::Problem::Required;
        out.error = "no model: name one in the request, set one on the conversation, or give "
                    "the user a default_model";
        return out;
    }

    out.entry = find ? find(out.model_id) : std::nullopt;
    if (!out.entry) {
        out.problem = TurnModel::Problem::NotFound;
        out.error = "model not found: " + out.model_id + "; GET /v1/served-models lists them";
        out.sticky = false; // an id that names nothing must not be saved
        return out;
    }
    if (!out.entry->available) {
        out.problem = TurnModel::Problem::Unavailable;
        out.error = "model " + out.model_id + " is unavailable: " + out.entry->unavailable_reason;
        out.sticky = false;
        return out;
    }
    return out;
}

} // namespace mm
