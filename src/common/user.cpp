#include "common/user.hpp"

namespace mm {

void to_json(nlohmann::json& j, const UserProfile& p) {
    j = {{"id", p.id},
         {"display_name", p.display_name},
         {"custom_instructions", p.custom_instructions},
         {"default_model", p.default_model},
         {"memories_enabled", p.memories_enabled},
         {"tools_enabled", p.tools_enabled},
         {"created_at_ms", p.created_at_ms},
         {"updated_at_ms", p.updated_at_ms}};
}

void from_json(const nlohmann::json& j, UserProfile& p) {
    if (!j.is_object()) return;
    const auto text = [&](const char* key, std::string& dst) {
        if (j.contains(key) && j.at(key).is_string()) dst = j.at(key).get<std::string>();
    };
    const auto flag = [&](const char* key, bool& dst) {
        if (j.contains(key) && j.at(key).is_boolean()) dst = j.at(key).get<bool>();
    };
    const auto stamp = [&](const char* key, std::int64_t& dst) {
        if (j.contains(key) && j.at(key).is_number_integer()) dst = j.at(key).get<std::int64_t>();
    };
    text("id", p.id);
    text("display_name", p.display_name);
    text("custom_instructions", p.custom_instructions);
    text("default_model", p.default_model);
    flag("memories_enabled", p.memories_enabled);
    flag("tools_enabled", p.tools_enabled);
    stamp("created_at_ms", p.created_at_ms);
    stamp("updated_at_ms", p.updated_at_ms);
}

bool apply_profile_patch(const nlohmann::json& patch, UserProfile& p, std::string& out_error) {
    if (!patch.is_object()) {
        out_error = "a user profile must be a JSON object";
        return false;
    }
    // Validated in full before anything is applied, so a refused patch leaves
    // the profile exactly as it was.
    UserProfile next = p;
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        const std::string& key = it.key();
        const auto& v = it.value();
        const auto text = [&](std::string& dst) {
            if (!v.is_string()) {
                out_error = key + " must be a string";
                return false;
            }
            dst = v.get<std::string>();
            return true;
        };
        const auto flag = [&](bool& dst) {
            if (!v.is_boolean()) {
                out_error = key + " must be true or false";
                return false;
            }
            dst = v.get<bool>();
            return true;
        };

        bool ok = true;
        if (key == "id") {
            if (!v.is_string() || v.get<std::string>() != p.id) {
                out_error = "id cannot be changed; create a new user instead";
                ok = false;
            }
        } else if (key == "display_name") {
            ok = text(next.display_name);
        } else if (key == "custom_instructions") {
            ok = text(next.custom_instructions);
        } else if (key == "default_model") {
            ok = text(next.default_model);
        } else if (key == "memories_enabled") {
            ok = flag(next.memories_enabled);
        } else if (key == "tools_enabled") {
            ok = flag(next.tools_enabled);
        } else if (key == "created_at_ms" || key == "updated_at_ms") {
            // The server's; ignored rather than refused (see the header).
        } else {
            out_error = key +
                        " is not a user profile field (expected display_name, "
                        "custom_instructions, default_model, memories_enabled or tools_enabled)";
            ok = false;
        }
        if (!ok) return false;
    }
    p = std::move(next);
    return true;
}

} // namespace mm
