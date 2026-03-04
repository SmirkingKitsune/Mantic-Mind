#pragma once

#include <string>
#include <unordered_map>

namespace mm {

// Minimal TOML-subset config file reader.
//
// Supported syntax (one entry per line):
//   # comment
//   key = "string value"
//   key = 1234
//   key = 3.14
//   key = true / false
//
// Section headers ([section]) are parsed but ignored — all keys are flat.
// Env-var override pattern: call load() first, then override with env values.
class ConfigFile {
public:
    // Load from file. Returns false if the file cannot be opened.
    bool load(const std::string& path);

    // True if the key was present in the file.
    bool has(const std::string& key) const;

    // Typed accessors — return default if key not found or conversion fails.
    std::string get(const std::string& key,
                    const std::string& def = {}) const;
    int         get_int(const std::string& key,   int   def = 0)     const;
    float       get_float(const std::string& key, float def = 0.0f)  const;
    bool        get_bool(const std::string& key,  bool  def = false) const;

private:
    std::unordered_map<std::string, std::string> data_;
};

} // namespace mm
