// Soma container admission boundary: adversarial index mutations.
//
// These cases are deliberately header-only. The dangerous failures all produce
// plausible byte counts and finite decoded values; the reader must reject them
// before a shard byte is interpreted.

#include "soma/expert_store.hpp"
#include "soma/plan.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what;
    if (!detail.empty()) std::cout << " — " << detail;
    std::cout << "\n";
    if (!ok) ++failures;
}

std::vector<char> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::uint32_t u32(const std::vector<char>& bytes, std::size_t at) {
    std::uint32_t v = 0;
    std::memcpy(&v, bytes.data() + at, sizeof(v));
    return v;
}

void put_u32(std::vector<char>& bytes, std::size_t at, std::uint32_t v) {
    std::memcpy(bytes.data() + at, &v, sizeof(v));
}

struct Offsets {
    std::size_t hash = 20;
    std::size_t legacy_dtype = 0;
    std::size_t legacy_group = 0;
    std::size_t descriptor_count = 0;
    std::size_t roles = 0;
};

Offsets offsets(const std::vector<char>& bytes) {
    const auto after_hash = 20u + u32(bytes, 16);
    return {20u, after_hash + 12u, after_hash + 16u, after_hash + 20u, after_hash + 24u};
}

void copy_fixture(const fs::path& source, const fs::path& dest) {
    std::error_code ec;
    fs::create_directories(dest, ec);
    fs::copy(source, dest,
             fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) throw fs::filesystem_error("copy fixture", source, dest, ec);
}

soma::Status open_with(const fs::path& dir,
                       const soma::ArchIr& arch,
                       soma::IdentityPolicy policy = soma::IdentityPolicy::RequireStamped) {
    soma::OpenOptions opts;
    opts.identity = policy;
    soma::ExpertStore store;
    return store.open(dir.string(), arch, opts);
}

fs::path case_dir(const fs::path& root, const fs::path& fixture, const char* name) {
    const auto dir = root / name;
    copy_fixture(fixture, dir);
    return dir;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: container_integrity <v2-container-fixture>\n";
        return 2;
    }

    const fs::path fixture(argv[1]);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
                          ("soma-container-integrity-" + std::to_string(nonce));
    std::error_code cleanup_ec;
    fs::remove_all(root, cleanup_ec);

    try {
        soma::ArchIr arch;
        if (auto st = soma::resolve_arch(fixture.string(), {}, arch); !st.ok()) {
            std::cerr << "fixture IR failed: " << st.message() << "\n";
            return 2;
        }

        std::cout << "strict identity and stamping\n";
        {
            const auto dir = case_dir(root, fixture, "unstamped");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto hash_len = u32(bytes, 16);
            bytes.erase(bytes.begin() + 20, bytes.begin() + 20 + hash_len);
            put_u32(bytes, 16, 0);
            write_file(path, bytes);

            auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("never stamped") != std::string::npos,
                  "strict open rejects an empty on-disk hash", st.message());
            st = open_with(dir, arch, soma::IdentityPolicy::AllowUnstamped);
            check(st.ok(), "--allow-unstamped permits only the empty disk-hash case", st.message());

            auto no_identity = arch;
            no_identity.arch_hash.clear();
            no_identity.container_arch_hash.clear();
            st = open_with(dir, no_identity, soma::IdentityPolicy::AllowUnstamped);
            check(!st.ok(), "the escape does not accept an IR with no expected identity",
                  st.message());

            st = soma::stamp_container(dir.string(), arch);
            check(st.ok(), "stamp fills the hash and replaces the existing index", st.message());
            st = open_with(dir, arch);
            check(st.ok(), "strict open accepts the newly stamped index", st.message());

            const auto once = read_file(path);
            const auto old_time = fs::file_time_type::clock::now() - std::chrono::hours(24);
            fs::last_write_time(path, old_time, cleanup_ec);
            st = soma::stamp_container(dir.string(), arch);
            check(st.ok() && read_file(path) == once, "restamping is byte-for-byte idempotent",
                  st.message());
            if (!cleanup_ec) {
                check(fs::last_write_time(path) == old_time,
                      "an idempotent stamp does not rewrite the index");
            }
        }

        {
            const auto dir = case_dir(root, fixture, "wrong-hash");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            bytes[20] = bytes[20] == '0' ? '1' : '0';
            write_file(path, bytes);
            auto st = open_with(dir, arch, soma::IdentityPolicy::AllowUnstamped);
            check(!st.ok() && st.message().find("does not match") != std::string::npos,
                  "--allow-unstamped never permits a non-empty hash mismatch", st.message());
            st = soma::stamp_container(dir.string(), arch);
            check(!st.ok(), "stamp refuses to overwrite a conflicting prior identity", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "latent-width");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            bytes.erase(bytes.begin() + 20, bytes.begin() + 20 + u32(bytes, 16));
            put_u32(bytes, 16, 0);
            write_file(path, bytes);
            auto latent = arch;
            latent.ffn.routed_expert_hidden = arch.topology.d_model;
            latent.topology.d_model *= 2;
            latent.container_arch_hash.clear();
            check(soma::compute_arch_hash(latent, latent.arch_hash).ok(), "hash latent topology");
            check(soma::stamp_container(dir.string(), latent).ok(), "stamp uses routed latent width");
            check(open_with(dir, latent).ok(), "strict open uses routed latent width");
        }

        for (int mutation = 0; mutation < 4; ++mutation) {
            const auto name = "ranges-" + std::to_string(mutation);
            const auto dir = case_dir(root, fixture, name.c_str());
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            const auto sizes = o.descriptor_count + 40;
            const auto index = sizes + 16;
            if (mutation == 0) put_u32(bytes, index + 12, 0);
            if (mutation == 1) std::memcpy(bytes.data() + index + 16, bytes.data() + index, 16);
            if (mutation == 2) std::memset(bytes.data() + sizes, 0, 8);
            if (mutation == 3) put_u32(bytes, o.legacy_dtype - 4, 0xffffffffu);
            write_file(path, bytes);
            check(!open_with(dir, arch).ok(), "strict open rejects malformed " + name);
            check(!soma::stamp_container(dir.string(), arch).ok(), "stamp rejects malformed " + name);
        }

        std::cout << "per-role quantization\n";
        {
            const auto dir = case_dir(root, fixture, "role-permutation");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            // q6 gate + q4 up + q4 down has the same aggregate bytes as
            // q4 gate + q4 up + q6 down for these equal-element-count roles.
            put_u32(bytes, o.legacy_dtype, 4);      // q6_g
            put_u32(bytes, o.roles + 4, 4);         // gate q6_g
            put_u32(bytes, o.roles + 2 * 12 + 4, 6); // down q4_g
            write_file(path, bytes);
            auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("gate") != std::string::npos,
                  "equal-size gate/down dtype permutation is rejected", st.message());
            st = soma::stamp_container(dir.string(), arch);
            check(!st.ok(), "stamp cannot launder a role permutation", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "group-drift");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            const auto wrong = u32(bytes, o.legacy_group) == 1 ? 2u : 1u;
            put_u32(bytes, o.legacy_group, wrong);
            put_u32(bytes, o.roles + 8, wrong);
            write_file(path, bytes);
            const auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("group") != std::string::npos,
                  "effective-group drift is rejected per role", st.message());
        }

        std::cout << "version and descriptor grammar\n";
        {
            const auto dir = case_dir(root, fixture, "legacy-v1");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            put_u32(bytes, 8, 1);
            write_file(path, bytes);
            auto st = open_with(dir, arch);
            check(!st.ok() && st.code() == soma::StatusCode::VersionMismatch,
                  "strict open rejects v1 even when it has a hash and descriptor", st.message());
            st = open_with(dir, arch, soma::IdentityPolicy::AllowUnstamped);
            check(st.ok(), "the explicit development escape can read transitional v1", st.message());
            st = open_with(dir, arch, soma::IdentityPolicy::SkipArchHash);
            check(!st.ok(), "auxiliary hash policy still requires v2", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "descriptorless-v1");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            put_u32(bytes, 8, 1);
            put_u32(bytes, 12, 0);
            bytes.erase(bytes.begin() + o.descriptor_count,
                        bytes.begin() + o.descriptor_count + 4 + 3 * 12);
            write_file(path, bytes);
            const auto st = soma::stamp_container(dir.string(), arch);
            check(!st.ok() && st.message().find("descriptorless") != std::string::npos,
                  "stamp refuses to manufacture evidence for descriptorless v1", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "descriptorless-v2");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            put_u32(bytes, 12, 0);
            bytes.erase(bytes.begin() + o.descriptor_count,
                        bytes.begin() + o.descriptor_count + 4 + 3 * 12);
            write_file(path, bytes);
            const auto st = open_with(dir, arch, soma::IdentityPolicy::AllowUnstamped);
            check(!st.ok() && st.message().find("mandatory") != std::string::npos,
                  "v2 cannot omit its mandatory role descriptor", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "duplicate-role");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            put_u32(bytes, o.roles + 12, 2); // up entry claims to be gate too
            write_file(path, bytes);
            const auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("duplicate gate") != std::string::npos,
                  "duplicate roles are rejected", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "wide-role-id");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto o = offsets(bytes);
            // TensorRole's underlying type is u8. A cast-before-validation would
            // truncate 258 to 2 and silently accept this as gate.
            put_u32(bytes, o.roles, 258);
            write_file(path, bytes);
            const auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("unknown expert role") != std::string::npos,
                  "wide role ids cannot alias valid u8 enum values", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "unknown-flag");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            put_u32(bytes, 12, u32(bytes, 12) | 0x80000000u);
            write_file(path, bytes);
            const auto st = open_with(dir, arch);
            check(!st.ok() && st.code() == soma::StatusCode::VersionMismatch,
                  "unknown must-understand flags are rejected", st.message());
        }

        {
            const auto dir = case_dir(root, fixture, "trailing-byte");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            bytes.push_back('\0');
            write_file(path, bytes);
            const auto st = open_with(dir, arch);
            check(!st.ok() && st.message().find("trailing") != std::string::npos,
                  "trailing index bytes are rejected", st.message());
        }
    } catch (const std::exception& e) {
        std::cerr << "test setup failed: " << e.what() << "\n";
        ++failures;
    }

    fs::remove_all(root, cleanup_ec);
    std::cout << "\n" << (failures == 0 ? "container integrity PASS" : "container integrity FAIL")
              << "\n";
    return failures == 0 ? 0 : 1;
}
