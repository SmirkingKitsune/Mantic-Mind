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
#include <cstdlib>
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

/// Portable setenv/unsetenv, for the one test that has to flip a probe override.
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr)
        ::unsetenv(name);
    else
        ::setenv(name, value, 1);
#endif
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
            // Clear the role bit ONLY. Zeroing the whole word would also drop
            // kFlagExpertDigests, and the digest table would then read as trailing
            // bytes — a different refusal than the one this case is about.
            put_u32(bytes, 12, u32(bytes, 12) & ~soma::kFlagPerRoleQuant);
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
            // Clear the role bit ONLY. Zeroing the whole word would also drop
            // kFlagExpertDigests, and the digest table would then read as trailing
            // bytes — a different refusal than the one this case is about.
            put_u32(bytes, 12, u32(bytes, 12) & ~soma::kFlagPerRoleQuant);
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


        std::cout << "\ncold bandwidth probe\n";

        // The probe's number feeds the verdict, so what it measures decides
        // whether a host is told streaming is affordable. It used to read through
        // the page cache, which on a container just written by the converter is
        // memcpy: 5793 MB/s against 26 MB/s unbuffered, on one file on one host.
        {
            const auto dir = case_dir(root, fixture, "probe-method");
            soma::ExpertStore store;
            check(store.open(dir.string(), arch).ok(), "probe fixture opens", "");

            std::uint64_t bw = 0;
            soma::BandwidthReport report;
            // Measured on its own line. Folding the call into check()'s condition
            // while the detail argument reads `bw` leaves the two unsequenced, and
            // the detail printed 0 B/s beside a passing check.
            const auto probe_st = store.measure_bandwidth(bw, &report);
            check(probe_st.ok() && bw > 0, "the probe reports a rate",
                  std::to_string(bw) + " B/s");
            check(report.bytes_per_second == bw && report.samples > 0 &&
                      report.bytes_moved > 0,
                  "and a report consistent with it",
                  std::to_string(report.samples) + " samples");

            // The method is a property of the filesystem under the fixture, so
            // this asserts it is NAMED rather than which one it is. Reporting the
            // weakest silently is the failure being guarded against: a caller
            // deriving a verdict has to be able to see that the figure came from a
            // warm cache.
            check(std::string(soma::to_string(report.method)) != "unknown",
                  "and names how it read", soma::to_string(report.method));
        }

        {
            // The fallback, which no filesystem on a given host may exercise.
            // Forced, so the path that runs where O_DIRECT is refused is not
            // shipped untested.
            const auto dir = case_dir(root, fixture, "probe-fallback");
            set_env("SOMA_PROBE_NO_DIRECT", "1");
            soma::ExpertStore store;
            check(store.open(dir.string(), arch).ok(), "fallback fixture opens", "");
            std::uint64_t bw = 0;
            soma::BandwidthReport report;
            const auto st = store.measure_bandwidth(bw, &report);
            set_env("SOMA_PROBE_NO_DIRECT", nullptr);
            check(st.ok() && bw > 0, "the fallback probe still measures", st.message());
            check(report.method != soma::BandwidthMethod::Unbuffered,
                  "and does not claim unbuffered when it was refused",
                  soma::to_string(report.method));
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


        std::cout << "\npayload digests\n";

        // The first section in this file that reads a shard byte. Everything above
        // proves the container is SHAPED right — canonical ranges, exact shard
        // sizes, the roles the IR names — all of which a correctly sized file full
        // of wrong bytes satisfies completely.
        const auto corrupt_one_expert = [&](const fs::path& dir) {
            const auto shard = dir / "experts-00000.bin";
            auto bytes = read_file(shard);
            if (bytes.size() < 64) throw std::runtime_error("shard is too small to damage");
            // Mid-payload and size-preserving, which is what bit rot and a partial
            // copy both look like.
            bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 0xFF);
            write_file(shard, bytes);
        };

        {
            const auto dir = case_dir(root, fixture, "digest-verify-clean");
            soma::PayloadReport report;
            const auto st = soma::verify_payload(dir.string(), report);
            check(st.ok() && report.experts_checked > 0 && report.mismatches == 0,
                  "an undamaged container verifies, and says how much it read",
                  std::to_string(report.experts_checked) + " experts");
        }

        {
            const auto dir = case_dir(root, fixture, "digest-verify-damaged");
            corrupt_one_expert(dir);
            soma::PayloadReport report;
            const auto st = soma::verify_payload(dir.string(), report);
            check(!st.ok() && st.code() == soma::StatusCode::DataCorruption &&
                      report.mismatches == 1,
                  "one flipped byte in a correctly sized shard is caught", st.message());
        }

        {
            // The read path, which is what stands between damaged bytes and the
            // model. Nothing else in this file exercises it.
            const auto dir = case_dir(root, fixture, "digest-read-path");
            corrupt_one_expert(dir);
            soma::ExpertStore store;
            soma::OpenOptions opts;
            check(store.open(dir.string(), arch, opts).ok(),
                  "a damaged container still OPENS — nothing has been read yet", "");
            std::uint64_t bandwidth = 0;
            check(store.measure_bandwidth(bandwidth).ok(),
                  "bandwidth probe reads without certifying payload", "");

            bool found = false;
            std::vector<std::byte> buf(store.header().expert_bytes);
            for (soma::LayerIndex l = 0; l < store.header().n_layers && !found; ++l) {
                for (soma::ExpertId e = 0; e < store.header().n_experts && !found; ++e) {
                    if (store.locate(l, e).length == 0) continue;
                    found = store.read(l, e, buf) == soma::StatusCode::DataCorruption;
                    if (found)
                        check(store.read(l, e, buf) == soma::StatusCode::DataCorruption,
                              "a retry cannot admit a corrupt expert", "");
                }
            }
            check(found, "and the damaged expert is refused at read, not decoded", "");
        }

        {
            const auto dir = case_dir(root, fixture, "digest-trust-policy");
            corrupt_one_expert(dir);
            soma::ExpertStore store;
            soma::OpenOptions opts;
            opts.payload = soma::PayloadPolicy::Trust;
            check(store.open(dir.string(), arch, opts).ok(), "Trust opens the same container", "");
            std::vector<std::byte> buf(store.header().expert_bytes);
            bool any_refused = false;
            for (soma::LayerIndex l = 0; l < store.header().n_layers; ++l)
                for (soma::ExpertId e = 0; e < store.header().n_experts; ++e)
                    if (store.locate(l, e).length != 0 &&
                        store.read(l, e, buf) == soma::StatusCode::DataCorruption)
                        any_refused = true;
            check(!any_refused, "and Trust really does skip the check it opts out of", "");
        }

        {
            const auto dir = case_dir(root, fixture, "digest-stamp-refuses");
            corrupt_one_expert(dir);
            // Force a real stamp rather than the idempotent no-op, so the payload
            // pass actually runs.
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            const auto hash_len = u32(bytes, 16);
            bytes.erase(bytes.begin() + 20, bytes.begin() + 20 + hash_len);
            put_u32(bytes, 16, 0);
            write_file(path, bytes);
            const auto st = soma::stamp_container(dir.string(), arch);
            check(!st.ok() && st.code() == soma::StatusCode::DataCorruption,
                  "stamp will not certify a container whose payload changed", st.message());
        }

        {
            // A container converted before digests existed must keep working. The
            // check is additive, and refusing one would strand every container
            // already on disk for the sake of evidence it never had.
            const auto dir = case_dir(root, fixture, "no-digest-table");
            const auto path = dir / "soma.container";
            auto bytes = read_file(path);
            // n_layers and n_experts sit immediately after the length-prefixed hash.
            const std::size_t after_hash = 20u + u32(bytes, 16);
            const auto slots = static_cast<std::size_t>(u32(bytes, after_hash)) *
                               u32(bytes, after_hash + 4);
            bytes.resize(bytes.size() - static_cast<std::size_t>(slots) * 8);
            put_u32(bytes, 12, u32(bytes, 12) & ~soma::kFlagExpertDigests);
            write_file(path, bytes);
            check(open_with(dir, arch).ok(), "a container with no digest table still opens", "");

            soma::PayloadReport report;
            const auto st = soma::verify_payload(dir.string(), report);
            check(!st.ok() && st.code() == soma::StatusCode::Unsupported,
                  "but verify says it cannot check, rather than reporting success",
                  st.message());
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
