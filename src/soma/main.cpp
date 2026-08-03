// Soma — the engine executable.
//
//   soma serve  --model-dir DIR [--port N] [--host H] [--ctx-size N] ...
//   soma plan   --model-dir DIR [--json]
//
// `plan` exists as a subcommand of the same binary rather than a separate tool
// because the planner it runs is the one the server runs: an operator asking
// "what will this do on this host?" and the engine deciding what to do must not
// be able to disagree.

#include "soma/arch_ir.hpp"
#include "soma/plan.hpp"
#include "soma/serve.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

int usage() {
    std::cerr << "usage:\n"
                 "  soma serve --model-dir DIR [--host H] [--port N] [--ctx-size N]\n"
                 "             [--ram-budget BYTES] [--pin BYTES] [--kv-dir DIR]\n"
                 "             [--served-name NAME]\n"
                 "  soma plan  --model-dir DIR [--json]\n";
    return 2;
}

int cmd_plan(int argc, char** argv) {
    std::string dir;
    bool as_json = false;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc)
            dir = argv[++i];
        else if (a == "--json")
            as_json = true;
    }
    if (dir.empty()) return usage();

    // Host budget from the machine this runs on. The verdict is a property of
    // (model, quantization, HOST) — running `plan` on a different box than the
    // one that will serve gives a different and equally correct answer, which
    // is why the registry stores the admission host's verdict and the node
    // re-derives its own.
    soma::HostBudget host;
    host.ram_total_bytes = 16ull << 30;
    host.ram_free_bytes = 8ull << 30;
    host.disk_bandwidth = 1230ull * 1000 * 1000;

    soma::PlanDocument doc;
    // Try the container path first (a converted model carries arch.json), and
    // fall back to adapting the upstream config.json. Both are legitimate inputs:
    // an operator asking "what will this do here?" usually has the HF checkpoint,
    // not a container, and refusing them would make `plan` useless exactly when
    // it is most wanted — before conversion.
    if (auto st = soma::compute_plan(dir, host, doc); !st.ok()) {
        std::string cfg_text;
        soma::ArchIr arch;
        std::ifstream in(std::filesystem::path(dir) / "config.json", std::ios::binary);
        if (!in) {
            std::cerr << "plan failed: " << st.message() << "\n";
            return 1;
        }
        cfg_text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (auto a = soma::adapt_hf_config(cfg_text, arch); !a.ok()) {
            std::cerr << "plan failed: " << a.message() << "\n";
            return 1;
        }
        if (auto p = soma::compute_plan(arch, host, doc); !p.ok()) {
            std::cerr << "plan failed: " << p.message() << "\n";
            return 1;
        }
    }
    if (as_json) {
        std::string js;
        (void)soma::serialize_plan(doc, js);
        std::cout << js << "\n";
    } else {
        std::cout << "verdict      " << soma::to_string(doc.verdict) << "\n"
                  << "reason       " << doc.verdict_reason << "\n"
                  << "routed       " << (doc.total_routed_bytes >> 20) << " MiB\n"
                  << "bytes/token  " << (doc.bytes_per_token >> 20) << " MiB\n"
                  << "max_batch    " << doc.max_batch << "\n";
    }
    return 0;
}

int cmd_serve(int argc, char** argv) {
    soma::ServeConfig cfg;
    if (auto st = soma::parse_serve_config(argc, argv, cfg); !st.ok()) {
        std::cerr << st.message() << "\n";
        return usage();
    }

    soma::ServeServer server;
    if (auto st = server.open(cfg); !st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        return 1;
    }

    // Printed AFTER open() succeeds and the routes are live, so a supervisor
    // that races the log against the port finds the port already accepting.
    // Readiness is still the /health poll — this line is for humans.
    std::cout << "soma serve listening on " << cfg.host << ":" << cfg.port
              << "  model=" << cfg.model_dir
              << "  verdict=" << soma::to_string(server.plan().verdict) << std::endl;

    if (auto st = server.listen(); !st.ok()) {
        std::cerr << st.message() << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "serve") return cmd_serve(argc - 2, argv + 2);
    if (cmd == "plan") return cmd_plan(argc - 2, argv + 2);
    if (cmd == "--help" || cmd == "-h") {
        usage();
        return 0;
    }
    std::cerr << "unknown command '" << cmd << "'\n";
    return usage();
}
