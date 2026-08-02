// Soma — dump the engine's per-layer activations for comparison against the
// transformers reference.
//
// Built because whole-model logit comparison has a resolution limit and G4 hit
// it: four rounds of hypothesise-and-eliminate localised an MLA defect to
// "somewhere positional" and no further, because the only observable was the
// final logits. Every one of those rounds was a guess that had to be coded,
// built and measured. A per-layer diff replaces the guessing with a bisection.
//
// Writes the same container format as tools/admission/dump_activations.py, so
// the two are diffed by one script that does not care which side produced which
// file.
//
// Usage: actdump_g4 <fixture_dir> <out.somaact> [n_positions]

#include "soma/f32_model.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Record {
    std::uint32_t      layer;
    std::string        point;
    std::vector<float> data;
};

struct Collector {
    std::vector<Record> records;
};

void on_emit(void* ctx, std::uint32_t layer, const char* point, const float* data,
             std::size_t n) {
    auto* c = static_cast<Collector*>(ctx);
    Record r;
    r.layer = layer;
    r.point = point;
    r.data.assign(data, data + n);
    c->records.push_back(std::move(r));
}

void put_u32(std::ofstream& o, std::uint32_t v) {
    o.write(reinterpret_cast<const char*>(&v), 4);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: actdump_g4 <fixture_dir> <out.somaact> [n_positions]\n";
        return 2;
    }
    const fs::path dir(argv[1]);
    const fs::path out_path(argv[2]);
    const std::uint32_t n_pos = (argc > 3) ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 8;

    soma::F32Model model;
    if (auto st = soma::load_f32_model(dir.string(), model); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }

    // The oracle's own token ids, so both sides see identical input. Reading
    // them from the fixture rather than inventing a sequence is what makes the
    // two dumps comparable at all.
    std::ifstream oin(dir / "oracle.bin", std::ios::binary);
    if (!oin) {
        std::cerr << "no oracle.bin in " << dir << "\n";
        return 2;
    }
    char magic[8]{};
    oin.read(magic, 8);
    std::uint32_t hdr[5]{};
    oin.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const auto avail = hdr[1];
    std::vector<std::int32_t> ids(avail);
    oin.read(reinterpret_cast<char*>(ids.data()),
             static_cast<std::streamsize>(ids.size() * 4));

    const auto n = std::min(n_pos, avail);
    std::vector<soma::TokenId> toks(ids.begin(), ids.begin() + n);

    Collector collector;
    soma::F32Workspace ws;
    ws.sink.emit = &on_emit;
    ws.sink.ctx = &collector;

    std::vector<float> logits;
    if (auto st = soma::forward_f32(model, toks, ws, logits); !st.ok()) {
        std::cerr << "forward failed: " << st.message() << "\n";
        return 1;
    }

    std::ofstream o(out_path, std::ios::binary | std::ios::trunc);
    if (!o) {
        std::cerr << "cannot write " << out_path << "\n";
        return 2;
    }
    o.write("SOMAACT1", 8);
    put_u32(o, static_cast<std::uint32_t>(collector.records.size() + 1));
    for (const auto& r : collector.records) {
        put_u32(o, r.layer);
        put_u32(o, static_cast<std::uint32_t>(r.point.size()));
        o.write(r.point.data(), static_cast<std::streamsize>(r.point.size()));
        put_u32(o, static_cast<std::uint32_t>(r.data.size()));
        o.write(reinterpret_cast<const char*>(r.data.data()),
                static_cast<std::streamsize>(r.data.size() * 4));
    }
    // Logits last, under a sentinel layer, so a run that diverges only at the
    // head is distinguishable from one that never diverges.
    {
        const std::string p = "logits";
        put_u32(o, 0xFFFFFFFFu);
        put_u32(o, static_cast<std::uint32_t>(p.size()));
        o.write(p.data(), static_cast<std::streamsize>(p.size()));
        put_u32(o, static_cast<std::uint32_t>(logits.size()));
        o.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * 4));
    }

    std::cout << "wrote " << (collector.records.size() + 1) << " records for " << n
              << " positions to " << out_path << "\n";
    return 0;
}
