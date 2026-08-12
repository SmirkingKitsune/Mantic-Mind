# Soma — MoE streaming inference engine.
#
# This directory builds the G0–G4 engine core, its conformance targets, and a
# public-header self-test. The self-test compiles every public header standalone
# (and twice, to exercise the include guard), catching missing includes, cyclic
# dependencies, and broken guards independently of the implementation build.
#
# Repo conventions honoured here: explicit source lists (no file(GLOB) anywhere
# in this project), and the root's global /W4 /WX + -Wall -Wextra -Wpedantic
# -Werror apply unconditionally.

# ── Public headers ────────────────────────────────────────────────────────────
# Invariant core. These may NOT include soma/arch/ and may not name an
# architecture in code — tools/ci/check_seam.py enforces both.
set(SOMA_CORE_HEADERS
    soma/types.hpp
    soma/threading.hpp
    soma/quant.hpp
    soma/arch_ir.hpp
    soma/model.hpp
    soma/attention_backend.hpp
    soma/arch_backend.hpp
    soma/memory_hierarchy.hpp
    soma/expert_store.hpp
    soma/kernels.hpp
    soma/scheduler.hpp
    soma/kv_checkpoint.hpp
    soma/conformance.hpp
    soma/tokenizer.hpp
    soma/plan.hpp
    soma/routing.hpp
    soma/registry.hpp
    soma/telemetry.hpp
    soma/serve.hpp
)

# Per-architecture backends. Co-designed against Qwen3-MoE (GQA) and
# DeepSeek-V2-Lite (MLA) simultaneously, so the seam is not accidentally shaped
# to whichever ships first.
set(SOMA_ARCH_HEADERS
    soma/arch/gqa.hpp
    soma/arch/mla.hpp
)

list(APPEND SOMA_CORE_HEADERS
    soma/safetensors.hpp
    soma/kernels_f32.hpp
    soma/kernels_simd.hpp
    soma/kv_cache.hpp
    soma/sampler.hpp
    soma/quant_format.hpp
    soma/kernel_registry.hpp
    soma/f32_model.hpp
)

set(SOMA_HEADERS ${SOMA_CORE_HEADERS} ${SOMA_ARCH_HEADERS})

# ── mm_soma ───────────────────────────────────────────────────────────────────
# The G0 fp32 path. Explicit source list, per repo convention.
set(MM_SOMA_SOURCES
    types.cpp
    threading.cpp
    quant.cpp
    arch_ir.cpp
    safetensors.cpp
    conformance.cpp
    kernels_f32.cpp
    quant_format.cpp
    kernels_quant.cpp
    kernel_registry.cpp
    autotune.cpp
    expert_store.cpp
    plan.cpp
    memory_hierarchy.cpp
    f32_model.cpp
    kv_checkpoint.cpp
    telemetry.cpp
    kv_checkpoint_header.cpp   # standalone codec; the node links this object alone
    routing.cpp
    sampler.cpp
    serve.cpp
    scheduler.cpp
    tokenizer.cpp
    arch_registry.cpp      # the ONE core TU permitted to include soma/arch/
    arch/gqa.cpp
    arch/mla.cpp
)

# ── SIMD quantized kernels ────────────────────────────────────────────────────
# x86-64 only, and its own translation unit so the AVX2 flag applies to nothing
# else. The rest of the library must keep running on hosts without AVX2 —
# dispatch is a runtime CPUID check in kernels_quant.cpp, which is compiled at
# the baseline ISA precisely so the detector itself is always safe to call.
#
# aarch64 is a stated cross-compilation target and simply omits this file; the
# scalar reference path is the fallback there until NEON kernels land.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^([Xx]86_64|AMD64|amd64)$")
    set(SOMA_SIMD_SOURCES kernels_quant_avx2.cpp kernels_f32_avx2.cpp)
    list(APPEND MM_SOMA_SOURCES ${SOMA_SIMD_SOURCES})

    # AVX-512 is a THIRD tier, gated on the compiler being able to emit it — not
    # on the build host supporting it. Dispatch is a runtime CPUID check, so a
    # binary built here still runs on a machine without AVX-512; it just resolves
    # to the AVX2 tier. Building it in is therefore free for portability and only
    # costs compile time.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS_SAVED "${CMAKE_REQUIRED_FLAGS}")
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512dq -mavx512bw -mavx512vl")
    endif()
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 a = _mm512_set1_ps(1.0f);
            return (int)_mm512_reduce_add_ps(_mm512_mul_ps(a, a));
        }" SOMA_COMPILER_HAS_AVX512)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVED}")

    if(SOMA_COMPILER_HAS_AVX512)
        set(SOMA_AVX512_SOURCE kernels_avx512.cpp)
        list(APPEND MM_SOMA_SOURCES ${SOMA_AVX512_SOURCE})
    else()
        message(STATUS "soma: compiler cannot emit AVX-512 — building AVX2 tier only.")
    endif()
endif()

add_library(mm_soma STATIC ${MM_SOMA_SOURCES})
target_include_directories(mm_soma PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(mm_soma PUBLIC cxx_std_20)

if(SOMA_SIMD_SOURCES)
    if(MSVC)
        set_source_files_properties(${SOMA_SIMD_SOURCES}
                                    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(${SOMA_SIMD_SOURCES}
                                    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
    target_compile_definitions(mm_soma PRIVATE SOMA_HAS_AVX2=1)
endif()

if(SOMA_AVX512_SOURCE)
    if(MSVC)
        set_source_files_properties(${SOMA_AVX512_SOURCE}
                                    PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties(
            ${SOMA_AVX512_SOURCE}
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl")
    endif()
    target_compile_definitions(mm_soma PRIVATE SOMA_HAS_AVX512=1)
endif()

# nlohmann/json for config.json and the safetensors header; OpenSSL for
# arch_hash. Deliberately nothing else — the engine's dependency surface is a
# property worth defending early.
target_link_libraries(mm_soma PUBLIC nlohmann_json::nlohmann_json OpenSSL::Crypto)

# httplib for `soma serve`. PUBLIC because serve.hpp is part of the library's
# surface; the header itself does not include httplib, but the TU does and the
# link has to reach whoever pulls mm_soma in.
if(TARGET httplib::httplib)
    target_link_libraries(mm_soma PUBLIC httplib::httplib)
elseif(TARGET PkgConfig::CPP_HTTPLIB)
    target_link_libraries(mm_soma PUBLIC PkgConfig::CPP_HTTPLIB)
endif()

# ── soma ──────────────────────────────────────────────────────────────────────
# The engine executable the node supervises: `serve` and `plan`.
add_executable(soma main.cpp)
target_link_libraries(soma PRIVATE mm_soma)

# The node launches this binary as a subprocess with an inherited environment, so
# a PATH that happens to be right in a developer's shell is not a guarantee.
# Without its OpenSSL DLLs beside it the child dies with 0xC0000135 before main(),
# and the supervisor reports "engine exited before becoming ready" — accurate, and
# about a fault with nothing to do with supervision.
if(WIN32)
    add_custom_command(TARGET soma POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:soma>" "$<TARGET_FILE_DIR:soma>"
        COMMAND_EXPAND_LISTS)
endif()

# ── Header self-test ──────────────────────────────────────────────────────────
set(SOMA_SELFTEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/header_selftest")
file(MAKE_DIRECTORY "${SOMA_SELFTEST_DIR}")

set(SOMA_SELFTEST_SOURCES "")
set(SOMA_HEADER_PATHS "")
foreach(header IN LISTS SOMA_HEADERS)
    list(APPEND SOMA_HEADER_PATHS "${CMAKE_SOURCE_DIR}/include/${header}")
    string(MAKE_C_IDENTIFIER "${header}" header_tag)
    set(generated "${SOMA_SELFTEST_DIR}/st_${header_tag}.cpp")
    file(WRITE "${generated}"
"// Generated by src/soma/CMakeLists.txt. Do not edit.\n"
"//\n"
"// Included twice on purpose: the second include proves the guard works.\n"
"#include \"${header}\"\n"
"#include \"${header}\"\n"
"\n"
"namespace { int soma_header_selftest_${header_tag}() { return 0; } }\n"
"// Referenced so -Wunused-function does not fire on the probe itself.\n"
"int (*soma_header_selftest_ref_${header_tag})() = &soma_header_selftest_${header_tag};\n")
    list(APPEND SOMA_SELFTEST_SOURCES "${generated}")
endforeach()

add_library(soma_header_selftest OBJECT ${SOMA_SELFTEST_SOURCES})
target_include_directories(soma_header_selftest PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_features(soma_header_selftest PRIVATE cxx_std_20)

# Show the headers in IDEs even though nothing compiles them directly yet.
# Plain list rather than $<LIST:TRANSFORM,...>, which needs CMake 3.27 while
# this project requires 3.25.
target_sources(soma_header_selftest PRIVATE ${SOMA_HEADER_PATHS})
set_source_files_properties(${SOMA_HEADER_PATHS} PROPERTIES HEADER_FILE_ONLY ON)

# ── Seam falsification ────────────────────────────────────────────────────────
# Registered as a CTest entry so `ctest` covers it. A structural rule that is
# only checked by hand is a structural rule that decays.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(BUILD_TESTING)
    # ── G0 conformance ────────────────────────────────────────────────────────
    # Runs the fp32 path against the committed tiny-random fixtures and their
    # transformers oracle. Fixtures are committed precisely so this runs on every
    # commit without torch in the image.
    add_executable(soma_conformance_g0
                   ${CMAKE_SOURCE_DIR}/tests/soma/conformance_g0.cpp)
    target_link_libraries(soma_conformance_g0 PRIVATE mm_soma)

    add_test(NAME soma_conformance_g0
             COMMAND soma_conformance_g0 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny")

    # ── G0 tokenizer round-trip ───────────────────────────────────────────────
    # Compiled tokenizers vs HF `tokenizers`, byte-for-byte. The cheapest place
    # to catch a tokenizer fault; at G2 the same fault presents as "the model is
    # subtly stupid" rather than as a tokenizer error.
    add_executable(soma_tokenizer_g0
                   ${CMAKE_SOURCE_DIR}/tests/soma/tokenizer_g0.cpp)
    target_link_libraries(soma_tokenizer_g0 PRIVATE mm_soma)

    add_test(NAME soma_tokenizer_g0
             COMMAND soma_tokenizer_g0 "${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizers")

    # ── G1 quantized path ─────────────────────────────────────────────────────
    # Quantized vs fp32, per dtype. The fp32 path is the reference because G0
    # already tied it to transformers; comparing quantized output to the oracle
    # directly would conflate quantization error with a forward bug.
    add_executable(soma_conformance_g1
                   ${CMAKE_SOURCE_DIR}/tests/soma/conformance_g1.cpp)
    target_link_libraries(soma_conformance_g1 PRIVATE mm_soma)

    add_test(NAME soma_conformance_g1
             COMMAND soma_conformance_g1 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny")

    # ── G1 autotuner ──────────────────────────────────────────────────────────
    # Verifies candidate equivalence, shape extraction, and table round-trip.
    # The timings it prints are informational — a CI runner's numbers are not
    # the dev box's, and asserting on absolute throughput would make this test
    # flaky for no benefit.
    add_executable(soma_autotune_g1
                   ${CMAKE_SOURCE_DIR}/tests/soma/autotune_g1.cpp)
    target_link_libraries(soma_autotune_g1 PRIVATE mm_soma)

    add_test(NAME soma_autotune_g1
             COMMAND soma_autotune_g1 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny")

    # ── G2a container + verdict ───────────────────────────────────────────────
    # Round-trips the container against the engine's own quantization, and checks
    # the verdict function against the worked table in schemas/arch-ir.md §8 —
    # code and doc disagreeing means one of them is wrong.
    add_executable(soma_container_g2
                   ${CMAKE_SOURCE_DIR}/tests/soma/container_g2.cpp)
    target_link_libraries(soma_container_g2 PRIVATE mm_soma)

    add_test(NAME soma_container_g2
             COMMAND soma_container_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G2 streaming path ─────────────────────────────────────────────────────
    # LRU, eviction safety under a live borrow, the thrash gate, prefetch gating,
    # heat bootstrap, and measured-vs-predicted bytes/token. Run with a cache far
    # smaller than the routed set — a test whose cache holds everything measures
    # nothing about streaming.
    add_executable(soma_streaming_g2
                   ${CMAKE_SOURCE_DIR}/tests/soma/streaming_g2.cpp)
    target_link_libraries(soma_streaming_g2 PRIVATE mm_soma)

    add_test(NAME soma_streaming_g2
             COMMAND soma_streaming_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # Operator tool, not a test: points at a real container outside the repo, so
    # it is built but never registered with CTest.
    add_executable(soma_probe_container
                   ${CMAKE_SOURCE_DIR}/tests/soma/probe_container.cpp)
    target_link_libraries(soma_probe_container PRIVATE mm_soma)

    # Conformance stage 3. Also an operator tool rather than a CTest entry: it
    # needs a real container and a real reference, and neither belongs in the
    # repo. CI runs stages 1-2 on the committed tiny fixtures.
    add_executable(soma_stage3_g2
                   ${CMAKE_SOURCE_DIR}/tests/soma/stage3_g2.cpp)
    target_link_libraries(soma_stage3_g2 PRIVATE mm_soma)

    # ── G3 batch union + telemetry dump ───────────────────────────────────────
    # The union's CSR structure, the unique-vs-naive ratio across batch sizes,
    # and the G3-mandated text dump. Runs on the tiny fixtures, so the ratio is
    # checked on every commit rather than only when someone points the harness
    # at a real container.
    add_executable(soma_union_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/union_g3.cpp)
    target_link_libraries(soma_union_g3 PRIVATE mm_soma)

    add_test(NAME soma_union_g3
             COMMAND soma_union_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G3 SIMD quantized kernels ─────────────────────────────────────────────
    # SIMD vs the scalar reference, per dtype and per shape. A CTest entry: it
    # needs no fixtures, and a kernel that unpacks a sub-byte format wrongly is
    # exactly the kind of bug that survives a model-level check.
    add_executable(soma_simd_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/simd_g3.cpp)
    target_link_libraries(soma_simd_g3 PRIVATE mm_soma)

    add_test(NAME soma_simd_g3 COMMAND soma_simd_g3)

    # ── G3 threading determinism ──────────────────────────────────────────────
    # Bit-identity of the forward across thread counts. Re-executes itself with
    # SOMA_THREADS set, because the pool reads that once at first use and one
    # process cannot hold two pool sizes.
    add_executable(soma_threading_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/threading_g3.cpp)
    target_link_libraries(soma_threading_g3 PRIVATE mm_soma)

    add_test(NAME soma_threading_g3
             COMMAND soma_threading_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G3 streamed-path determinism ──────────────────────────────────────────
    # The gate whose absence let a concurrent-read race in ExpertStore surface as
    # a performance anomaly instead of a test failure. Covers what threading_g3
    # cannot: the cache, the loader threads, and eviction under a tiny budget.
    add_executable(soma_streamed_determinism_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/streamed_determinism_g3.cpp)
    target_link_libraries(soma_streamed_determinism_g3 PRIVATE mm_soma)

    add_test(NAME soma_streamed_determinism_g3
             COMMAND soma_streamed_determinism_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G3 step-major scheduler ───────────────────────────────────────────────
    # The load-bearing check: batching sequences together must not change what
    # any of them says. Plus the union across concurrent sequences and the
    # cache-aware gate.
    add_executable(soma_scheduler_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/scheduler_g3.cpp)
    target_link_libraries(soma_scheduler_g3 PRIVATE mm_soma)

    add_test(NAME soma_scheduler_g3
             COMMAND soma_scheduler_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G3 KV checkpoints and preemption ──────────────────────────────────────
    # preempt -> resume must reproduce the continuation byte-for-byte, and every
    # mismatched checkpoint must be refused rather than read.
    add_executable(soma_checkpoint_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/checkpoint_g3.cpp)
    target_link_libraries(soma_checkpoint_g3 PRIVATE mm_soma)

    add_test(NAME soma_checkpoint_g3
             COMMAND soma_checkpoint_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")

    # ── G3 sampling ───────────────────────────────────────────────────────────
    # Structural checks plus an empirical frequency test: a sampler drawing from
    # a subtly wrong distribution passes every structural assertion.
    add_executable(soma_sampler_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/sampler_g3.cpp)
    target_link_libraries(soma_sampler_g3 PRIVATE mm_soma)

    add_test(NAME soma_sampler_g3 COMMAND soma_sampler_g3)

    # ── G4 MLA positional path ────────────────────────────────────────────────
    # The cached-decode path agrees with the teacher-forced one, for every family.
    # Conformance cannot cover this: generate_greedy_f32 re-runs the PREFILL each
    # step, so `attention_kv` never executes there — which is how MLA's could stay
    # a stub returning Unsupported with every gate green.
    add_executable(soma_decode_kv_g4 ${CMAKE_SOURCE_DIR}/tests/soma/decode_kv_g4.cpp)
    target_link_libraries(soma_decode_kv_g4 PRIVATE mm_soma)
    add_test(NAME soma_decode_kv_g4
             COMMAND soma_decode_kv_g4 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny")

    add_executable(soma_mla_g4 ${CMAKE_SOURCE_DIR}/tests/soma/mla_g4.cpp)
    target_link_libraries(soma_mla_g4 PRIVATE mm_soma)
    add_test(NAME soma_mla_g4 COMMAND soma_mla_g4)

    # ── G6 telemetry ──────────────────────────────────────────────────────────
    # bucket_heat() is pure, so the downsampling claims are checked without a
    # model: a grid that exceeds the cap, or loses counts on the way down, is a
    # fault the brain view would show as understated load exactly where it is
    # highest.
    add_executable(soma_telemetry_g6 ${CMAKE_SOURCE_DIR}/tests/soma/telemetry_g6.cpp)
    target_link_libraries(soma_telemetry_g6 PRIVATE mm_soma)
    add_test(NAME soma_telemetry_g6 COMMAND soma_telemetry_g6)

    # ── G8 the admission ladder ───────────────────────────────────────────────
    # Runs the REAL `soma conform` against two half-complete fixtures. The gate is
    # that a stage which did not run says so, rather than passing by omission.
    add_executable(soma_conform_g8 ${CMAKE_SOURCE_DIR}/tests/soma/conform_g8.cpp)
    target_link_libraries(soma_conform_g8 PRIVATE nlohmann_json::nlohmann_json)
    add_test(NAME soma_conform_g8
             COMMAND soma_conform_g8
                     $<TARGET_FILE:soma>
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/Qwen3-30B-A3B"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tokenizers/Qwen3-30B-A3B")

    # ── G5 verdict routing ────────────────────────────────────────────────────
    # The whole policy table, plus the gate: Mixtral admits resident-only and
    # routes to the fallback with no override and no model-specific special case.
    add_executable(soma_routing_g5 ${CMAKE_SOURCE_DIR}/tests/soma/routing_g5.cpp)
    target_link_libraries(soma_routing_g5 PRIVATE mm_soma)
    add_test(NAME soma_routing_g5 COMMAND soma_routing_g5)

    # Activation dumper. An operator tool, not a CTest entry: its counterpart is
    # a Python reference run, and the diff between them is the deliverable.
    add_executable(soma_actdump_g4 ${CMAKE_SOURCE_DIR}/tests/soma/actdump_g4.cpp)
    target_link_libraries(soma_actdump_g4 PRIVATE mm_soma)

    # Batch scaling in the disk-bound regime. An operator tool, not a CTest
    # entry: the effect only exists against a container far larger than the
    # cache, and the tiny fixtures are the opposite of that.
    add_executable(soma_scaling_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/scaling_g3.cpp)
    target_link_libraries(soma_scaling_g3 PRIVATE mm_soma)
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(NAME soma_seam_check
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_seam.py"
                     "${CMAKE_SOURCE_DIR}")
elseif(BUILD_TESTING)
    message(STATUS
        "soma: Python3 not found — the seam check will not run under ctest. "
        "CI runs it independently, so this is a local-convenience gap only.")
endif()

# ── Roadmap status ────────────────────────────────────────────────────────────
# G0–G4 are implemented by mm_soma and the conformance targets above. G5 remains:
#
#   G5  soma executable — serve | plan --json | admit-verify
#
# src/soma/arch_registry.cpp is the ONE core TU permitted to include soma/arch/;
# it holds resolve_arch_backend() and resolve_attention_backend().
