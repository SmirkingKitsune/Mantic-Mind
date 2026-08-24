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
    soma/prompt_codec.hpp
)

# Per-architecture backends. Co-designed against Qwen3-MoE (GQA) and
# DeepSeek-V2-Lite (MLA) simultaneously, so the seam is not accidentally shaped
# to whichever ships first.
set(SOMA_ARCH_HEADERS
    soma/arch/gqa.hpp
    soma/arch/mla.hpp
    soma/arch/kda.hpp
    soma/arch/gdn.hpp
    soma/arch/compressed_sparse.hpp
    soma/arch/deepseek_dspark.hpp
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
    arch/kda.cpp
    arch/gdn.cpp
    arch/compressed_sparse.cpp
    arch/deepseek_dsml.cpp
    arch/deepseek_dspark.cpp
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

# ── soma, beside the binaries that look for it ────────────────────────────────
# The node resolves `soma` BY NAME and the documented rule is that it ships
# beside the node. In a build tree it did not: soma lands in src/soma/ and the
# node in src/node/, and neither is on the other's PATH — so a cluster config
# naming Soma as its primary engine produced `absent` for a binary sitting one
# directory away, on every fresh build (D58). Control needs the same copy for a
# different reason: admission shells out to `soma plan` and `soma conform`.
#
# Copying rather than teaching the node about build-tree layout. "../soma/
# Release/soma.exe" is knowledge of a directory structure that does not exist in
# a deployment, compiled into production code — the honest fix is to make the
# claim true where it is false.
#
# Issued here because add_custom_command(TARGET) must be, in the directory that
# created the target; the runtime DLLs ride along because a soma.exe that cannot
# find its OpenSSL dies with 0xC0000135 before main() and the supervisor reports
# it as "exited before becoming ready".
foreach(host mantic-mind mantic-mind-control)
    add_dependencies(${host} soma)
    add_custom_command(TARGET soma POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${host}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:soma>" "$<TARGET_FILE_DIR:${host}>")
    if(WIN32)
        add_custom_command(TARGET soma POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_RUNTIME_DLLS:soma>" "$<TARGET_FILE_DIR:${host}>"
            COMMAND_EXPAND_LISTS)
    endif()
endforeach()

# Same rule for an installed tree: mantic-mind and mantic-mind-control both
# install to ${CMAKE_INSTALL_BINDIR}, so soma installing there is what makes
# "beside the node" true after `cmake --install` as well as in the build tree.
install(TARGETS soma
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

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
target_link_libraries(soma_header_selftest PRIVATE nlohmann_json::nlohmann_json)
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

    # PER-FAMILY REGISTRATION. The binary already took an optional fixture
    # name; only Qwen3 (GQA) was ever registered, so every assertion here was
    # made against one attention family. That is how streaming_g2 came to
    # hardcode "layer 0 is routed" — true for Qwen3, false for DeepSeek
    # (first_k_dense_replace=1) and GLM-5.2 (=3) — and report six failures on a
    # correct engine the first time it was pointed at MLA.
    add_test(NAME soma_autotune_g1
             COMMAND soma_autotune_g1 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny")
    add_test(NAME soma_autotune_mla_g1
             COMMAND soma_autotune_g1 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny" "DeepSeek-V2-Lite")
    add_test(NAME soma_autotune_dsa_g1
             COMMAND soma_autotune_g1 "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny" "GLM-5.2")

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
    # Both MLA families, and MLA+DSA. GLM-5.2 had a tiny checkpoint and NO
    # container, so every container-dependent gate was structurally blind to
    # the only DSA architecture; the container is now a committed fixture.
    add_test(NAME soma_streaming_mla_g2
             COMMAND soma_streaming_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_streaming_moonlight_g2
             COMMAND soma_streaming_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures" "Moonlight-16B-A3B")
    add_test(NAME soma_streaming_dsa_g2
             COMMAND soma_streaming_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_streaming_v4_g2
             COMMAND soma_streaming_g2 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

    # Operator tool, not a test: points at a real container outside the repo, so
    # it is built but never registered with CTest.
    add_executable(soma_probe_container
                   ${CMAKE_SOURCE_DIR}/tests/soma/probe_container.cpp)
    target_link_libraries(soma_probe_container PRIVATE mm_soma)

    # Conformance stage 3 — logit-KL against the transformers reference.
    #
    # This was built and NEVER registered, on the stated grounds that it "needs a
    # real container and a real reference, and neither belongs in the repo". That
    # was true when written and is not true now, and the consequence was that the
    # only DISTRIBUTIONAL correctness gate in the ladder ran for no architecture
    # at all — stages 1-2 assert token-exactness on tiny-random weights, which is
    # a different property from "the engine's logits match what transformers
    # produced".
    #
    # Both halves of the premise have since changed:
    #   * every tiny fixture carries `tf_logits` — a real teacher-forced
    #     reference from tools/admission/make_oracle.py, committed
    #   * every family now has a converted container, GLM-5.2 included (D50)
    #
    # Measured: 0.1 s per family, mean KL 0.00000-0.00029 nats against a 0.050
    # bar. And it DISCRIMINATES — pointing a container at another family's
    # reference fails, which is what makes registering it worth anything.
    add_executable(soma_stage3_g2
                   ${CMAKE_SOURCE_DIR}/tests/soma/stage3_g2.cpp)
    target_link_libraries(soma_stage3_g2 PRIVATE mm_soma)

    # One per family, because stage 3 is where a per-family numerical fault would
    # show and nowhere else does. GQA, MLA (two routers), MLA+DSA, and MHA.
    #
    # Written out rather than looped: mm_test_registry greps this file for a
    # LITERAL test name, so a foreach() registers tests it cannot see and reports
    # the suite as under-running. It caught that on the first attempt — and then
    # caught a comment here quoting its own grep pattern, which it read as a test
    # named "...". Names must be as discoverable statically as at configure time,
    # and prose in this file is scanned too.
    add_test(NAME soma_stage3_gqa_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/Qwen3-30B-A3B"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Qwen3-30B-A3B")
    add_test(NAME soma_stage3_mla_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/DeepSeek-V2-Lite"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/DeepSeek-V2-Lite")
    add_test(NAME soma_stage3_mla_sigmoid_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/Moonlight-16B-A3B"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Moonlight-16B-A3B")
    add_test(NAME soma_stage3_coarse_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/Mixtral-8x7B-v0.1"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Mixtral-8x7B-v0.1")
    add_test(NAME soma_stage3_mha_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/OLMoE-1B-7B-0924"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/OLMoE-1B-7B-0924")
    add_test(NAME soma_stage3_dsa_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/GLM-5.2"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/GLM-5.2")
    add_test(NAME soma_stage3_v4_g2
             COMMAND soma_stage3_g2
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/containers/DeepSeek-V4-Pro-0813"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/DeepSeek-V4-Pro-0813")

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
    add_test(NAME soma_union_mla_g3
             COMMAND soma_union_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_union_dsa_g3
             COMMAND soma_union_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_union_v4_g3
             COMMAND soma_union_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

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
    # Bit-identity across worker counts is a property of the SCHEDULER, but the
    # work it schedules is per-family attention. Verified passing on all three.
    add_test(NAME soma_threading_mla_g3
             COMMAND soma_threading_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_threading_dsa_g3
             COMMAND soma_threading_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_threading_v4_g3
             COMMAND soma_threading_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

    # ── G3 streamed-path determinism ──────────────────────────────────────────
    # The gate whose absence let a concurrent-read race in ExpertStore surface as
    # a performance anomaly instead of a test failure. Covers what threading_g3
    # cannot: the cache, the loader threads, and eviction under a tiny budget.
    add_executable(soma_streamed_determinism_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/streamed_determinism_g3.cpp)
    target_link_libraries(soma_streamed_determinism_g3 PRIVATE mm_soma)

    add_test(NAME soma_streamed_determinism_g3
             COMMAND soma_streamed_determinism_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")
    add_test(NAME soma_streamed_determinism_mla_g3
             COMMAND soma_streamed_determinism_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_streamed_determinism_dsa_g3
             COMMAND soma_streamed_determinism_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_streamed_determinism_v4_g3
             COMMAND soma_streamed_determinism_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

    # ── G3 step-major scheduler ───────────────────────────────────────────────
    # The load-bearing check: batching sequences together must not change what
    # any of them says. Plus the union across concurrent sequences and the
    # cache-aware gate.
    add_executable(soma_scheduler_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/scheduler_g3.cpp)
    target_link_libraries(soma_scheduler_g3 PRIVATE mm_soma)

    add_test(NAME soma_scheduler_g3
             COMMAND soma_scheduler_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")
    add_test(NAME soma_scheduler_mla_g3
             COMMAND soma_scheduler_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_scheduler_dsa_g3
             COMMAND soma_scheduler_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_scheduler_v4_g3
             COMMAND soma_scheduler_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

    # ── G3 KV checkpoints and preemption ──────────────────────────────────────
    # preempt -> resume must reproduce the continuation byte-for-byte, and every
    # mismatched checkpoint must be refused rather than read.
    add_executable(soma_checkpoint_g3
                   ${CMAKE_SOURCE_DIR}/tests/soma/checkpoint_g3.cpp)
    target_link_libraries(soma_checkpoint_g3 PRIVATE mm_soma)

    add_test(NAME soma_checkpoint_g3
             COMMAND soma_checkpoint_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures")
    add_test(NAME soma_checkpoint_mla_g3
             COMMAND soma_checkpoint_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V2-Lite")
    add_test(NAME soma_checkpoint_dsa_g3
             COMMAND soma_checkpoint_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "GLM-5.2")
    add_test(NAME soma_checkpoint_v4_g3
             COMMAND soma_checkpoint_g3 "${CMAKE_SOURCE_DIR}/tests/fixtures" "DeepSeek-V4-Pro-0813")

    # ── G4 target-verified speculative decoding ──────────────────────────────
    # Every possible draft acceptance prefix must reproduce the exact ordinary
    # autoregressive stream and expose truthful acceptance counters.
    add_executable(soma_speculative_g4
                   ${CMAKE_SOURCE_DIR}/tests/soma/speculative_g4.cpp)
    target_link_libraries(soma_speculative_g4 PRIVATE mm_soma)
    add_test(NAME soma_speculative_g4
             COMMAND soma_speculative_g4
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/DeepSeek-V4-Pro-0813")

    add_executable(soma_deepseek_dspark_oracle
                   ${CMAKE_SOURCE_DIR}/tests/soma/deepseek_dspark_oracle.cpp)
    target_link_libraries(soma_deepseek_dspark_oracle PRIVATE mm_soma)
    add_test(NAME soma_deepseek_dspark_oracle
             COMMAND soma_deepseek_dspark_oracle
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/oracles/DeepSeek-V4-Pro-0813-DSpark"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/DeepSeek-V4-Pro-0813-DSpark")

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

    # V4's official prompt/completion grammar: exact direct/thinking prefixes,
    # Unicode, tools, stable call ids, and explicit malformed-output failures.
    add_executable(soma_deepseek_v4_codec
                   ${CMAKE_SOURCE_DIR}/tests/soma/deepseek_v4_codec.cpp)
    target_link_libraries(soma_deepseek_v4_codec PRIVATE mm_soma)
    add_test(NAME soma_deepseek_v4_codec
             COMMAND soma_deepseek_v4_codec
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/dsml/encoding")

    add_executable(soma_deepseek_v4_plan
                   ${CMAKE_SOURCE_DIR}/tests/soma/deepseek_v4_plan.cpp)
    target_link_libraries(soma_deepseek_v4_plan PRIVATE mm_soma)
    add_test(NAME soma_deepseek_v4_plan
             COMMAND soma_deepseek_v4_plan
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/DeepSeek-V4-Pro-0813")

    # The KDA kernel, checked against derived invariants rather than a
    # transcription: almost every misreading of a delta rule still converges.
    add_executable(soma_kda_kernel
                   ${CMAKE_SOURCE_DIR}/tests/soma/kda_kernel.cpp)
    target_link_libraries(soma_kda_kernel PRIVATE mm_soma)
    add_test(NAME soma_kda_kernel COMMAND soma_kda_kernel)

    # The KDA kernel against fla's own `fused_recurrent_kda`. The one EXTERNAL
    # check this family has; everything else is a second opinion from the author.
    add_executable(soma_kda_oracle
                   ${CMAKE_SOURCE_DIR}/tests/soma/kda_oracle.cpp)
    target_link_libraries(soma_kda_oracle PRIVATE mm_soma)
    add_test(NAME soma_kda_oracle
             COMMAND soma_kda_oracle
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/kernels/kda")

    # The block residual: mixing pinned by identities, sequencing by two traces
    # computed by hand under a zero score vector.
    add_executable(soma_block_residual
                   ${CMAKE_SOURCE_DIR}/tests/soma/block_residual.cpp)
    target_link_libraries(soma_block_residual PRIVATE mm_soma)
    add_test(NAME soma_block_residual COMMAND soma_block_residual)

    # The latent MoE, driven through the real forward. Family-agnostic, so it
    # runs on GQA — the simplest backend that reaches the expert loop.
    add_executable(soma_latent_moe
                   ${CMAKE_SOURCE_DIR}/tests/soma/latent_moe.cpp)
    target_link_libraries(soma_latent_moe PRIVATE mm_soma)
    add_test(NAME soma_latent_moe COMMAND soma_latent_moe)

    # MLA's NoPE and output-gate variants. The G4 fixtures configure neither,
    # so without this they are guarded code nobody has ever run.
    add_executable(soma_mla_variants
                   ${CMAKE_SOURCE_DIR}/tests/soma/mla_variants.cpp)
    target_link_libraries(soma_mla_variants PRIVATE mm_soma)
    add_test(NAME soma_mla_variants COMMAND soma_mla_variants)

    # The layer around the kernel: prefill, streaming and cached decode are
    # three routes through one recurrence and must agree, or one of them is wrong.
    add_executable(soma_kda_layer
                   ${CMAKE_SOURCE_DIR}/tests/soma/kda_layer.cpp)
    target_link_libraries(soma_kda_layer PRIVATE mm_soma)
    add_test(NAME soma_kda_layer COMMAND soma_kda_layer)

    # Kimi-K3: hybrid KDA/MLA, latent MoE, multimodal wrapper. Config-only —
    # no fixture, because every claim it makes is a property of the adapter and
    # the planner, and the checkpoint is ~1.5 TB.
    add_executable(soma_kimi_k3_plan
                   ${CMAKE_SOURCE_DIR}/tests/soma/kimi_k3_plan.cpp)
    target_link_libraries(soma_kimi_k3_plan PRIVATE mm_soma)
    add_test(NAME soma_kimi_k3_plan COMMAND soma_kimi_k3_plan)

    # Qwen3.5-MoE: the SECOND hybrid — GQA full layers with a fused output gate
    # and partial rope, Gated DeltaNet linear layers whose state is indexed by a
    # different head count than their key projection. Config-only for the same
    # reason as Kimi's above, and more so: the checkpoint is 4.89 TB.
    add_executable(soma_qwen3_5_moe_plan
                   ${CMAKE_SOURCE_DIR}/tests/soma/qwen3_5_moe_plan.cpp)
    target_link_libraries(soma_qwen3_5_moe_plan PRIVATE mm_soma)
    add_test(NAME soma_qwen3_5_moe_plan COMMAND soma_qwen3_5_moe_plan)

    # The four Gated DeltaNet operators in isolation, plus the property the
    # oracle cannot reach: prefill and stepwise decode are the same recurrence.
    add_executable(soma_gdn_kernel ${CMAKE_SOURCE_DIR}/tests/soma/gdn_kernel.cpp)
    target_link_libraries(soma_gdn_kernel PRIVATE mm_soma)
    add_test(NAME soma_gdn_kernel COMMAND soma_gdn_kernel)

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
# it holds resolve_f32_backend() and resolve_attention_backend().
