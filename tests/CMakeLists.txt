add_executable(mm_reliability_tests
    reliability_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_config_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/control/engine_config_store.cpp
    ${CMAKE_SOURCE_DIR}/src/control/engine_group_planner.cpp
    ${CMAKE_SOURCE_DIR}/src/control/model_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/control/route_scope.cpp
    ${CMAKE_SOURCE_DIR}/src/control/control_api_server.cpp
    ${CMAKE_SOURCE_DIR}/src/control/performance_tracker.cpp
    ${CMAKE_SOURCE_DIR}/src/control/tts_service_client.cpp
    ${CMAKE_SOURCE_DIR}/src/control/node_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/node/engine_process.cpp
    ${CMAKE_SOURCE_DIR}/src/node/engine_descriptor.cpp
    ${CMAKE_SOURCE_DIR}/src/node/engine_supervisor.cpp
    ${CMAKE_SOURCE_DIR}/src/node/kv_checkpoint_backend.cpp
    ${CMAKE_SOURCE_DIR}/src/node/runtime_process.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/node/vllm_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_cpp_provisioner.cpp
    ${CMAKE_SOURCE_DIR}/src/node/engine_provisioner.cpp
    ${CMAKE_SOURCE_DIR}/src/node/engine_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_state.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_modal.cpp
    ${CMAKE_SOURCE_DIR}/src/node/ray_orchestration.cpp
)

target_include_directories(mm_reliability_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(mm_reliability_tests PRIVATE MM_TESTING)

target_link_libraries(mm_reliability_tests PRIVATE
    mm_common
    # NOT mm_node_engine: this target defines MM_TESTING, which enables
    # add_ready_test_engine(), so it compiles those sources itself — the same
    # reason it compiles control's. Linking the library too would give one binary
    # two definitions of every symbol in it.
    # Same reason the control target links it: agent_scheduler.cpp resolves an
    # agent's engine through soma::select_backend(), so the policy has one
    # implementation rather than a control-side copy that drifts from the engine.
    mm_soma
    ${MM_PLATFORM_LIBS}
)

# On Windows this binary resolves SQLiteCpp/OpenSSL/spdlog from PATH. Under
# `ctest` that PATH is whatever the invoking shell had, so the loader fails with
# 0xC0000135 before main() and the test reports an exception with no output at
# all — the least diagnosable failure shape there is. Copying the dependency set
# beside the .exe makes it runnable from any shell.
if(WIN32)
    add_custom_command(TARGET mm_reliability_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:mm_reliability_tests>"
                "$<TARGET_FILE_DIR:mm_reliability_tests>"
        COMMAND_EXPAND_LISTS)
endif()

add_test(NAME mm_reliability_tests COMMAND mm_reliability_tests)

# The admission test drives the REAL `soma plan --json` rather than a mock, so it
# needs the binary and a model. Passed as environment rather than argv because
# reliability_tests takes an optional test-name filter and nothing else.
# Called here as well as in src/soma: find_package results live in the directory
# scope that ran them, and src/soma is a sibling, not a parent.
find_package(Python3 COMPONENTS Interpreter QUIET)

# Prefer the admission venv, because the interpreter decides whether these tests
# TEST anything.
#
# find_package returns the system interpreter, which has no numpy. The two tests
# that convert fixtures skip on ImportError and return 0, so ctest printed
# "mm_fused_experts ... Passed" in 0.11 s while running nothing at all — D4's
# byte-identity check has never once executed in the suite. A green run that
# proves nothing is worse than a red one (roadmap D28).
#
# Two changes: look here first, and — below — make a skip REPORT as skipped so
# the next missing dependency is visible instead of silently green.
set(MM_ADMISSION_PY "${Python3_EXECUTABLE}")
foreach(candidate
        "${CMAKE_SOURCE_DIR}/tools/admission/.venv/Scripts/python.exe"
        "${CMAKE_SOURCE_DIR}/tools/admission/.venv/bin/python")
    if(EXISTS "${candidate}")
        set(MM_ADMISSION_PY "${candidate}")
        break()
    endif()
endforeach()
message(STATUS "admission tests will run under: ${MM_ADMISSION_PY}")

#
# MM_TEST_PYTHON drives the fetch stage against a STUB fetch.py. The stage is
# the only one that touches the network, so it is the only one that cannot be
# tested against the real thing — what is under test is the C++ side: the
# progress parsing, the resolved path, and the two ways a fetch can lie about
# succeeding.
set_tests_properties(mm_reliability_tests PROPERTIES
    ENVIRONMENT "MM_TEST_SOMA_PATH=$<TARGET_FILE:soma>;MM_TEST_MODEL_DIR=${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Qwen3-30B-A3B;MM_TEST_PYTHON=${Python3_EXECUTABLE};MM_TEST_CONTAINER_DIR=${CMAKE_SOURCE_DIR}/tests/fixtures/containers/Qwen3-30B-A3B")

if(Python3_Interpreter_FOUND)
    # The direction require_complete_coverage() cannot walk. It asserts every
    # REGISTERED handler has a scope; nothing asserted that every DOCUMENTED
    # route was ever registered, and one sat unbuilt from the day the API
    # document was written.
    add_test(NAME mm_api_docs_check
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_api_docs.py"
                     "${CMAKE_SOURCE_DIR}")

    # P1's other half. check_api_docs walks docs -> scope table and check_ui_api
    # walks panel includes; neither asks whether a capability the TUI or CLI can
    # reach is reachable through /v1 at all.
    #
    # It was not: AgentScheduler::suspend_agent was private, the node API had
    # /api/node/suspend-slot, and no /v1 route could reach either — so an
    # operator holding the whole control API could not suspend an agent while
    # the scheduler did it on its own under capacity pressure.
    # A design header said exactly that in a comment and was compiled by
    # nothing (roadmap D46), so nothing failed. This check is the answer to
    # that: the claim now lives somewhere a build can refuse it.
    add_test(NAME mm_surface_parity
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_surface_parity.py"
                     "${CMAKE_SOURCE_DIR}")

    # Pure, no network, no huggingface_hub: which files a repo transfer takes is
    # both the economic argument for this stage and its safety boundary.
    add_test(NAME mm_fetch_selection
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_fetch_selection.py"
                     "${CMAKE_SOURCE_DIR}")

    # The suite must run every test it declares.
    #
    # ctest reads the build directory's GENERATED test list, and that copy can go
    # stale against CMakeLists.txt — which it did, hiding the MLA and DSA
    # checkpoint round-trips while "29/29 passed" looked like a full green run
    # (roadmap D44). The count is the only outward signal and nobody remembers
    # last week's count.
    #
    # Self-checking is sound here because staleness only ever HIDES NEW tests: a
    # stale file is an older complete one, so this check keeps running and keeps
    # seeing what was added after it.
    add_test(NAME mm_test_registry
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_test_registry.py"
                     "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

    # The fused expert layout must convert byte-identically to the per-expert
    # one. The gate/up split is contiguous here and INTERLEAVED in gpt_oss, and
    # both readings produce a container that loads and runs — so this compares
    # the two readers against each other on the same weights rather than
    # trusting either. Skips itself when numpy/safetensors are absent.
    add_test(NAME mm_fused_experts
             COMMAND "${MM_ADMISSION_PY}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_fused_experts.py"
                     "${CMAKE_SOURCE_DIR}")

    # verify_payload.py is what stands between a 460 GB expert payload and an
    # admission that never read it (roadmap D27). A checker nobody has broken on
    # purpose reports "OK" for reasons unknown, so this damages a good container
    # five ways and requires the RIGHT pass to catch each — including a two-expert
    # swap, which leaves every offset and length perfect and is visible only to
    # the decode pass's decoy margin. Skips itself when numpy/safetensors/torch
    # are absent.
    add_test(NAME mm_verify_payload
             COMMAND "${MM_ADMISSION_PY}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_verify_payload.py"
                     "${CMAKE_CURRENT_BINARY_DIR}/verify_payload_tmp")

    # MiniMax-M3's production checkpoint spells its tensors differently from
    # every fixture, and convert.py is the only thing that reconciles them. The
    # 59 shards that would exercise that map are most of a terabyte, so this
    # manufactures the input: it rewrites the committed fixture INTO the
    # production dialect, converts both, and asserts the two containers describe
    # the same weights. Skips itself when numpy/safetensors are absent.
    add_test(NAME mm_minimax_dialect
             COMMAND "${MM_ADMISSION_PY}"
                     "${CMAKE_SOURCE_DIR}/tools/ci/check_minimax_dialect.py"
                     "${CMAKE_SOURCE_DIR}"
                     "${CMAKE_CURRENT_BINARY_DIR}/minimax_dialect_tmp")

    # Opt-in release smoke for a real Linux multi-node vLLM/Ray deployment.
    # The script skips unless its control URL and agent ID are supplied.
    add_test(NAME mm_vllm_ray_smoke
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tests/vllm_ray_smoke.py")

    # 77 is ctest's conventional "skipped". Without this a missing dependency
    # exits 0 and reads as Passed, which is how mm_fused_experts ran nothing for
    # its whole existence while showing green.
    set_tests_properties(mm_fused_experts mm_verify_payload mm_minimax_dialect
                         mm_vllm_ray_smoke PROPERTIES
                         SKIP_RETURN_CODE 77)
endif()
