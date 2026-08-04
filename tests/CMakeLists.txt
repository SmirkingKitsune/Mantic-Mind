add_executable(mm_reliability_tests
    reliability_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_config_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_scheduler.cpp
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
    ${CMAKE_SOURCE_DIR}/src/node/llama_cpp_provisioner.cpp
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
set_tests_properties(mm_reliability_tests PROPERTIES
    ENVIRONMENT "MM_TEST_SOMA_PATH=$<TARGET_FILE:soma>;MM_TEST_MODEL_DIR=${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Qwen3-30B-A3B")
