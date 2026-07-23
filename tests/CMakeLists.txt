add_executable(mm_reliability_tests
    reliability_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_config_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/control/control_api_server.cpp
    ${CMAKE_SOURCE_DIR}/src/control/performance_tracker.cpp
    ${CMAKE_SOURCE_DIR}/src/control/tts_service_client.cpp
    ${CMAKE_SOURCE_DIR}/src/control/node_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/control/node_operations.cpp
    ${CMAKE_SOURCE_DIR}/src/node/slot_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/node/process_exec.cpp
    ${CMAKE_SOURCE_DIR}/src/node/runtime_process.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_cpp_provisioner.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_service.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_state.cpp
    ${CMAKE_SOURCE_DIR}/src/node/model_store.cpp
)

target_include_directories(mm_reliability_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(mm_reliability_tests PRIVATE MM_TESTING)

target_link_libraries(mm_reliability_tests PRIVATE
    mm_common
    ${MM_PLATFORM_LIBS}
)

add_test(NAME mm_reliability_tests COMMAND mm_reliability_tests)

add_executable(mm_aio_config_tests
    aio_config_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/aio/aio_config.cpp
)

target_include_directories(mm_aio_config_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(mm_aio_config_tests PRIVATE
    mm_common
    ${MM_PLATFORM_LIBS}
)

add_test(NAME mm_aio_config_tests COMMAND mm_aio_config_tests)

add_executable(mm_aio_node_operations_tests
    aio_node_operations_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/aio/local_node_operations.cpp
    ${CMAKE_SOURCE_DIR}/src/control/node_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/control/node_operations.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_service.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_api_server.cpp
    ${CMAKE_SOURCE_DIR}/src/node/node_state.cpp
    ${CMAKE_SOURCE_DIR}/src/node/slot_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/node/runtime_process.cpp
    ${CMAKE_SOURCE_DIR}/src/node/process_exec.cpp
    ${CMAKE_SOURCE_DIR}/src/node/model_store.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_runtime.cpp
)

target_include_directories(mm_aio_node_operations_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(mm_aio_node_operations_tests PRIVATE MM_TESTING)

target_link_libraries(mm_aio_node_operations_tests PRIVATE
    mm_common
    ${MM_PLATFORM_LIBS}
)

add_test(NAME mm_aio_node_operations_tests COMMAND mm_aio_node_operations_tests)

add_executable(mm_node_host_tests
    node_host_tests.cpp
)

target_link_libraries(mm_node_host_tests PRIVATE
    mm_node_core
)

add_test(NAME mm_node_host_tests COMMAND mm_node_host_tests)
set_tests_properties(mm_node_host_tests PROPERTIES RUN_SERIAL TRUE)

add_executable(mm_aio_host_integration_tests
    aio_host_integration_tests.cpp
)

target_link_libraries(mm_aio_host_integration_tests PRIVATE
    mm_aio_core
)

add_test(NAME mm_aio_host_integration_tests
    COMMAND mm_aio_host_integration_tests
)
set_tests_properties(mm_aio_host_integration_tests PROPERTIES RUN_SERIAL TRUE)

add_test(NAME mm_install_layout
    COMMAND ${CMAKE_COMMAND}
        -DMM_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DMM_CONFIG=$<CONFIG>
        -DMM_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
        -DMM_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
        -DMM_INSTALL_DOCDIR=${CMAKE_INSTALL_DOCDIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/install_layout_test.cmake
)
set_tests_properties(mm_install_layout PROPERTIES RUN_SERIAL TRUE)
