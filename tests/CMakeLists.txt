add_executable(mm_reliability_tests
    reliability_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_queue.cpp
    ${CMAKE_SOURCE_DIR}/src/control/agent_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/node/slot_manager.cpp
    ${CMAKE_SOURCE_DIR}/src/node/llama_server_process.cpp
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
