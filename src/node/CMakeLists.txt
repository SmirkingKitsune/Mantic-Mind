# ── mm_node_engine ────────────────────────────────────────────────────────────
# The engine-neutral supervision layer, split out as a library because the
# conformance harness has to exercise it directly. A crash watchdog verified only
# through the node executable is a watchdog verified by inspection.
#
# It deliberately does NOT depend on ftxui or NodeState: an EngineProcess that
# knows what a TUI is cannot be tested without one.
add_library(mm_node_engine STATIC
    engine_process.cpp
    engine_descriptor.cpp
    engine_supervisor.cpp
    kv_checkpoint_backend.cpp
)

# mm_soma is here for ONE object: the KV checkpoint header codec. The node must
# read a checkpoint header before spawning an engine for it, and a second parser
# living in src/node is how a wire format acquires two definitions. Static-library
# linkage pulls only the objects actually referenced, so this does not drag the
# engine into the node — src/soma/kv_checkpoint_header.cpp exists as its own TU
# precisely to keep that true.
target_link_libraries(mm_node_engine PUBLIC
    mm_common
    mm_soma
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mm_node_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(mantic-mind
    main.cpp
    process_exec.cpp
    runtime_process.cpp
    llama_runtime.cpp
    llama_cpp_provisioner.cpp
    model_store.cpp
    node_api_server.cpp
    node_state.cpp
    node_ui.cpp
    singleton_lock.cpp
)

target_link_libraries(mantic-mind PRIVATE
    mm_common
    mm_node_engine
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mantic-mind PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

install(TARGETS mantic-mind
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ── G5 supervision ────────────────────────────────────────────────────────────
# Launches the real `soma` binary and kills it behind the supervisor's back. A
# CTest entry rather than an operator tool: `soma serve` accepts the committed
# tiny fixtures, so the watchdog is exercised on every commit instead of only
# when someone points the harness at a real container.
if(BUILD_TESTING)
    add_executable(soma_engine_g5 ${CMAKE_SOURCE_DIR}/tests/soma/engine_g5.cpp)
    target_link_libraries(soma_engine_g5 PRIVATE mm_node_engine)

    add_test(NAME soma_engine_g5
             COMMAND soma_engine_g5
                     "$<TARGET_FILE:soma>"
                     "${CMAKE_SOURCE_DIR}/tests/fixtures/tiny/Qwen3-30B-A3B")

    # On Windows both binaries resolve OpenSSL/SQLiteCpp/spdlog beside the .exe or
    # from PATH; with neither, the loader fails with 0xC0000135 before main() and
    # the test reports "engine exited before becoming ready" — a true statement
    # about a fault that has nothing to do with the supervisor. Copying the
    # dependency set next to each binary keeps the test runnable from any shell.
    if(WIN32)
        add_custom_command(TARGET soma_engine_g5 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_RUNTIME_DLLS:soma_engine_g5>" "$<TARGET_FILE_DIR:soma_engine_g5>"
            COMMAND_EXPAND_LISTS)
    endif()
    # `soma` gets the same treatment in src/soma/CMakeLists.txt — add_custom_command
    # has to be issued in the directory that created the target.
endif()
