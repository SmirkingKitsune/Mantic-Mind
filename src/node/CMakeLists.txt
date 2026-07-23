set(MM_NODE_CORE_SOURCES
    process_exec.cpp
    runtime_process.cpp
    llama_runtime.cpp
    llama_cpp_provisioner.cpp
    model_store.cpp
    node_host.cpp
    node_api_server.cpp
    node_service.cpp
    node_state.cpp
    singleton_lock.cpp
    slot_manager.cpp
)

add_library(mm_node_core STATIC ${MM_NODE_CORE_SOURCES})

target_link_libraries(mm_node_core PUBLIC
    mm_common
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mm_node_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(mm_node_core PUBLIC cxx_std_20)

add_executable(mantic-mind
    main.cpp
    node_ui.cpp
)

target_link_libraries(mantic-mind PRIVATE
    mm_node_core
    ftxui::screen
    ftxui::dom
    ftxui::component
)

install(TARGETS mantic-mind
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
