add_executable(mantic-mind
    main.cpp
    llama_runtime_manager.cpp
    llama_server_process.cpp
    model_storage.cpp
    node_api_server.cpp
    node_state.cpp
    node_ui.cpp
    singleton_lock.cpp
    slot_manager.cpp
)

target_link_libraries(mantic-mind PRIVATE
    mm_common
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mantic-mind PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
