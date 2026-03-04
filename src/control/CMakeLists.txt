add_executable(mantic-mind-control
    main.cpp
    node_registry.cpp
    model_router.cpp
    model_distributor.cpp
    agent_scheduler.cpp
    agent_queue.cpp
    agent_manager.cpp
    agent_config_validator.cpp
    control_api_server.cpp
    control_ui.cpp
)

target_link_libraries(mantic-mind-control PRIVATE
    mm_common
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mantic-mind-control PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
