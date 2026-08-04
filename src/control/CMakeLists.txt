add_executable(mantic-mind-control
    main.cpp
    node_registry.cpp
    agent_scheduler.cpp
    model_registry.cpp
    route_scope.cpp
    agent_queue.cpp
    agent_manager.cpp
    agent_config_validator.cpp
    control_api_server.cpp
    performance_tracker.cpp
    audio_player.cpp
    tts_service_client.cpp
    control_ui.cpp
)

# mm_soma is here for the ROUTING decision — select_backend() and its verdict
# types. Control does not run the engine; it decides which engine an agent gets,
# and that policy has one implementation shared with the engine rather than a
# reimplementation that drifts. src/soma/routing.cpp depends only on
# soma/types.hpp, so static linkage pulls two small objects.
target_link_libraries(mantic-mind-control PRIVATE
    mm_common
    mm_soma
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mantic-mind-control PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

install(TARGETS mantic-mind-control
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
