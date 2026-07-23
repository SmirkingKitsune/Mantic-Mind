set(MM_CONTROL_CORE_SOURCES
    node_operations.cpp
    node_registry.cpp
    agent_scheduler.cpp
    agent_queue.cpp
    agent_manager.cpp
    agent_config_validator.cpp
    control_host.cpp
    control_api_server.cpp
    performance_tracker.cpp
    audio_player.cpp
    tts_service_client.cpp
)

add_library(mm_control_core STATIC ${MM_CONTROL_CORE_SOURCES})

target_link_libraries(mm_control_core PUBLIC
    mm_common
    ${MM_PLATFORM_LIBS}
)

target_include_directories(mm_control_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(mm_control_core PUBLIC cxx_std_20)

add_executable(mantic-mind-control
    main.cpp
    control_ui.cpp
)

target_link_libraries(mantic-mind-control PRIVATE
    mm_control_core
    ftxui::screen
    ftxui::dom
    ftxui::component
)

install(TARGETS mantic-mind-control
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
