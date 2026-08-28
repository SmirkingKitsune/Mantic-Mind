add_executable(mantic-mind-control
    main.cpp
    node_registry.cpp
    agent_scheduler.cpp
    engine_config_store.cpp
    engine_group_planner.cpp
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
    admission_panels.cpp
    engine_panels.cpp
    soma_dashboard.cpp
    soma_panels.cpp
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

if(BUILD_TESTING)
    # ── G7 the brain grid ─────────────────────────────────────────────────────
    #
    # layout_heat() is pure — no FTXUI, no HTTP, no clock — so the reduction is
    # checkable as arithmetic rather than by looking at a terminal. A grid that
    # is subtly wrong looks exactly like a model that routes oddly, and an
    # operator has no way to tell those apart.
    #
    # Compiles the guarded TU directly rather than linking the control binary,
    # which is an executable. That also keeps the test honest about the TU's
    # dependencies: if a panel ever needed NodeRegistry, this would stop linking.
    # The Admissions tab's renderers, drawn to an off-screen buffer and asserted
    # as text. Compiles admission_panels.cpp directly — the panels are pure over
    # a snapshot, so none of this needs a server, a registry, or a terminal.
    add_executable(mm_admission_panels
        ${CMAKE_CURRENT_SOURCE_DIR}/admission_panels.cpp
        ${CMAKE_SOURCE_DIR}/tests/control/admission_panels_test.cpp)
    target_link_libraries(mm_admission_panels PRIVATE
        mm_common ftxui::screen ftxui::dom ftxui::component ${MM_PLATFORM_LIBS})
    target_include_directories(mm_admission_panels PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME mm_admission_panels COMMAND mm_admission_panels)

    add_executable(mm_dashboard_g7
        ${CMAKE_CURRENT_SOURCE_DIR}/soma_dashboard.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/soma_panels.cpp
        ${CMAKE_SOURCE_DIR}/tests/soma/dashboard_g7.cpp)
    target_link_libraries(mm_dashboard_g7 PRIVATE
        mm_common ftxui::screen ftxui::dom ftxui::component ${MM_PLATFORM_LIBS})
    target_include_directories(mm_dashboard_g7 PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME mm_dashboard_g7 COMMAND mm_dashboard_g7)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        # "Grep-verifiable" as a test rather than a promise. The existing TUI
        # already holds NodeRegistry&, AgentManager& and AgentScheduler&, so a
        # panel needing one number is four keystrokes from taking it directly.
        add_test(NAME mm_ui_api_check
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tools/ci/check_ui_api.py"
                         "${CMAKE_SOURCE_DIR}")
    endif()
endif()
