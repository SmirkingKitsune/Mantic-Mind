set(MM_AIO_CORE_SOURCES
    aio_config.cpp
    aio_host.cpp
    local_node_operations.cpp
    ${CMAKE_SOURCE_DIR}/src/control/control_ui.cpp
)

add_library(mm_aio_core STATIC ${MM_AIO_CORE_SOURCES})

target_link_libraries(mm_aio_core PUBLIC
    mm_control_core
    mm_node_core
)

target_link_libraries(mm_aio_core PRIVATE
    ftxui::screen
    ftxui::dom
    ftxui::component
)

target_include_directories(mm_aio_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(mm_aio_core PUBLIC cxx_std_20)

add_executable(mantic-mind-aio main.cpp)

target_link_libraries(mantic-mind-aio PRIVATE mm_aio_core)

install(TARGETS mantic-mind-aio
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
