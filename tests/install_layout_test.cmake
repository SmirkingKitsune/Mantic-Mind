if(NOT DEFINED MM_BUILD_DIR OR MM_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "MM_BUILD_DIR is required")
endif()

set(MM_INSTALL_PREFIX "${MM_BUILD_DIR}/install-layout-test")
file(REMOVE_RECURSE "${MM_INSTALL_PREFIX}")

set(MM_INSTALL_COMMAND
    "${CMAKE_COMMAND}" --install "${MM_BUILD_DIR}"
    --prefix "${MM_INSTALL_PREFIX}"
)
if(DEFINED MM_CONFIG AND NOT MM_CONFIG STREQUAL "")
    list(APPEND MM_INSTALL_COMMAND --config "${MM_CONFIG}")
endif()

execute_process(
    COMMAND ${MM_INSTALL_COMMAND}
    RESULT_VARIABLE MM_INSTALL_RESULT
    OUTPUT_VARIABLE MM_INSTALL_OUTPUT
    ERROR_VARIABLE MM_INSTALL_ERROR
)
if(NOT MM_INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR
        "cmake --install failed (${MM_INSTALL_RESULT})\n"
        "${MM_INSTALL_OUTPUT}\n${MM_INSTALL_ERROR}")
endif()

foreach(MM_EXPECTED IN ITEMS
    "${MM_INSTALL_BINDIR}/mantic-mind${MM_EXECUTABLE_SUFFIX}"
    "${MM_INSTALL_BINDIR}/mantic-mind-control${MM_EXECUTABLE_SUFFIX}"
    "${MM_INSTALL_BINDIR}/mantic-mind-aio${MM_EXECUTABLE_SUFFIX}"
    "${MM_INSTALL_BINDIR}/mantic-mind.toml"
    "${MM_INSTALL_BINDIR}/mantic-mind-control.toml"
    "${MM_INSTALL_BINDIR}/mantic-mind-aio.toml"
    "${MM_INSTALL_DOCDIR}/aio.md")
    if(NOT EXISTS "${MM_INSTALL_PREFIX}/${MM_EXPECTED}")
        message(FATAL_ERROR
            "Installed layout is missing ${MM_EXPECTED}\n${MM_INSTALL_OUTPUT}")
    endif()
endforeach()

message(STATUS "Validated all executables, default configs, and AIO documentation")
