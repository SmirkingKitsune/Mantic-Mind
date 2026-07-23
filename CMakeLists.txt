cmake_minimum_required(VERSION 3.25)
project(mantic-mind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

set(MM_LOCAL_DEPS_DIR "${CMAKE_SOURCE_DIR}/.deps")
foreach(MM_LOCAL_DEP sqlitecpp)
    if(EXISTS "${MM_LOCAL_DEPS_DIR}/${MM_LOCAL_DEP}/usr")
        list(PREPEND CMAKE_PREFIX_PATH "${MM_LOCAL_DEPS_DIR}/${MM_LOCAL_DEP}/usr")
    endif()
endforeach()

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /WX /utf-8 /FS /wd4068)
    # /wd4068: suppress unknown pragma warnings (from third-party headers)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror
        -Wno-unknown-pragmas
        # Designated-initializer aggregates intentionally leave trailing members
        # value-initialized (e.g. Message{.role=…, .content=…}). MSVC /W4 does
        # not flag this; silence GCC/Clang to match.
        -Wno-missing-field-initializers)
endif()

# -- Dependencies -------------------------------------------------------------
find_package(ftxui CONFIG REQUIRED)

find_package(SQLiteCpp CONFIG QUIET)
if(TARGET SQLiteCpp::SQLiteCpp AND NOT TARGET SQLiteCpp)
    add_library(SQLiteCpp INTERFACE IMPORTED)
    target_link_libraries(SQLiteCpp INTERFACE SQLiteCpp::SQLiteCpp)
endif()
if(NOT TARGET SQLiteCpp)
    find_package(SQLite3 REQUIRED)
    find_path(SQLiteCpp_INCLUDE_DIR NAMES SQLiteCpp/SQLiteCpp.h)
    find_library(SQLiteCpp_LIBRARY NAMES SQLiteCpp sqlitecpp)
    if(NOT SQLiteCpp_INCLUDE_DIR OR NOT SQLiteCpp_LIBRARY)
        message(FATAL_ERROR
            "SQLiteCpp was not found. Install the system package "
            "(Ubuntu/Debian: sudo apt-get install libsqlitecpp-dev), "
            "build with vcpkg, or set CMAKE_PREFIX_PATH/SQLiteCpp_DIR.")
    endif()
    add_library(SQLiteCpp UNKNOWN IMPORTED)
    set_target_properties(SQLiteCpp PROPERTIES
        IMPORTED_LOCATION "${SQLiteCpp_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SQLiteCpp_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES SQLite::SQLite3
    )
endif()

find_package(httplib CONFIG QUIET)
if(NOT TARGET httplib::httplib)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(CPP_HTTPLIB QUIET IMPORTED_TARGET cpp-httplib)
    endif()

    if(TARGET PkgConfig::CPP_HTTPLIB)
        add_library(httplib::httplib INTERFACE IMPORTED)
        target_link_libraries(httplib::httplib INTERFACE PkgConfig::CPP_HTTPLIB)
    else()
        find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h)
        find_library(HTTPLIB_LIBRARY NAMES cpp-httplib httplib)
        if(NOT HTTPLIB_INCLUDE_DIR)
            message(FATAL_ERROR
                "cpp-httplib was not found. Install the system package "
                "(Ubuntu/Debian: sudo apt-get install libcpp-httplib-dev), "
                "build with vcpkg, or set CMAKE_PREFIX_PATH/httplib_DIR.")
        endif()

        add_library(httplib::httplib INTERFACE IMPORTED)
        target_include_directories(httplib::httplib INTERFACE "${HTTPLIB_INCLUDE_DIR}")
        if(HTTPLIB_LIBRARY)
            target_link_libraries(httplib::httplib INTERFACE "${HTTPLIB_LIBRARY}")
        endif()
    endif()
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# ── Platform libraries ────────────────────────────────────────────────────────
if(WIN32)
    set(MM_PLATFORM_LIBS ws2_32 crypt32 user32)
else()
    set(MM_PLATFORM_LIBS "")
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(src/common)
add_subdirectory(src/node)
add_subdirectory(src/control)
add_subdirectory(src/aio)

install(FILES
    ${CMAKE_SOURCE_DIR}/tools/mantic-mind.toml
    ${CMAKE_SOURCE_DIR}/tools/mantic-mind-control.toml
    ${CMAKE_SOURCE_DIR}/tools/mantic-mind-aio.toml
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_SOURCE_DIR}/README.md
    ${CMAKE_SOURCE_DIR}/docs/aio.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
