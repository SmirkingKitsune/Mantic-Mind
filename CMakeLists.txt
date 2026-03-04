cmake_minimum_required(VERSION 3.25)
project(mantic-mind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /WX /utf-8 /wd4068)
    # /wd4068: suppress unknown pragma warnings (from third-party headers)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror
        -Wno-unknown-pragmas)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ftxui CONFIG REQUIRED)
find_package(SQLiteCpp CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# ── Platform libraries ────────────────────────────────────────────────────────
if(WIN32)
    set(MM_PLATFORM_LIBS ws2_32 crypt32)
else()
    set(MM_PLATFORM_LIBS "")
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(src/common)
add_subdirectory(src/node)
add_subdirectory(src/control)
