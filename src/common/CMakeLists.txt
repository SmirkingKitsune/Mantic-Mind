set(MM_COMMON_SOURCES
    util.cpp
    logger.cpp
    agent.cpp
    agent_db.cpp
    llama_cpp_client.cpp
    conversation_manager.cpp
    memory_manager.cpp
    tool_executor.cpp
    http_server.cpp
    http_client.cpp
    sse_emitter.cpp
    config_file.cpp
    pairing.cpp
    node_discovery.cpp
    gguf_metadata.cpp
)

add_library(mm_common STATIC ${MM_COMMON_SOURCES})

target_include_directories(mm_common PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(mm_common PUBLIC
    SQLiteCpp
    httplib::httplib
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    OpenSSL::Crypto
    ${MM_PLATFORM_LIBS}
)

target_compile_features(mm_common PUBLIC cxx_std_20)
