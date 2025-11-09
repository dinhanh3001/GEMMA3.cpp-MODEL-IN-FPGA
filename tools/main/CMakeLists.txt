set(TARGET llama-cli)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)


if (USE_FPGA)
    target_compile_definitions(${TARGET} PRIVATE USE_FPGA)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()