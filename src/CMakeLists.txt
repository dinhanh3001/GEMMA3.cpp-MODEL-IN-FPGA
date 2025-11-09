llama_add_compile_flags()

# Option to enable FPGA/XRT support from CMake command line (cmake -DUSE_FPGA=ON ...)
option(USE_FPGA "Enable FPGA / XRT support (requires XRT installed)" OFF)

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            fpga_host.cpp
            fpga_host.h
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

if (USE_FPGA)
   # 1. Thêm source file FPGA chỉ khi bật option
    target_sources(llama PRIVATE fpga_host.cpp fpga_host.h)
    # try to locate XRT headers and library
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_device.h HINTS /opt/xilinx/xrt/include /usr/include)
    find_library(XRT_LIB NAMES xrt_core xrt_coreutil xrt HINTS /opt/xilinx/xrt/lib /usr/lib)
    if (NOT XRT_INCLUDE_DIR OR NOT XRT_LIB)
        message(WARNING "USE_FPGA is ON but XRT headers/libs not found. Please install XRT or set XRT_INCLUDE_DIR/XRT_LIB variables.")
    else()
        message(STATUS "Found XRT: include=${XRT_INCLUDE_DIR} lib=${XRT_LIB}")
        target_include_directories(llama PRIVATE ${XRT_INCLUDE_DIR})
        
        #target_link_libraries(llama PRIVATE ${XRT_LIB})
        #target_compile_definitions(llama PRIVATE USE_FPGA)

        # Link thêm các thư viện phụ trợ thường cần cho XRT
        target_link_libraries(llama PRIVATE ${XRT_LIB} -luuid -lstdc++fs)
        # 3. Quan trọng: PUBLIC để main.cpp cũng thấy define này
        target_compile_definitions(llama PUBLIC USE_FPGA)
    endif()
endif()
