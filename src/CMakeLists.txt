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
    # Thêm source mới
    #target_sources(llama PRIVATE fpga_host.cpp fpga_host.h)
    
    # QUAN TRỌNG: KHÔNG link thư viện XRT nữa
    # Chỉ cần define macro
    target_compile_definitions(llama PUBLIC USE_FPGA)
    
    # Nếu trước đây có: target_link_libraries(llama PRIVATE xrt_coreutil ...) -> XÓA NÓ ĐI
endif()