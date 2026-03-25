cmake_minimum_required(VERSION 3.14)
project(inference_engine)
set(CMAKE_CXX_STANDARD 17)

set(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

include_directories(${LLAMA_ROOT}/include)
include_directories(${LLAMA_ROOT}/ggml/include)

add_executable(inference_test
    main_test.cpp
    ModelWrapper.cpp
)

target_link_libraries(inference_test
    "${LLAMA_ROOT}/build/src/Release/llama.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml-base.lib"
)