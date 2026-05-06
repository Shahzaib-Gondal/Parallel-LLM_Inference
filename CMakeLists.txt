cmake_minimum_required(VERSION 3.14)
project(parallel_llm_inference)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories(${LLAMA_ROOT}/include)
include_directories(${LLAMA_ROOT}/ggml/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(worker_process
    worker_process.cpp
    ModelWrapper.cpp
)
target_link_libraries(worker_process
    "${LLAMA_ROOT}/build/src/Release/llama.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml-base.lib"
)

add_executable(m2_benchmark
    m2_benchmark.cpp
    ProcessPool.cpp
)
target_link_libraries(m2_benchmark
    "${LLAMA_ROOT}/build/src/Release/llama.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml-base.lib"
)

add_executable(m2_threadbenchmark
    m2_threadbenchmark.cpp
    WorkerPool.cpp
    Logger.cpp
    ModelWrapper.cpp
)
target_link_libraries(m2_threadbenchmark
    "${LLAMA_ROOT}/build/src/Release/llama.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml-base.lib"
)
add_executable(m3_batch_benchmark
    m3_batch_benchmark.cpp
    WorkerPool.cpp
    Logger.cpp
    ModelWrapper.cpp
)
target_link_libraries(m3_batch_benchmark
    "${LLAMA_ROOT}/build/src/Release/llama.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml.lib"
    "${LLAMA_ROOT}/build/ggml/src/Release/ggml-base.lib"
)