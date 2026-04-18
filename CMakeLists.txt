cmake_minimum_required(VERSION 3.14)
project(m2_multiprocess)
set(CMAKE_CXX_STANDARD 17)

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
)

add_executable(m2_benchmark
    m2_benchmark.cpp
    ProcessPool.cpp
)