Building all files together:
g++ main.cpp WorkerPool.cpp InferenceAPI.cpp ModelWrapper.cpp -I"llama.cpp/include" -I"llama.cpp/ggml/include" -L"llama.cpp/build/src" -L"llama.cpp/build/ggml/src" -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -lgomp -lws2_32 -lrpcrt4 -pthread -o engine.exe

Getting llama2c:
git clone https://github.com/ggerganov/llama.cpp 
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
cmake --build . --config Release 

