#include "ModelWrapper.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <io.h>
#include <fcntl.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[worker] Usage: worker_process <model_path>\n";
        return 1;
    }

    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    ModelWrapper* wrapper = nullptr;
    try {
        wrapper = new ModelWrapper(argv[1], 512, 4, 42);
    } catch (const std::exception& e) {
        std::cout << "ERROR:Failed to load model: " << e.what() << "\n";
        std::cout.flush();
        return 1;
    }

    std::cout << "READY\n";
    std::cout.flush();

    std::string prompt;
    while (std::getline(std::cin, prompt)) {
        if (prompt == "SHUTDOWN") break;
        if (prompt.empty()) {
            std::cout << "ERROR:Empty prompt\n";
            std::cout.flush();
            continue;
        }

        InferenceResult result = wrapper->run_inference(prompt, 50, 0.7f, 0.9f);

        if (result.status == InferenceStatus::SUCCESS) {
            std::string out = result.output;
            std::replace(out.begin(), out.end(), '\n', ' ');
            std::replace(out.begin(), out.end(), '\r', ' ');
            std::cout << "OK:" << out << "\n";
        } else {
            std::cout << "ERROR:" << result.error_message << "\n";
        }
        std::cout.flush();
    }

    delete wrapper;
    return 0;
}