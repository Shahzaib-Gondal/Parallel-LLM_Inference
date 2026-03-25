#include "ModelWrapper.h"
#include <iostream>
#include <cassert>

void test_basic(ModelWrapper& w) {
    std::cout << "\n--- test: basic inference ---\n";
    auto r = w.run_inference("What is 2 + 2?", 50);
    assert(r.status == InferenceStatus::SUCCESS);
    std::cout << "Output: " << r.output << "\n";
    std::cout << "Tokens: " << r.tokens_generated << "\n";
}

void test_empty(ModelWrapper& w) {
    std::cout << "\n--- test: empty prompt ---\n";
    auto r = w.run_inference("", 50);
    assert(r.status == InferenceStatus::FAILURE_EMPTY_PROMPT);
    std::cout << "Correctly rejected: " << r.error_message << "\n";
}

void test_determinism(ModelWrapper& w) {
    std::cout << "\n--- test: determinism ---\n";
    auto r1 = w.run_inference("Tell me a fact about cats.", 40);
    auto r2 = w.run_inference("Tell me a fact about cats.", 40);
    std::cout << "Run 1: " << r1.output << "\n";
    std::cout << "Run 2: " << r2.output << "\n";
    if (r1.output == r2.output)
        std::cout << "PASS: outputs match\n";
    else
        std::cout << "WARN: outputs differ\n";
}

int main(int argc, char* argv[]) {
    const std::string model_path = (argc > 1)
        ? argv[1] : "../models/tinyllama.gguf";

    std::cout << "Loading: " << model_path << "\n";
    try {
        ModelWrapper wrapper(model_path, 512, 4, 42);
        test_basic(wrapper);
        test_empty(wrapper);
        test_determinism(wrapper);
        std::cout << "\nAll tests done.\n";
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}