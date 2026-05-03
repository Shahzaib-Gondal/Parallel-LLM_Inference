#include "ModelWrapper.h"
#include "WorkerPool.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 

struct TestCase {
    std::string prompt;
    std::string expected_keyword;
};

void run_concurrency_check(ModelWrapper& llm, int num_threads) {
    //setting up prompts that TinyLLaMA should be able to complete correctly if KV-Cache isolation is maintained
    std::vector<TestCase> test_cases = {
        {"The color of the clear daytime sky is usually", "blue"},
        {"The capital city of France is called", "paris"},
        {"A large animal with a trunk that lives in Africa is called a", "elephant"},
        {"To bake a cake, the most essential appliance is an", "oven"},
        {"What is the number that comes immediately after one?", "two"},
        {"The fruit that shows up in the summer season only is", "mango"},
        {"The opposite of the word 'hot' is the word", "cold"},
        {"The sun rises in the morning from the", "east"}
    };

    //limiting num_threads to the size of our test cases
    int actual_threads = std::min(num_threads, (int)test_cases.size());
    std::vector<std::string> prompts;
    for (int i = 0; i < actual_threads; ++i) {
        prompts.push_back(test_cases[i].prompt);
    }

    std::cout << "[Verifier] Launching " << actual_threads << " completion-based threads...\n";

    //using Temperature 0.0 for deterministic results during verification
    std::vector<InferenceResult> results = llm.run_batch_inference(prompts, 15, 0.0f, 1.0f);

    int failures = 0;
    for (int i = 0; i < results.size(); ++i) {
        std::string output = results[i].output;
        // Convert to lowercase for easier matching
        std::transform(output.begin(), output.end(), output.begin(), ::tolower);
        
        std::string expected = test_cases[i].expected_keyword;

        //Correctness(Did it finish its own sentence?)
        if (output.find(expected) == std::string::npos) {
            std::cerr << "ERROR: Thread " << i << " (Topic: " << expected 
                      << ") failed. Produced: \"" << results[i].output << "\"\n";
            failures++;
        } else {
            std::cout << "Thread " << i << " maintained integrity (" << expected << ")\n Produced: \"" << results[i].output << "\"\n";
        }

        // 2. Cross-Contamination(Did another thread's context leak here?)
        for (int j = 0; j < actual_threads; ++j) {
            if (i == j) continue;
            std::string forbidden = test_cases[j].expected_keyword;
            if (output.find(forbidden) != std::string::npos) {
                std::cerr << "CRITICAL LEAK: Thread " << i << " leaked context from Thread " << j 
                          << "! (Found '" << forbidden << "' in output)\n";
                failures++;
            }
        }
    }

    std::cout << "\n--- Verification Report ---\n";
    if (failures == 0) {
        std::cout << "SUCCESS: All " << actual_threads << " threads maintained strict KV-Cache isolation.\n";
    } else {
        std::cout << "Verification FAILED: " << failures << " issues detected.\n";
    }
}

int main() {
    try {
        ModelWrapper llm_model("models/tinyllama.gguf", 512, 4); 
        run_concurrency_check(llm_model, 8); 
    } catch (const std::exception& e) {
        std::cerr << "Exception during test: " << e.what() << "\n";
        return 1;
    }
    return 0;
}