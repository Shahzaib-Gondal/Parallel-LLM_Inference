#include "ModelWrapper.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>

struct TestCase {
    std::string prompt;
    std::string expected_keyword;
};

struct ThreadResult {
    int thread_id;
    std::string topic;
    InferenceResult result;
};

std::mutex cout_mutex;

void worker_task(int id,
                 ModelWrapper* wrapper,
                 const std::string& prompt,
                 ThreadResult* out)
{
    // Each thread MUST initialize its own llama context before inference
    wrapper->thread_context(id);

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[Thread " << id << "] Context ready, running inference...\n";
    }

    out->result = wrapper->run_inference(prompt, 15, 0.0f, 1.0f);
}

void run_concurrency_check(ModelWrapper& llm, int num_threads) {
    std::vector<TestCase> test_cases = {
        {"The color of the clear daytime sky is usually",        "blue"},
        {"The capital city of France is called",                  "paris"},
        {"A large animal with a trunk that lives in Africa is a", "elephant"},
        {"To bake a cake, the most essential appliance is an",    "oven"},
        {"What is the number that comes immediately after one?",  "two"},
        {"The fruit that shows up in the summer season only is",  "mango"},
        {"The opposite of the word 'hot' is the word",           "cold"},
        {"The sun rises in the morning from the",                 "east"}
    };

    int actual_threads = std::min(num_threads, (int)test_cases.size());
    std::cout << "[Verifier] Launching " << actual_threads << " threads...\n";

    std::vector<std::thread>    threads(actual_threads);
    std::vector<ThreadResult>   results(actual_threads);

    for (int i = 0; i < actual_threads; ++i) {
        results[i].thread_id = i;
        results[i].topic     = test_cases[i].expected_keyword;
        threads[i] = std::thread(worker_task,
                                 i,
                                 &llm,
                                 test_cases[i].prompt,
                                 &results[i]);
    }

    for (auto& t : threads)
        t.join();

    // ── Verification ────────────────────────────────────────────────────────
    int failures = 0;

    for (int i = 0; i < actual_threads; ++i) {
        std::string output = results[i].result.output;
        std::transform(output.begin(), output.end(), output.begin(), ::tolower);

        const std::string& expected = test_cases[i].expected_keyword;

        // 1. Correctness
        if (output.find(expected) == std::string::npos) {
            std::cerr << "ERROR: Thread " << i
                      << " (Topic: " << expected
                      << ") failed. Produced: \""
                      << results[i].result.output << "\"\n";
            ++failures;
        } else {
            std::cout << "OK  Thread " << i
                      << " (" << expected << "): \""
                      << results[i].result.output << "\"\n";
        }

        // 2. Cross-contamination
        for (int j = 0; j < actual_threads; ++j) {
            if (i == j) continue;
            if (output.find(test_cases[j].expected_keyword) != std::string::npos) {
                std::cerr << "CRITICAL LEAK: Thread " << i
                          << " output contains keyword from Thread " << j
                          << " ('" << test_cases[j].expected_keyword << "')\n";
                ++failures;
            }
        }
    }

    std::cout << "\n--- Verification Report ---\n";
    if (failures == 0)
        std::cout << "Verification PASSED: All " << actual_threads
                  << " threads maintained strict KV-Cache isolation.\n";
    else
        std::cout << "Verification FAILED: " << failures << " issues detected.\n";
}

int main() {
    try {
        // Note: main thread does NOT need to call thread_context()
        // because it never calls run_inference() directly here.
        ModelWrapper llm("D:/SSGC/Downloads/pdc project/llama.cpp/models/tinyllama.gguf",
                         512, 4);
        run_concurrency_check(llm, 8);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}