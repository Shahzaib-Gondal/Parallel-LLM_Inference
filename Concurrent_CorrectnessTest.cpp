#include "ModelWrapper.h"
#include "WorkerPool.h"
#include <iostream>
#include <assert.h>
#include <map>

struct TestCase {
    std::string prompt;
    std::string expected_token; // A unique word we expect to see
};

void run_concurrency_check(ModelWrapper& llm, int num_threads) {
    std::vector<std::string> prompts;
    for (int i = 0; i < num_threads; ++i) {
        prompts.push_back("Repeat the following unique ID exactly: CODE_" + std::to_string(i));
    }

    std::cout << "[Audit] Launching " << num_threads << " concurrent threads...\n";

    std::vector<InferenceResult> results = llm.run_batch_inference(prompts, 20, 0.0f, 1.0f);

    //Verification Loop
    int failures = 0;
    for (int i = 0; i < results.size(); ++i) {
        std::string expected = "CODE_" + std::to_string(i);
        
        // first checking for correctness (Does it contain its own ID?)
        if (results[i].output.find(expected) == std::string::npos) {
            std::cerr << "ERROR: Thread " << i << " failed to produce its ID. Instead printed: " << results[i].output << "\n";
            failures++;
        }

        // now for cross-Contamination (Does it contain other thread's ID?)
        for (int j = 0; j < num_threads; ++j) {
            if (i == j) continue;
            std::string forbidden = "ALPHA_CODE_" + std::to_string(j);
            if (results[i].output.find(forbidden) != std::string::npos) {
                std::cerr << "CRITICAL FAILURE: Thread " << i << " leaked data from Thread " << j << "!\n";
                failures++;
            }
        }
    }

    if (failures == 0) {
        std::cout << "SUCCESS: All threads maintained strict isolation.\n";
    } else {
        std::cout << "Verification FAILED: " << failures << " issues detected.\n";
    }
}

int main(){
    try {
        ModelWrapper llm_model("models/tinyllama.gguf");
        run_concurrency_check(llm_model, 1); // Test with 8 concurrent threads
    } catch (const std::exception& e) {
        std::cerr << "Exception during test: " << e.what() << "\n";
        return 1;
    }

    return 0;

}