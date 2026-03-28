#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cassert>

#include "Logger.h"
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "InferenceAPI.h"
#include "ModelWrapper.h"
#include "ThreadSafeQueue.h"
#include "WorkerPool.h"
#include "JobDispatcher.h"

Logger logger;

void test_invalid_inputs(InferenceAPI& api, ResultsStorage& results_db) {
    logger.log(">>> Running Test: Invalid Inputs", LogLevel::INFO);
    
    // Test 1: Empty prompt
    std::string id1 = api.submit_req("", 128, 0.7f);
    assert(api.check_status(id1) == JobStatus::FAILED);
    assert(api.fetch_result(id1).status == InferenceStatus::FAILURE_EMPTY_PROMPT);
    
    // Test 2: Invalid tokens
    std::string id2 = api.submit_req("Hello", -5, 0.7f);
    assert(api.check_status(id2) == JobStatus::FAILED);
    assert(api.fetch_result(id2).status == InferenceStatus::FAILURE_TOKEN_LIMIT);

    // Test 3: Invalid temperature
    std::string id3 = api.submit_req("Hello", 128, 2.5f);
    assert(api.check_status(id3) == JobStatus::FAILED);
    assert(api.fetch_result(id3).status == InferenceStatus::FAILURE_RUNTIME_ERROR);

    logger.log(">>> Passed Test: Invalid Inputs", LogLevel::INFO);
}

void test_concurrent_stress(InferenceAPI& api, int num_requests) {
    logger.log(">>> Running Test: Concurrent Stress Test (" + std::to_string(num_requests) + " requests)", LogLevel::INFO);
    
    std::vector<std::thread> clients;
    std::vector<std::string> job_ids(num_requests);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_requests; ++i) {
        clients.emplace_back([&, i]() {
            std::string prompt = "Stress test prompt " + std::to_string(i);
            job_ids[i] = api.submit_req(prompt, 50, 0.7f);
        });
    }

    for (auto& client : clients) {
        client.join();
    }

    // Wait for all to finish
    bool all_done = false;
    while (!all_done) {
        all_done = true;
        for (const auto& id : job_ids) {
            JobStatus status = api.check_status(id);
            if (status != JobStatus::COMPLETED && status != JobStatus::FAILED) {
                all_done = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    logger.log(">>> Passed Test: Concurrent Stress Test in " + std::to_string(diff.count()) + " seconds", LogLevel::INFO);
}

int main(int argc, char* argv[]) {
    logger.log("=== STARTING M1 VERIFICATION TESTS ===", LogLevel::INFO);
    
    const std::string model_path = (argc > 1) ? argv[1] : "../models/tinyllama.gguf";

    try {
        ModelWrapper llm_model(model_path);
        ResultsStorage results_db;
        ThreadSafeQueue<InferenceJob> pending_queue(100);
        ThreadSafeQueue<InferenceJob> execution_queue(10);
        
        WorkerPool worker_pool(2, execution_queue, results_db, llm_model, logger);
        JobDispatcher dispatcher(pending_queue, execution_queue, logger);
        InferenceAPI api(pending_queue, results_db, logger);

        test_invalid_inputs(api, results_db);
        
        // Concurrent submission / stress test
        test_concurrent_stress(api, 15);

        logger.log("Initiating Shutdown...", LogLevel::INFO);
        pending_queue.shutdown();
        execution_queue.shutdown();

    } catch (const std::exception& e) {
        logger.log(std::string("FATAL ERROR: ") + e.what(), LogLevel::LOG_ERROR);
        return 1;
    }

    logger.log("=== ALL TESTS PASSED ===", LogLevel::INFO);
    return 0;
}
