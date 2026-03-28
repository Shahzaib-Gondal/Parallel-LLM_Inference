#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

// ==========================================
// MEMBER 4: Logging & Validation
// ==========================================
#include "Logger.h"

// ==========================================
// MEMBER 1: Job API & Storage
// ==========================================
//#pragma once
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "InferenceAPI.h"

// ==========================================
// MEMBER 3: Model Execution
// ==========================================
#include "ModelWrapper.h"

// ==========================================
// MEMBER 2: Queue, Dispatcher & Workers
// ==========================================
#include "ThreadSafeQueue.h"
#include "WorkerPool.h"
#include "JobDispatcher.h"

int main() {
    Logger logger; 
    logger.log("=== STARTING PARALLEL INFERENCE ENGINE (MILESTONE 1) ===", LogLevel::INFO);

    // 1. Initialize Member 4's Logger
    logger.log("System booting up...", LogLevel::INFO);

    // 2. Initialize Member 3's Model Wrapper
    // (Pointing to wherever they decide to load the TinyLLaMA model)
    ModelWrapper llm_model("models/tinyllama.gguf");

    // 3. Initialize Member 1's Result Storage 
    // (This holds the finished results so the API can fetch them later)
    ResultsStorage results_db;

    // 4. Initialize Member 2's Queues (Backpressure enabled!)
    ThreadSafeQueue<InferenceJob> pending_queue(100);   // High capacity for incoming API traffic
    ThreadSafeQueue<InferenceJob> execution_queue(10);  // Low capacity to throttle the workers

    // 5. Wire up Member 2's Orchestration Layer
    // The workers need to know about the queue, the model (to run it), and the results_db (to save output)
    WorkerPool worker_pool(4, execution_queue, results_db, llm_model, logger);
    
    // The dispatcher moves jobs from pending -> execution
    JobDispatcher dispatcher(pending_queue, execution_queue, logger);

    // 6. Wire up Member 1's Frontend API
    // The API needs the pending queue to push jobs, and the results_db to fetch answers
    InferenceAPI api(pending_queue, results_db, logger);

    logger.log("[Main] All subsystems online. Simulating incoming web traffic...", LogLevel::INFO);

    // ==========================================
    // SIMULATING THE LIFECYCLE
    // ==========================================

    std::vector<std::string> test_prompts = {
        "What is the capital of France?",
        "Write a poem about C++.",
        "Explain backpressure.",
        "What is 2+2?",
        "Tell me a joke."
    };


    std::vector<std::string> tracked_job_ids;

    // A. Submit Requests asynchronously via Member 1's API
    for (const auto& prompt : test_prompts) {
        std::string job_id = api.submit_req(prompt, 128, 0.7f);
        tracked_job_ids.push_back(job_id);
        logger.log("[Client] Submitted: '" + prompt + "' -> Assigned ID: " + job_id, LogLevel::INFO);
    }

    logger.log("[Main] Traffic burst complete. Waiting for engine to process...", LogLevel::INFO);

    // Allow workers time to process the simulated load
    bool all_done = false;
    while (!all_done) {
    all_done = true;
    for (const auto& id : tracked_job_ids) {
        if (api.check_status(id) != JobStatus::COMPLETED) {
            all_done = false;
            break;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small pause so we don't spam the CPU
}

    // B. Fetch Results via Member 1's API
    for (const auto& job_id : tracked_job_ids) {
        // Here you would use Member 1's check_status or fetch_result logic
        JobStatus status = api.check_status(job_id);
        
        if (status == JobStatus::COMPLETED) {
            InferenceResult final_answer = api.fetch_result(job_id);
            logger.log(">>> Result for Job " + job_id + ": " + final_answer.output, LogLevel::INFO);
        } else {
            logger.log(">>> Job " + job_id + " is still processing or failed.", LogLevel::WARNING);
        }
    }

    // ==========================================
    // GRACEFUL SHUTDOWN (Member 2)
    // ==========================================
    logger.log("[Main] Initiating Graceful Shutdown...", LogLevel::INFO);
    
    // 1. Shut down the API queue first. This allows the Dispatcher thread to exit.
    pending_queue.shutdown();
    
    // 2. Shut down the execution queue. This allows the Worker threads to exit.
    execution_queue.shutdown();

    logger.log("=== SYSTEM SHUTDOWN COMPLETE ===", LogLevel::INFO);

    return 0; // All destructors fire here, cleanly joining the threads.
}