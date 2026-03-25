#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "InferenceAPI.h"
#include "ModelWrapper.h"
#include "ThreadSafeQueue.h"
#include "WorkerPool.h"
#include "JobDispatcher.h"

int main() {
    std::cout << "=== STARTING PARALLEL INFERENCE ENGINE (MILESTONE 1) ===\n\n";


    ModelWrapper llm_model("models/tinyllama.gguf");

    ResultsStorage results_db;

    ThreadSafeQueue<InferenceJob> pending_queue(100);   // High capacity for incoming API traffic
    ThreadSafeQueue<InferenceJob> execution_queue(10);  // Low capacity to throttle the workers

    WorkerPool worker_pool(4, execution_queue, results_db, llm_model);
    
    JobDispatcher dispatcher(pending_queue, execution_queue);

    InferenceAPI api(pending_queue, results_db);

    std::cout << "[Main] All subsystems online. Simulating incoming web traffic...\n\n";

    std::vector<std::string> test_prompts = {
        "What is the capital of France?",
        "Write a poem about C++.",
        "Explain backpressure.",
        "What is 2+2?",
        "Tell me a joke."
    };


    std::vector<std::string> tracked_job_ids;

    for (const auto& prompt : test_prompts) {
        std::string job_id = api.submit_req(prompt, 128, 0.7f);
        tracked_job_ids.push_back(job_id);
        std::cout << "[Client] Submitted: '" << prompt << "' -> Assigned ID: " << job_id << "\n";
    }

    std::cout << "\n[Main] Traffic burst complete. Waiting for engine to process...\n\n";

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

    for (const auto& job_id : tracked_job_ids) {
        JobStatus status = api.check_status(job_id);
        
        if (status == JobStatus::COMPLETED) {
            InferenceResult final_answer = api.fetch_result(job_id);
            std::cout << ">>> Result for Job " << job_id << ": " << final_answer.output << "\n";
        } else {
            std::cout << ">>> Job " << job_id << " is still processing or failed.\n";
        }
    }

    std::cout << "\n[Main] Initiating Graceful Shutdown...\n";
    
    pending_queue.shutdown();
    
    execution_queue.shutdown();

    std::cout << "=== SYSTEM SHUTDOWN COMPLETE ===\n";

    return 0;
}