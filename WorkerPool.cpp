//#pragma once
#include "ThreadSafeQueue.h" // The queue we built earlier
#include "InferenceJob.h"
#include "WorkerPool.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"
#include <vector>
#include <thread>
#include <iostream>
#include <mutex>
#include <chrono>

using namespace std;

void WorkerPool::worker_loop(int worker_id) {
    InferenceJob current_job;

    while (job_queue_.wait_and_pop(current_job)) {
        logger_.log("[Worker " + std::to_string(worker_id) + "] Started processing Job " + current_job.jobid, LogLevel::INFO);

        this_thread::sleep_for(chrono::milliseconds(150)); 
        {
        static std::mutex model_mutex; 
        std::lock_guard<std::mutex> lock(model_mutex);
        try {
            current_job.output = llm.run_inference(current_job.prompt, current_job.tokens, current_job.temperature, current_job.top_p);
        } catch (const std::exception& e) {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Exception during generation for Job " + current_job.jobid + ": " + e.what(), LogLevel::LOG_ERROR);
            current_job.output = {"", InferenceStatus::FAILURE_RUNTIME_ERROR, e.what(), 0};
        }
        }
        results_store.update_res(current_job.jobid, current_job.output);
        
        if (current_job.output.status == InferenceStatus::SUCCESS) {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Successfully Finished Job " + current_job.jobid, LogLevel::INFO);
        } else {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Failed Job " + current_job.jobid, LogLevel::LOG_ERROR);
        }
    }

    logger_.log("[Worker " + std::to_string(worker_id) + "] Shutting down.", LogLevel::INFO);
}

WorkerPool::WorkerPool(size_t num_threads, ThreadSafeQueue<InferenceJob>& queue, ResultsStorage& results, ModelWrapper& llm, Logger& logger) 
    : job_queue_(queue), results_store(results), llm(llm), logger_(logger) {
    
    logger_.log("Starting Worker Pool with " + std::to_string(num_threads) + " threads...", LogLevel::INFO);
    
    for (size_t i = 0; i < num_threads; ++i) {
        // Create a thread and assign it the worker_loop function
        workers_.emplace_back(&WorkerPool::worker_loop, this, i);
    }
}

WorkerPool::~WorkerPool() {
    for (thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join(); // Wait for the thread to finish its current loop
        }
    }
    logger_.log("Worker Pool successfully destroyed.", LogLevel::INFO);
}

/*
int main() {
    std::cout << "--- STARTING THREAD POOL TEST ---\n";

    // 1. Create your queue with backpressure (e.g., max 5 items in queue at once)
    ThreadSafeQueue<InferenceJob> test_queue(10);

    // 2. Spin up the Worker Pool with 3 parallel threads
    WorkerPool pool(3, test_queue);

    // 3. Simulate Member 1's API receiving 10 rapid-fire user requests
    std::cout << "[Main] Incoming web requests...\n";
    for (int i = 1; i <= 10; ++i) {
        InferenceJob new_job;
        new_job.jobid = i;
        new_job.prompt = "Test prompt number " + std::to_string(i);
        new_job.status = JobStatus::PENDING;

        // Push to queue (will block if backpressure kicks in)
        test_queue.push(new_job);
        std::cout << "[Main] Successfully queued Job " << i << "\n";
    }

    std::cout << "[Main] All 10 jobs submitted. Waiting for workers to catch up...\n";

    // Let the threads work for a few seconds
    std::this_thread::sleep_for(std::chrono::seconds(6));

    // 4. Trigger the graceful shutdown
    std::cout << "[Main] Shutting down the system...\n";
    // NOTE: Make sure your ThreadSafeQueue has the shutdown() method we built earlier!
    test_queue.shutdown(); 

    return 0; // When main exits, the WorkerPool destructor automatically joins all threads
}
*/