#pragma once

#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "Logger.h"
#include <thread>
#include <atomic>

class JobDispatcher {
private:
    ThreadSafeQueue<InferenceJob>& pending_queue_;
    ThreadSafeQueue<InferenceJob>& execution_queue_;
    std::thread dispatcher_thread_;
    Logger& logger_;

    void dispatch_loop() {
        InferenceJob current_job;

        logger_.log("[Dispatcher] Online. Listening for incoming requests...", LogLevel::INFO);

        // Pull from the API queue
        while (pending_queue_.wait_and_pop(current_job)) {
            
            logger_.log("[Dispatcher] Received Job " + current_job.jobid + ". Analyzing...", LogLevel::DEBUG);

            // ========================================================
            // MILESTONE 1 LOGIC:
            // Just update the state and pass it immediately to the workers.
            // ========================================================
            current_job.status = JobStatus::RUNNING; // Update state tracking
            
            // Push to the worker pool's queue
            execution_queue_.push(current_job);


        }

        logger_.log("[Dispatcher] Shutting down...", LogLevel::INFO);
    }

public:
    JobDispatcher(ThreadSafeQueue<InferenceJob>& api_queue, 
                  ThreadSafeQueue<InferenceJob>& worker_queue,
                  Logger& logger)
        : pending_queue_(api_queue), execution_queue_(worker_queue), logger_(logger) {
        
        // start the dispatcher thread
        dispatcher_thread_ = std::thread(&JobDispatcher::dispatch_loop, this);
    }

    ~JobDispatcher() {
        if (dispatcher_thread_.joinable()) {
            dispatcher_thread_.join();
        }
    }
};