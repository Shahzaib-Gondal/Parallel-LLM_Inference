#pragma once

#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include <thread>
#include <iostream>
#include <atomic>

class JobDispatcher {
private:
    ThreadSafeQueue<InferenceJob>& pending_queue_;
    ThreadSafeQueue<InferenceJob>& execution_queue_;
    std::thread dispatcher_thread_;

    void dispatch_loop() {
        InferenceJob current_job;

        std::cout << "[Dispatcher] Online. Listening for incoming requests...\n";

        // Pull from the API queue
        while (pending_queue_.wait_and_pop(current_job)) {
            
            std::cout << "[Dispatcher] Received Job " << current_job.jobid << ". Analyzing...\n";

            //update the state and pass it immediately to the workers.

            current_job.status = JobStatus::RUNNING; // update state tracking
            
            //push to the worker pool's queue
            execution_queue_.push(current_job);


        }

        std::cout << "[Dispatcher] Shutting down...\n";
    }

public:
    JobDispatcher(ThreadSafeQueue<InferenceJob>& api_queue, 
                  ThreadSafeQueue<InferenceJob>& worker_queue)
        : pending_queue_(api_queue), execution_queue_(worker_queue) {
        
        // start the dispatcher thread
        dispatcher_thread_ = std::thread(&JobDispatcher::dispatch_loop, this);
    }

    ~JobDispatcher() {
        if (dispatcher_thread_.joinable()) {
            dispatcher_thread_.join();
        }
    }
};