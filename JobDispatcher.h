#pragma once

#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "Logger.h"
#include <thread>
#include<iostream>
#include <atomic>

class JobDispatcher {
private:
    ThreadSafeQueue<InferenceJob>& pending_queue_;
    ThreadSafeQueue<std::vector<InferenceJob>>& execution_queue_;
    std::thread dispatcher_thread_;
    size_t max_batch_size_ = 4;
    std::chrono::milliseconds timeout_ = std::chrono::milliseconds(500);
    atomic<bool> is_running{true};
    Logger& logger_;

    void dispatch_loop() {
        vector<InferenceJob>current_batch;
        InferenceJob current_job;

        logger_.log("[Dispatcher] Online. Listening for incoming requests...", LogLevel::INFO);

        // Pull from the API queue
        while (is_running) {
            bool got_job = pending_queue_.wait_and_pop_timeout(current_job, timeout_);
            if(got_job){
                current_job.status = JobStatus::RUNNING; 
                current_batch.push_back(std::move(current_job));
            }

            //update the state and pass it immediately to the workers.
            if((current_batch.size()==max_batch_size_ || !got_job)&& !current_batch.empty()){
                execution_queue_.push(current_batch);
                current_batch.clear();
            }
        }

        logger_.log("[Dispatcher] Shutting down...", LogLevel::INFO);
    }

public:
    JobDispatcher(ThreadSafeQueue<InferenceJob>& api_queue, 
                  ThreadSafeQueue<vector<InferenceJob>>& worker_queue)
        : pending_queue_(api_queue), execution_queue_(worker_queue), logger_(logger) {
        
        // start the dispatcher thread
        dispatcher_thread_ = std::thread(&JobDispatcher::dispatch_loop, this);
    }

    ~JobDispatcher() {
        is_running=false;
        if (dispatcher_thread_.joinable()) {
            dispatcher_thread_.join();
        }
    }
};