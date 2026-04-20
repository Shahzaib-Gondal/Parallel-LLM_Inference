//#pragma once
#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "WorkerPool.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"
#include <vector>
#include <thread>
#include <iostream>
#include <chrono>
#include <windows.h>

using namespace std;

// --- m2: pin worker thread to a logical core for cache locality ---
void WorkerPool::pintocore(int core_id){
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    if (SetThreadAffinityMask(hThread, mask) == 0) {
        logger_.log("Failed to pin Worker " + std::to_string(core_id) + " to core", LogLevel::LOG_ERROR);
    }
}

// --- m2: log kernel/user CPU time for this worker thread ---
void WorkerPool::efficiency_measure(int worker_id){
    FILETIME ftCreation, ftExit, ftKernel, ftUser;
    if (GetThreadTimes(GetCurrentThread(), &ftCreation, &ftExit, &ftKernel, &ftUser)) {
        ULARGE_INTEGER kernel, user;
        kernel.LowPart  = ftKernel.dwLowDateTime;
        kernel.HighPart = ftKernel.dwHighDateTime;
        user.LowPart    = ftUser.dwLowDateTime;
        user.HighPart   = ftUser.dwHighDateTime;

        logger_.log("[Worker " + std::to_string(worker_id) + "] Telemetry - Kernel: "
                    + std::to_string(kernel.QuadPart) + " User: "
                    + std::to_string(user.QuadPart), LogLevel::INFO);
    }
}

void WorkerPool::worker_loop(int worker_id) {
    pintocore(worker_id);
    InferenceJob current_job;

    while (job_queue_.wait_and_pop(current_job)) {
        logger_.log("[Worker " + std::to_string(worker_id) + "] Started processing Job "
                    + current_job.jobid, LogLevel::INFO);

        // --- Latency: measure queue wait time ---
        current_job.inference_start = std::chrono::high_resolution_clock::now();
        current_job.queue_wait_ms   = std::chrono::duration<double, std::milli>(
            current_job.inference_start - current_job.enqueue_time).count();

        // --- Inference ---
        try {
            current_job.output = llm.run_inference(
                current_job.prompt,
                current_job.tokens,
                current_job.temperature,
                current_job.top_p
            );
        } catch (const std::exception& e) {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Exception during generation for Job "
                        + current_job.jobid + ": " + e.what(), LogLevel::LOG_ERROR);
            current_job.output = {"", InferenceStatus::FAILURE_RUNTIME_ERROR, e.what(), 0};
        }

        // --- Latency: record inference end and compute both durations ---
        current_job.inference_end   = std::chrono::high_resolution_clock::now();
        current_job.inference_ms    = std::chrono::duration<double, std::milli>(
            current_job.inference_end - current_job.inference_start).count();
        current_job.e2e_latency_ms  = std::chrono::duration<double, std::milli>(
            current_job.inference_end - current_job.enqueue_time).count();

        logger_.log("[Worker " + std::to_string(worker_id) + "] Job " + current_job.jobid
                    + " | queue_wait=" + std::to_string(current_job.queue_wait_ms)   + " ms"
                    + " | inference="  + std::to_string(current_job.inference_ms)    + " ms"
                    + " | e2e="        + std::to_string(current_job.e2e_latency_ms)  + " ms",
                    LogLevel::INFO);

        efficiency_measure(worker_id);
        results_store.update_res(current_job.jobid, current_job.output);

        // --- Accumulate into pool-level stats (thread-safe) ---
        // Using fetch_add on the raw bits of a double is UB; use a mutex-guarded accumulator instead.
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            total_inference_ms_  += current_job.inference_ms;
            total_e2e_ms_        += current_job.e2e_latency_ms;
            total_queue_wait_ms_ += current_job.queue_wait_ms;
            completed_jobs_++;
        }

        if (current_job.output.status == InferenceStatus::SUCCESS) {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Successfully Finished Job "
                        + current_job.jobid, LogLevel::INFO);
        } else {
            logger_.log("[Worker " + std::to_string(worker_id) + "] Failed Job "
                        + current_job.jobid, LogLevel::LOG_ERROR);
        }
    }

    logger_.log("[Worker " + std::to_string(worker_id) + "] Shutting down.", LogLevel::INFO);
}

WorkerPool::WorkerPool(size_t num_threads,
                       ThreadSafeQueue<InferenceJob>& queue,
                       ResultsStorage& results,
                       ModelWrapper& llm,
                       Logger& logger)
    : job_queue_(queue), results_store(results), llm(llm), logger_(logger),
      total_inference_ms_(0.0), total_e2e_ms_(0.0),
      total_queue_wait_ms_(0.0), completed_jobs_(0)
{
    logger_.log("Starting Worker Pool with " + std::to_string(num_threads) + " threads...", LogLevel::INFO);

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&WorkerPool::worker_loop, this, static_cast<int>(i));
    }
}

WorkerPool::~WorkerPool() {
    for (thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    logger_.log("Worker Pool successfully destroyed.", LogLevel::INFO);
}

// --- Latency accessors (call AFTER destructor joins all threads) ---
WorkerPool::LatencyStats WorkerPool::get_latency_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    LatencyStats s{};
    if (completed_jobs_ > 0) {
        s.avg_inference_ms   = total_inference_ms_   / completed_jobs_;
        s.avg_e2e_ms         = total_e2e_ms_         / completed_jobs_;
        s.avg_queue_wait_ms  = total_queue_wait_ms_  / completed_jobs_;
        s.jobs_completed     = completed_jobs_;
    }
    return s;
}
