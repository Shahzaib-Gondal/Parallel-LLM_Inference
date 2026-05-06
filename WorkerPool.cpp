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

void WorkerPool::pintocore(int core_id) {
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    if (SetThreadAffinityMask(hThread, mask) == 0) {
        logger_.log("Failed to pin Worker " + std::to_string(core_id) + " to core", LogLevel::LOG_ERROR);
    }
}

void WorkerPool::efficiency_measure(int worker_id) {
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
    llm.thread_context(worker_id);  // register this thread's llama context
    pintocore(worker_id);

    std::vector<InferenceJob> current_batch;

    while (job_queue_.wait_and_pop(current_batch)) {
        logger_.log("[Worker " + std::to_string(worker_id) + "] Received a batch of "
                    + std::to_string(current_batch.size()) + " jobs", LogLevel::INFO);

        auto batch_inference_start = std::chrono::high_resolution_clock::now();

        std::vector<std::string> batch_prompts;
        for (auto& job : current_batch) {
            job.inference_start = batch_inference_start;
            job.queue_wait_ms   = std::chrono::duration<double, std::milli>(
                                      batch_inference_start - job.enqueue_time).count();
            batch_prompts.push_back(job.prompt);
        }

        std::vector<InferenceResult> batch_outputs;

        try {
            batch_outputs = llm.run_batch_inference(
                batch_prompts,
                current_batch[0].tokens,
                current_batch[0].temperature,
                current_batch[0].top_p
            );
        } catch (const std::exception& e) {
            std::string error_msg = e.what();
            logger_.log("[Worker " + std::to_string(worker_id)
                        + "] Critical Batch Failure: " + error_msg, LogLevel::LOG_ERROR);
            for (auto& job : current_batch) {
                job.output = {"", InferenceStatus::FAILURE_RUNTIME_ERROR, error_msg, 0};
                results_store.update_res(job.jobid, job.output);
            }
            continue;
        }

        auto batch_inference_end = std::chrono::high_resolution_clock::now();

        for (auto& job : current_batch) {
            job.inference_end  = batch_inference_end;
            job.inference_ms   = std::chrono::duration<double, std::milli>(
                                     job.inference_end - job.inference_start).count();
            job.e2e_latency_ms = std::chrono::duration<double, std::milli>(
                                     job.inference_end - job.enqueue_time).count();
        }

        // populate job.output from batch_outputs BEFORE counting success
        for (int j = 0; j < (int)current_batch.size() && j < (int)batch_outputs.size(); j++) {
            current_batch[j].output = batch_outputs[j];
        }

       
        efficiency_measure(worker_id);

        double batch_inference_ms  = 0;
        double batch_e2e_ms        = 0;
        double batch_queue_wait_ms = 0;
        int    batch_success_count = 0;
        int    batch_tokens        = 0;

        for (auto& job : current_batch) {
            batch_inference_ms  += job.inference_ms;
            batch_e2e_ms        += job.e2e_latency_ms;
            batch_queue_wait_ms += job.queue_wait_ms;
            if (job.output.status == InferenceStatus::SUCCESS)
                batch_success_count++;
            batch_tokens += job.output.tokens_generated;
        }

        results_store.update_res_batch(current_batch, batch_outputs);

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            total_inference_ms_  += batch_inference_ms;
            total_e2e_ms_        += batch_e2e_ms;
            total_queue_wait_ms_ += batch_queue_wait_ms;
            completed_jobs_      += (int)current_batch.size();
            total_tokens         += batch_tokens;
        }

        logger_.log("[Worker " + std::to_string(worker_id)
                    + "] Batch Finished. Size: " + std::to_string(current_batch.size())
                    + " | Success: " + std::to_string(batch_success_count)
                    + "/" + std::to_string(current_batch.size()), LogLevel::INFO);
    }

    logger_.log("[Worker " + std::to_string(worker_id) + "] Shutting down.", LogLevel::INFO);
}

WorkerPool::WorkerPool(size_t num_threads,
                       ThreadSafeQueue<vector<InferenceJob>>& queue,
                       ResultsStorage& results,
                       ModelWrapper& llm,
                       Logger& logger)
    : job_queue_(queue), results_store(results), llm(llm), logger_(logger),
      total_inference_ms_(0.0), total_e2e_ms_(0.0), total_queue_wait_ms_(0.0),
      completed_jobs_(0), total_tokens(0)
{
    logger_.log("Starting Worker Pool with " + std::to_string(num_threads)
                + " threads...", LogLevel::INFO);

    for (size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back(&WorkerPool::worker_loop, this, static_cast<int>(i));
}

WorkerPool::~WorkerPool() {
    for (thread& worker : workers_)
        if (worker.joinable()) worker.join();
    logger_.log("Worker Pool successfully destroyed.", LogLevel::INFO);
}


WorkerPool::LatencyStats WorkerPool::get_latency_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    LatencyStats s{};
    if (completed_jobs_ > 0) {
        s.avg_inference_ms  = total_inference_ms_  / completed_jobs_;
        s.avg_e2e_ms        = total_e2e_ms_        / completed_jobs_;
        s.avg_queue_wait_ms = total_queue_wait_ms_ / completed_jobs_;
        s.jobs_completed    = completed_jobs_;
        s.total_tokens      = total_tokens;
    }
    return s;
}