#pragma once

#include <vector>
#include <thread>
#include <cstddef> // for size_t
#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"
#include "Logger.h"

class WorkerPool {
private:
    std::vector<std::thread> workers_;
    //ThreadSafeQueue<InferenceJob>& job_queue_;
    ThreadSafeQueue<vector<InferenceJob>>& job_queue_;
    void worker_loop(int worker_id);
    //m2 funcs to measure
    void pintocore(int core_id);
    void efficiency_measure(int worker_id);
    ResultsStorage& results_store;
    ModelWrapper& llm;
    Logger& logger_;
    mutable std::mutex stats_mutex_;
    double total_inference_ms_   = 0.0;
    double total_e2e_ms_         = 0.0;
    double total_queue_wait_ms_  = 0.0;
    int    completed_jobs_       = 0;
    long long total_tokens = 0;

public:
    //WorkerPool(size_t num_threads, ThreadSafeQueue<InferenceJob>& queue, ResultsStorage& results, ModelWrapper& llm, Logger& logger);
    WorkerPool(size_t num_threads, ThreadSafeQueue<vector<InferenceJob>>& queue, ResultsStorage& results, ModelWrapper& llm, , Logger& logger);
    ~WorkerPool();
     struct LatencyStats {
        double avg_inference_ms  = 0.0;  //pure model execution time just inference
        double avg_e2e_ms        = 0.0;  //end2end time
        double avg_queue_wait_ms = 0.0;  //time sitting in queue
        int    jobs_completed    = 0;
        long long total_tokens = 0;
    };
 
    LatencyStats get_latency_stats() const;
};