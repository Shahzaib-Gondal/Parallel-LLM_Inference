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
    ThreadSafeQueue<InferenceJob>& job_queue_;
    void worker_loop(int worker_id);
    //m2 funcs to measure
    void pintocore(int core_id);
    void efficiency_measure(int worker_id);
    ResultsStorage& results_store;
    ModelWrapper& llm;
    Logger& logger_;

public:
    WorkerPool(size_t num_threads, ThreadSafeQueue<InferenceJob>& queue, ResultsStorage& results, ModelWrapper& llm, Logger& logger);
    ~WorkerPool();
};