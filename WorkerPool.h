#pragma once

#include <vector>
#include <thread>
#include <cstddef> // for size_t
#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"

class WorkerPool {
private:
    std::vector<std::thread> workers_;
    ThreadSafeQueue<vector<InferenceJob>>& job_queue_;
    void worker_loop(int worker_id);
    ResultsStorage& results_store;
    ModelWrapper& llm;

public:
    WorkerPool(size_t num_threads, ThreadSafeQueue<vector<InferenceJob>>& queue, ResultsStorage& results, ModelWrapper& llm);
    ~WorkerPool();
};