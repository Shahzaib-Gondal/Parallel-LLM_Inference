#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>
#include "JobDispatcher.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"
#include "WorkerPool.h"
#include "InferenceAPI.h"

class InferenceEngine {
private:
    Logger logger;
    ResultsStorage storage;
    ThreadSafeQueue<InferenceJob> pending_q;          // Fed by API
    ThreadSafeQueue<std::vector<InferenceJob>> exec_q; // Fed by Dispatcher

    ModelWrapper model;
    std::unique_ptr<InferenceAPI> api;
    std::unique_ptr<JobDispatcher> dispatcher;
    std::unique_ptr<WorkerPool> workers;

public:
    InferenceEngine(std::string path, int n_workers = 2, int b_size = 4, int threads = 4)
        : pending_q(256),
          exec_q(64),
          model(path, 2048, threads) 
    {
        api = std::make_unique<InferenceAPI>(pending_q, storage, logger);
        dispatcher = std::make_unique<JobDispatcher>(pending_q, exec_q, logger);
        workers = std::make_unique<WorkerPool>(n_workers, exec_q, storage, model, logger);
        logger.log("[Engine] Integrated System Online.", LogLevel::INFO);
    }

    // Main program can now call submit_req
    InferenceAPI* get_api() { return api.get(); }

    ~InferenceEngine() {
        pending_q.shutdown(); 
        dispatcher.reset(); 
        exec_q.shutdown();
        workers.reset();
        logger.log("[Engine] Controlled Shutdown Complete.", LogLevel::INFO);
    }
};