#pragma once
#include <string>
#include <chrono>
#include "InferenceResult.h"

//JobCycle states
enum class JobStatus{
    PENDING, 
    RUNNING, 
    COMPLETED, 
    FAILED,
    CANCELLED
};

struct InferenceJob{
    std::string jobid;
    std::string prompt;
    //llm params with defaults set 
    int tokens = 128; 
    float temperature = 0.7f;
    float top_p = 0.9f;
    JobStatus status = JobStatus::PENDING; //tracking request across states
    //output
    InferenceResult output;


    // Set by the JobDispatcher before pushing to queue
    std::chrono::high_resolution_clock::time_point enqueue_time{};
    // Set by the workerthread just before inference begins
    std::chrono::high_resolution_clock::time_point inference_start{};
    // Set by the workerthread immediately after inference completes
    std::chrono::high_resolution_clock::time_point inference_end{};

    //populated by workerthread after inference
    double queue_wait_ms  = 0.0; // time spent waiting in queue before worker picks it up
    double inference_ms   = 0.0; // pure model execution time
    double e2e_latency_ms = 0.0; // (queue_wait + inference = end to end)
};
