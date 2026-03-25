#pragma once
#include <string>
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
};
