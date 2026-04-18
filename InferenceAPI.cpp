#include <windows.h>
#include <rpcdce.h>
#include "InferenceAPI.h"
#include <iostream>

//func to gen unique job ids
std::string InferenceAPI::gen_jobid(){
    UUID uuid;
    UuidCreate(&uuid);
    unsigned char* str;
    UuidToStringA(&uuid, &str);
    std::string jobid((char*)str);
    RpcStringFreeA(&str);
    return jobid;
}

std::string InferenceAPI::submit_req(const std::string& prompt, int tokens, float temp){
    std::string id = gen_jobid();
    InferenceJob job;
    job.jobid = id;
    job.prompt = prompt;
    job.tokens = tokens;
    job.temperature = temp;
    //default stat already set to pending 

    // --- Input Validation ---
    if (prompt.empty()) {
        logger_.log("Job " + id + " failed validation: Empty prompt.", LogLevel::LOG_ERROR);
        job.status = JobStatus::FAILED;
        job.output = {"", InferenceStatus::FAILURE_EMPTY_PROMPT, "Prompt cannot be empty", 0};
        results.add_job(job);
        return id;
    }
    if (tokens <= 0 || tokens > 2048) {
        logger_.log("Job " + id + " failed validation: Invalid token count (" + std::to_string(tokens) + ").", LogLevel::LOG_ERROR);
        job.status = JobStatus::FAILED;
        job.output = {"", InferenceStatus::FAILURE_TOKEN_LIMIT, "Tokens must be between 1 and 2048", 0};
        results.add_job(job);
        return id; 
    }
    if (temp < 0.0f || temp > 2.0f) {
        logger_.log("Job " + id + " failed validation: Invalid temperature (" + std::to_string(temp) + ").", LogLevel::LOG_ERROR);
        job.status = JobStatus::FAILED;
        job.output = {"", InferenceStatus::FAILURE_RUNTIME_ERROR, "Temperature must be between 0.0 and 2.0", 0};
        results.add_job(job);
        return id;
    }

    logger_.log("Job " + id + " received and validated. Queuing for execution.", LogLevel::INFO);
    
    //adding to result store
    results.add_job(job);
    //adding to incomingreq threadsafequeue
    incoming_req.push(job);
    return id;
}

JobStatus InferenceAPI::check_status(const std::string& jobid){
    return results.get_status(jobid);
}

InferenceResult InferenceAPI::fetch_result(const std::string& jobid){
    auto result = results.fetch_result(jobid);
    if (result.has_value()){
        return result.value();
    }
    return {" ", InferenceStatus::FAILURE_RUNTIME_ERROR, "Result could not be generated", 0};
}