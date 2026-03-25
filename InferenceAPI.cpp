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