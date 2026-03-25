#pragma once
#include<string>
#include "InferenceJob.h"
#include "InferenceResult.h"
#include<unordered_map>
#include<mutex>
#include<optional>

class ResultsStorage{
    std::unordered_map<std::string, InferenceJob> res_store;
    std::mutex mtx;

    public:
    void add_job(const InferenceJob& job){
        std::lock_guard<std::mutex> lock(mtx);
        res_store[job.jobid] = job;
    }

    JobStatus get_status(const std::string& jobid){
        std::lock_guard<std::mutex> lock(mtx);
        if(res_store.count(jobid)){
            return res_store[jobid].status;
        }
        return JobStatus::CANCELLED; //user prolly cancelled req
    }

    //either null or result
    std::optional<InferenceResult> fetch_result(const std::string& jobid){
        std::lock_guard<std::mutex> lock(mtx);
        auto iter = res_store.find(jobid);
        if(iter != res_store.end() && iter->second.status != JobStatus::RUNNING){
            InferenceResult result = iter->second.output;
            res_store.erase(iter);
            return result;
        }
        return std::nullopt;
    }

    //thread will call this to update result of a running job and update status
    void update_res(const std::string& jobid, const InferenceResult& result){
        std::lock_guard<std::mutex> lock(mtx);
        if(res_store.count(jobid)){
            res_store[jobid].output = result;

            //checking result status and updating job status
            if(result.status == InferenceStatus::SUCCESS){
                res_store[jobid].status = JobStatus::COMPLETED;
            }
            else{
                res_store[jobid].status = JobStatus::FAILED;
            }
        }
    }

};