#include <string>

#include "InferenceJob.h"
#include "InferenceResult.h"
#include "ResultStorage.h"
#include "ThreadSafeQueue.h"
#include "Logger.h"

//async implementation
class InferenceAPI{
    //add threadsafe incomingreq_queue here
    ThreadSafeQueue<InferenceJob>& incoming_req;
    //keeping a result store here that users can call for checking status
    ResultsStorage& results;
    Logger& logger_;

    std::string gen_jobid();

    public:
    InferenceAPI(ThreadSafeQueue<InferenceJob>& q, ResultsStorage& rs, Logger& logger): incoming_req(q), results(rs), logger_(logger) {}

    std::string submit_req(const std::string& prompt, int tokens = 128, float temp=0.7f);
    //result retreival mechanisms
    JobStatus check_status(const std::string& jobid);
    InferenceResult fetch_result(const std::string& jobid);
};
