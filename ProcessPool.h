#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <windows.h>

struct WorkerProcess {
    int    worker_id   = 0;
    HANDLE hProcess    = INVALID_HANDLE_VALUE;
    HANDLE hStdinWrite = INVALID_HANDLE_VALUE;
    HANDLE hStdoutRead = INVALID_HANDLE_VALUE;
    FILE*  in_stream   = nullptr;
    FILE*  out_stream  = nullptr;
    bool   ready       = false;
    std::mutex pipe_mutex;
};

struct ProcessJob {
    int         job_id;
    std::string prompt;
    int         max_tokens = 50;
};

struct ProcessResult {
    int         job_id;
    std::string output;
    bool        success;
    double      latency_ms;
};

class ProcessPool {
public:
    ProcessPool(int num_workers,
                const std::string& worker_exe_path,
                const std::string& model_path);
    ~ProcessPool();

    std::vector<ProcessResult> run_batch(const std::vector<ProcessJob>& jobs);
    int worker_count() const { return (int)workers_.size(); }

private:
    std::vector<WorkerProcess*> workers_;
    std::string worker_exe_;
    std::string model_path_;

    bool spawn_worker(WorkerProcess* w);
    void kill_worker(WorkerProcess* w);
    ProcessResult send_job(WorkerProcess* w, const ProcessJob& job);
};