#include "ProcessPool.h"
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <io.h>
#include <fcntl.h>

ProcessPool::ProcessPool(int num_workers,
                         const std::string& worker_exe_path,
                         const std::string& model_path)
    : worker_exe_(worker_exe_path), model_path_(model_path)
{
    for (int i = 0; i < num_workers; i++) {
        WorkerProcess* w = new WorkerProcess();
        w->worker_id = i;
        if (!spawn_worker(w)) {
            delete w;
            throw std::runtime_error("Failed to spawn worker " + std::to_string(i));
        }
        workers_.push_back(w);
        std::cerr << "[ProcessPool] Worker " << i << " ready.\n";
    }
}

ProcessPool::~ProcessPool() {
    for (auto* w : workers_) {
        if (w->in_stream) {
            fputs("SHUTDOWN\n", w->in_stream);
            fflush(w->in_stream);
        }
        kill_worker(w);
        delete w;
    }
    std::cerr << "[ProcessPool] All workers shut down.\n";
}

bool ProcessPool::spawn_worker(WorkerProcess* w) {
    HANDLE hStdinRead, hStdinWrite;
    HANDLE hStdoutRead, hStdoutWrite;

    SECURITY_ATTRIBUTES sa;
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    if (!CreatePipe(&hStdinRead,  &hStdinWrite,  &sa, 0)) return false;
    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
        CloseHandle(hStdinRead); CloseHandle(hStdinWrite);
        return false;
    }

    SetHandleInformation(hStdinWrite,  HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStdoutRead,  HANDLE_FLAG_INHERIT, 0);

    std::string cmd = "\"" + worker_exe_ + "\" \"" + model_path_ + "\"";
    std::vector<char> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back('\0');

    STARTUPINFOA si = {};
    si.cb         = sizeof(STARTUPINFOA);
    si.hStdInput  = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags    = STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessA(
        nullptr, cmdBuf.data(),
        nullptr, nullptr,
        TRUE, 0, nullptr, nullptr,
        &si, &pi
    );

    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);

    if (!ok) return false;

    CloseHandle(pi.hThread);
    w->hProcess    = pi.hProcess;
    w->hStdinWrite = hStdinWrite;
    w->hStdoutRead = hStdoutRead;

    int stdin_fd  = _open_osfhandle((intptr_t)hStdinWrite, 0);
    int stdout_fd = _open_osfhandle((intptr_t)hStdoutRead, _O_RDONLY);
    w->in_stream  = _fdopen(stdin_fd,  "w");
    w->out_stream = _fdopen(stdout_fd, "r");

    if (!w->in_stream || !w->out_stream) {
        kill_worker(w);
        return false;
    }

    char buf[64] = {};
    if (!fgets(buf, sizeof(buf), w->out_stream)) {
        kill_worker(w);
        return false;
    }

    std::string resp(buf);
    while (!resp.empty() && (resp.back() == '\n' || resp.back() == '\r'))
        resp.pop_back();

    if (resp != "READY") {
        std::cerr << "[ProcessPool] Worker " << w->worker_id
                  << " bad response: " << resp << "\n";
        kill_worker(w);
        return false;
    }

    w->ready = true;
    return true;
}

void ProcessPool::kill_worker(WorkerProcess* w) {
    if (w->in_stream)  { fclose(w->in_stream);  w->in_stream  = nullptr; }
    if (w->out_stream) { fclose(w->out_stream); w->out_stream = nullptr; }
    if (w->hProcess != INVALID_HANDLE_VALUE) {
        TerminateProcess(w->hProcess, 0);
        CloseHandle(w->hProcess);
        w->hProcess = INVALID_HANDLE_VALUE;
    }
    w->ready = false;
}

ProcessResult ProcessPool::send_job(WorkerProcess* w, const ProcessJob& job) {
    ProcessResult res;
    res.job_id     = job.job_id;
    res.success    = false;
    res.latency_ms = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(w->pipe_mutex);

    if (fputs((job.prompt + "\n").c_str(), w->in_stream) == EOF
        || fflush(w->in_stream) != 0) {
        res.output = "Pipe write failed";
        return res;
    }

    char buf[4096] = {};
    if (!fgets(buf, sizeof(buf), w->out_stream)) {
        res.output = "Pipe read failed";
        return res;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    res.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::string reply(buf);
    while (!reply.empty() && (reply.back() == '\n' || reply.back() == '\r'))
        reply.pop_back();

    if (reply.substr(0, 3) == "OK:") {
        res.output  = reply.substr(3);
        res.success = true;
    } else if (reply.size() >= 6 && reply.substr(0, 6) == "ERROR:") {
        res.output  = reply.substr(6);
    } else {
        res.output  = "Unknown: " + reply;
    }

    return res;
}

std::vector<ProcessResult> ProcessPool::run_batch(const std::vector<ProcessJob>& jobs) {
    std::vector<ProcessResult> results(jobs.size());

    std::queue<int> job_queue;
    std::mutex      queue_mutex;

    for (int i = 0; i < (int)jobs.size(); i++) job_queue.push(i);

    std::vector<std::thread> threads;
    for (int wi = 0; wi < (int)workers_.size(); wi++) {
        threads.emplace_back([&, wi]() {
            while (true) {
                int idx = -1;
                {
                    std::lock_guard<std::mutex> lk(queue_mutex);
                    if (job_queue.empty()) break;
                    idx = job_queue.front();
                    job_queue.pop();
                }
                results[idx] = send_job(workers_[wi], jobs[idx]);
            }
        });
    }

    for (auto& t : threads) t.join();
    return results;
}