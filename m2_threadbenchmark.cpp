#include "WorkerPool.h"
#include "ModelWrapper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <psapi.h>

//produces 
std::vector<InferenceJob> make_mt_jobs(int count) {
    std::vector<InferenceJob> jobs;
    jobs.reserve(count);
    for (int i = 0; i < count; i++) {
        InferenceJob job;
        job.jobid        = std::to_string(i);
        job.prompt       = "Stress test prompt " + std::to_string(i);
        job.tokens       = 50;
        job.enqueue_time = std::chrono::high_resolution_clock::now();
        jobs.push_back(job);
    }
    return jobs;
}

struct BenchResult {
    int    num_threads;
    double total_time_sec;
    double throughput;           // jobs / second
    double avg_inference_ms;     // pure model time per job
    double avg_e2e_ms;           // queue-wait + inference per job
    double avg_queue_wait_ms;    // contention overhead per job
    double ram_usage_mb;
};
double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}

BenchResult run_mt_benchmark(int num_threads, ModelWrapper& llm, Logger& logger) {
    constexpr int NUM_JOBS = 50;

    ThreadSafeQueue<std::vector<InferenceJob>> queue(200);
    ResultsStorage results;
auto jobs = make_mt_jobs(NUM_JOBS);
for (auto& job : jobs) {
    job.enqueue_time = std::chrono::high_resolution_clock::now();
}

for (auto& job : jobs) {
    job.enqueue_time = std::chrono::high_resolution_clock::now();
    queue.push(std::vector<InferenceJob>{job});
}

    double ram_usage = 0.0;
    WorkerPool::LatencyStats lat_stats;

    auto t_start = std::chrono::high_resolution_clock::now();
    {
       WorkerPool pool(num_threads, queue, results, llm, logger);
        queue.shutdown();
        pool.join();                           // all threads done
        lat_stats  = pool.get_latency_stats(); // read AFTER join
        ram_usage  = get_ram_mb();
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult br;
    br.num_threads       = num_threads;
    br.total_time_sec    = total_sec;
    br.throughput        = NUM_JOBS / total_sec;
    br.avg_inference_ms  = lat_stats.avg_inference_ms;
    br.avg_e2e_ms        = lat_stats.avg_e2e_ms;
    br.avg_queue_wait_ms = lat_stats.avg_queue_wait_ms;
    br.ram_usage_mb      = ram_usage;
    return br;
}

void write_csv(const std::vector<BenchResult>& results, const std::string& path) {
    std::ofstream f(path);
    f << "threads,total_time_sec,throughput,avg_inference_ms,avg_e2e_ms,avg_queue_wait_ms,ram_mb\n";
    for (const auto& r : results) {
        f << r.num_threads        << ","
          << r.total_time_sec     << ","
          << r.throughput         << ","
          << r.avg_inference_ms   << ","
          << r.avg_e2e_ms         << ","
          << r.avg_queue_wait_ms  << ","
          << r.ram_usage_mb       << "\n";
    }
    std::cout << "\n[CSV written to " << path << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: m2_mt_benchmark <model_path> [output.csv]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string csv_path   = (argc >= 3) ? argv[2] : "benchmark_results.csv";

    Logger logger;
    std::cout << "=== MILESTONE 2: MULTI-THREADING BENCHMARK (WINDOWS) ===\n";

    // Shared model weights — one load, N thread contexts
    ModelWrapper llm(model_path, 512, 4);

    std::vector<BenchResult> all_results;

    //cold cache avoidance run
    std::cout << "[Warm-up] Running 1-thread warm-up pass...\n";
    run_mt_benchmark(1, llm, logger);

    for (int t : {1, 2, 4, 8, 12, 16}) {
        std::cout << "[Benchmark] " << t << " thread(s)...\n";
        all_results.push_back(run_mt_benchmark(t, llm, logger));
    }

    // --- Results table ---
    std::cout << "\n=== RESULTS TABLE ===\n";
    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Throughput"
              << std::setw(18) << "Avg Infer(ms)"
              << std::setw(18) << "Avg E2E(ms)"
              << std::setw(18) << "Queue Wait(ms)"
              << "RAM(MB)\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : all_results) {
        std::cout << std::left
                  << std::setw(10) << r.num_threads
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(14) << std::setprecision(3) << r.throughput
                  << std::setw(18) << std::setprecision(1) << r.avg_inference_ms
                  << std::setw(18) << r.avg_e2e_ms
                  << std::setw(18) << r.avg_queue_wait_ms
                  << std::setprecision(0) << r.ram_usage_mb << " MB\n";
    }

    write_csv(all_results, csv_path);

    return 0;
}