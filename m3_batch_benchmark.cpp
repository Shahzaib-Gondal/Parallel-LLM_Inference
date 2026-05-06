#include "WorkerPool.h"
#include "ModelWrapper.h"
#include "Logger.h"
#include "ResultStorage.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <windows.h>
#include <psapi.h>

std::vector<InferenceJob> make_batch_jobs(int count) {
    std::vector<InferenceJob> jobs;
    for (int i = 0; i < count; i++) {
        InferenceJob job;
        job.jobid        = std::to_string(i);
        job.prompt       = "Stress test prompt " + std::to_string(i);
        job.tokens       = 50;
        job.temperature  = 0.7f;
        job.top_p        = 0.9f;
        job.enqueue_time = std::chrono::high_resolution_clock::now();
        jobs.push_back(job);
    }
    return jobs;
}

struct BatchBenchResult {
    int    batch_size;
    double total_time_sec;
    double throughput;
    double avg_inference_ms;
    double avg_e2e_ms;
    double avg_queue_wait_ms;
    double ram_mb;
};

double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}

BatchBenchResult run_batch_bench(int batch_size, ModelWrapper& llm, Logger& logger) {
    const int NUM_JOBS = 50;

    ThreadSafeQueue<std::vector<InferenceJob>> queue(200);
    ResultsStorage results;

    auto all_jobs = make_batch_jobs(NUM_JOBS);

    for (int i = 0; i < NUM_JOBS; i += batch_size) {
        std::vector<InferenceJob> batch;
        int end = (std::min)(i + batch_size, NUM_JOBS);
        for (int j = i; j < end; j++) {
            all_jobs[j].enqueue_time = std::chrono::high_resolution_clock::now();
            batch.push_back(all_jobs[j]);
        }
        queue.push(batch);
    }

    double ram_usage = 0.0;
    WorkerPool::LatencyStats stats;

    auto t_start = std::chrono::high_resolution_clock::now();
    {
        WorkerPool pool(4, queue, results, llm, logger);
        queue.shutdown();
        pool.join();                        // wait for all threads to finish
        stats    = pool.get_latency_stats(); // safe to read now
        ram_usage = get_ram_mb();
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    return {
        batch_size,
        total_sec,
        NUM_JOBS / total_sec,
        stats.avg_inference_ms,
        stats.avg_e2e_ms,
        stats.avg_queue_wait_ms,
        ram_usage
    };
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: m3_batch_benchmark <model_path> [output.csv]\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string csv_path   = (argc >= 3) ? argv[2] : "batch_results.csv";

    Logger logger;
    std::cout << "=== MILESTONE 3: BATCH SIZE BENCHMARK ===\n";

    ModelWrapper llm(model_path, 512, 4);

    std::vector<BatchBenchResult> all;

    std::cout << "[Warm-up] batch_size=1...\n";
    run_batch_bench(1, llm, logger);

    for (int bs : {1, 4, 8, 16}) {
        std::cout << "[Benchmark] batch_size=" << bs << "...\n";
        all.push_back(run_batch_bench(bs, llm, logger));
    }

    std::cout << "\n=== BATCH RESULTS ===\n";
    std::cout << std::left
              << std::setw(12) << "BatchSize"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Throughput"
              << std::setw(18) << "AvgInfer(ms)"
              << std::setw(16) << "AvgE2E(ms)"
              << std::setw(18) << "QueueWait(ms)"
              << "RAM(MB)\n";
    std::cout << std::string(95, '-') << "\n";

    std::ofstream csv(csv_path);
    csv << "batch_size,total_time_sec,throughput,avg_inference_ms,"
        << "avg_e2e_ms,avg_queue_wait_ms,ram_mb\n";

    for (auto& r : all) {
        std::cout << std::left
                  << std::setw(12) << r.batch_size
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(14) << std::setprecision(3) << r.throughput
                  << std::setw(18) << std::setprecision(1) << r.avg_inference_ms
                  << std::setw(16) << r.avg_e2e_ms
                  << std::setw(18) << r.avg_queue_wait_ms
                  << std::setprecision(0) << r.ram_mb << " MB\n";

        csv << r.batch_size        << ","
            << r.total_time_sec    << ","
            << r.throughput        << ","
            << r.avg_inference_ms  << ","
            << r.avg_e2e_ms        << ","
            << r.avg_queue_wait_ms << ","
            << r.ram_mb            << "\n";
    }

    std::cout << "\n[CSV written to " << csv_path << "]\n";
    return 0;
}