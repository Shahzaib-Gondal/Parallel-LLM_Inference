#include "ProcessPool.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>

std::vector<ProcessJob> make_jobs(int count) {
    std::vector<ProcessJob> jobs;
    for (int i = 0; i < count; i++)
        jobs.push_back({ i, "Stress test prompt " + std::to_string(i), 50 });
    return jobs;
}

struct BenchResult {
    std::string strategy;
    int         num_workers;
    int         total_jobs;
    int         succeeded;
    double      total_time_sec;
    double      throughput;
    double      avg_latency_ms;
    double      min_latency_ms;
    double      max_latency_ms;
    double      avg_queue_wait_ms; // not available for process pool, set to 0
};

BenchResult run_benchmark(int num_workers,
                          const std::string& worker_exe,
                          const std::string& model_path,
                          int num_jobs)
{
    std::cout << "\n[Benchmark] Spawning " << num_workers
              << " worker process(es) for " << num_jobs << " jobs..." << std::endl;

    ProcessPool pool(num_workers, worker_exe, model_path);
    auto jobs    = make_jobs(num_jobs);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto results = pool.run_batch(jobs);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult br;
    br.num_workers       = num_workers;
    br.total_jobs        = num_jobs;
    br.succeeded         = 0;
    br.total_time_sec    = total_sec;
    br.min_latency_ms    = 1e9;
    br.max_latency_ms    = 0;
    br.avg_queue_wait_ms = 0; // process pool doesn't track queue wait separately
    double sum_lat       = 0;

    for (auto& r : results) {
        if (r.success) br.succeeded++;
        sum_lat += r.latency_ms;
        if (r.latency_ms < br.min_latency_ms) br.min_latency_ms = r.latency_ms;
        if (r.latency_ms > br.max_latency_ms) br.max_latency_ms = r.latency_ms;
    }

    br.throughput     = br.succeeded / total_sec;
    br.avg_latency_ms = results.empty() ? 0 : sum_lat / results.size();

    if      (num_workers == 0) br.strategy = "Sequential";
    else if (num_workers == 1) br.strategy = "1 Process";
    else                       br.strategy = std::to_string(num_workers) + " Processes";

    std::cout << "  Strategy: "   << br.strategy
              << "  Done: "       << br.succeeded << "/" << num_jobs
              << "  Time: "       << std::fixed << std::setprecision(2) << total_sec << "s"
              << "  Throughput: " << std::setprecision(3) << br.throughput << " req/s"
              << "  AvgLat: "     << std::setprecision(1) << br.avg_latency_ms << "ms"
              << std::endl;

    return br;
}

void write_csv(const std::vector<BenchResult>& results, const std::string& path) {
    std::ofstream f(path);
    f << "strategy,workers,total_time_sec,throughput,avg_latency_ms,"
      << "min_latency_ms,max_latency_ms,success\n";
    for (const auto& r : results) {
        f << r.strategy        << ","
          << r.num_workers     << ","
          << r.total_time_sec  << ","
          << r.throughput      << ","
          << r.avg_latency_ms  << ","
          << r.min_latency_ms  << ","
          << r.max_latency_ms  << ","
          << r.succeeded << "/" << r.total_jobs << "\n";
    }
    std::cout << "\n[CSV written to " << path << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: m2_benchmark <worker_exe> <model_path> [output.csv]\n";
        return 1;
    }

    std::string worker_exe = argv[1];
    std::string model_path = argv[2];
    std::string csv_path   = (argc >= 4) ? argv[3] : "process_results.csv";
    const int   NUM_JOBS   = 30;

    std::cout << "=== MILESTONE 2: MULTI-PROCESS BENCHMARK ===\n";
    std::cout << "Jobs per run: " << NUM_JOBS << "\n";

    std::vector<BenchResult> all;

    // Sequential baseline — 1 worker, jobs run one at a time (no parallelism)
    std::cout << "\n[Sequential baseline] 1 worker, jobs dispatched one at a time...\n";
    {
        ProcessPool seq_pool(1, worker_exe, model_path);
        auto jobs    = make_jobs(NUM_JOBS);
        auto t_start = std::chrono::high_resolution_clock::now();

        // Force sequential: send jobs one by one instead of using run_batch
        std::vector<ProcessResult> seq_results;
        for (auto& job : jobs) {
            auto batch_result = seq_pool.run_batch({job});
            seq_results.insert(seq_results.end(),
                               batch_result.begin(), batch_result.end());
        }

        auto t_end   = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();

        BenchResult br;
        br.strategy          = "Sequential";
        br.num_workers       = 1;
        br.total_jobs        = NUM_JOBS;
        br.succeeded         = 0;
        br.total_time_sec    = total_sec;
        br.min_latency_ms    = 1e9;
        br.max_latency_ms    = 0;
        br.avg_queue_wait_ms = 0;
        double sum_lat       = 0;

        for (auto& r : seq_results) {
            if (r.success) br.succeeded++;
            sum_lat += r.latency_ms;
            if (r.latency_ms < br.min_latency_ms) br.min_latency_ms = r.latency_ms;
            if (r.latency_ms > br.max_latency_ms) br.max_latency_ms = r.latency_ms;
        }
        br.throughput     = br.succeeded / total_sec;
        br.avg_latency_ms = seq_results.empty() ? 0 : sum_lat / seq_results.size();

        std::cout << "  Sequential done: " << br.succeeded << "/" << NUM_JOBS
                  << "  Time: " << std::fixed << std::setprecision(2) << total_sec << "s"
                  << "  Throughput: " << std::setprecision(3) << br.throughput << " req/s"
                  << "  AvgLat: " << std::setprecision(1) << br.avg_latency_ms << "ms\n";
        all.push_back(br);
    }

    // Parallel process runs
    for (int w : {1, 2, 4}) {
        try {
            all.push_back(run_benchmark(w, worker_exe, model_path, NUM_JOBS));
        } catch (const std::exception& e) {
            std::cerr << "FAILED with " << w << " workers: " << e.what() << "\n";
        }
    }

    // Results table — same columns as thread benchmark
    std::cout << "\n=== RESULTS TABLE ===\n";
    std::cout << std::left
              << std::setw(16) << "Strategy"
              << std::setw(10) << "Workers"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Throughput"
              << std::setw(18) << "Avg Infer(ms)"
              << std::setw(16) << "Min Lat(ms)"
              << std::setw(16) << "Max Lat(ms)"
              << "Success\n";
    std::cout << std::string(105, '-') << "\n";

    for (const auto& r : all) {
        std::cout << std::left
                  << std::setw(16) << r.strategy
                  << std::setw(10) << r.num_workers
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(14) << std::setprecision(3) << r.throughput
                  << std::setw(18) << std::setprecision(1) << r.avg_latency_ms
                  << std::setw(16) << r.min_latency_ms
                  << std::setw(16) << r.max_latency_ms
                  << r.succeeded << "/" << r.total_jobs << "\n";
    }

    std::cout << "\nRAM: ~636 MB per worker process\n";
    std::cout << "=== DONE ===\n";

    write_csv(all, csv_path);
    return 0;
}