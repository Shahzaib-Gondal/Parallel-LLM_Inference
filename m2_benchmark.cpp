#include "ProcessPool.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

std::vector<ProcessJob> make_jobs(int count) {
    std::vector<ProcessJob> jobs;
    for (int i = 0; i < count; i++)
        jobs.push_back({ i, "Stress test prompt " + std::to_string(i), 50 });
    return jobs;
}

struct BenchResult {
    int    num_workers;
    int    total_jobs;
    int    succeeded;
    double total_time_sec;
    double throughput;
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
};

BenchResult run_benchmark(int num_workers,
                          const std::string& worker_exe,
                          const std::string& model_path,
                          int num_jobs)
{
    std::cout << "\n[Benchmark] Spawning " << num_workers
              << " worker process(es)..." << std::endl;

    ProcessPool pool(num_workers, worker_exe, model_path);
    auto jobs    = make_jobs(num_jobs);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto results = pool.run_batch(jobs);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult br;
    br.num_workers    = num_workers;
    br.total_jobs     = num_jobs;
    br.succeeded      = 0;
    br.total_time_sec = total_sec;
    br.min_latency_ms = 1e9;
    br.max_latency_ms = 0;
    double sum_lat    = 0;

    for (auto& r : results) {
        if (r.success) br.succeeded++;
        sum_lat += r.latency_ms;
        if (r.latency_ms < br.min_latency_ms) br.min_latency_ms = r.latency_ms;
        if (r.latency_ms > br.max_latency_ms) br.max_latency_ms = r.latency_ms;
    }

    br.throughput     = br.succeeded / total_sec;
    br.avg_latency_ms = sum_lat / results.size();

    std::cout << "  Workers: "    << num_workers
              << "  Done: "       << br.succeeded << "/" << num_jobs
              << "  Time: "       << std::fixed << std::setprecision(2) << total_sec << "s"
              << "  Throughput: " << std::setprecision(3) << br.throughput << " req/s"
              << "  AvgLat: "     << std::setprecision(1) << br.avg_latency_ms << "ms"
              << std::endl;

    return br;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: m2_benchmark <worker_exe> <model_path>\n";
        return 1;
    }

    std::string worker_exe = argv[1];
    std::string model_path = argv[2];
    const int   NUM_JOBS   = 15;

    std::cout << "=== MILESTONE 2: MULTI-PROCESS BENCHMARK ===" << std::endl;

    std::vector<BenchResult> all;

    BenchResult baseline;
    baseline.num_workers    = 0;
    baseline.total_jobs     = 15;
    baseline.succeeded      = 15;
    baseline.total_time_sec = 28.98;
    baseline.throughput     = 0.517;
    baseline.avg_latency_ms = 1932.0;
    baseline.min_latency_ms = 0;
    baseline.max_latency_ms = 0;
    all.push_back(baseline);

    for (int w : {1, 2, 4}) {
        try {
            all.push_back(run_benchmark(w, worker_exe, model_path, NUM_JOBS));
        } catch (const std::exception& e) {
            std::cerr << "FAILED with " << w << " workers: " << e.what() << "\n";
        }
    }

    std::cout << "\n=== RESULTS TABLE ===\n";
    std::cout << std::left
              << std::setw(18) << "Strategy"
              << std::setw(10) << "Workers"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Throughput"
              << std::setw(16) << "AvgLat(ms)"
              << "Success\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto& r : all) {
        std::string s = r.num_workers == 0 ? "M1 Sequential"
                      : r.num_workers == 1 ? "1 Process"
                      : std::to_string(r.num_workers) + " Processes";
        std::cout << std::left
                  << std::setw(18) << s
                  << std::setw(10) << (r.num_workers == 0 ? 1 : r.num_workers)
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(14) << std::setprecision(3) << r.throughput
                  << std::setw(16) << std::setprecision(1) << r.avg_latency_ms
                  << r.succeeded << "/" << r.total_jobs << "\n";
    }

    std::cout << "\nRAM: ~636MB per worker process\n";
    std::cout << "=== DONE ===\n";
    return 0;
}