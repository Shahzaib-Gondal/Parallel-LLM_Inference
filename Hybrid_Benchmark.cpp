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
#include <numeric>
#include <windows.h>
#include <psapi.h>

// ============================================================
// CONFIG — tweak these to control the benchmark scope
// ============================================================
static const int    NUM_JOBS        = 50;   // total jobs per run
static const int    NUM_REPEATS     = 1;    // repeats per config (results are averaged)
static const int    TOKENS_PER_JOB  = 50;
static const float  TEMPERATURE     = 0.7f;
static const float  TOP_P           = 0.9f;

static const std::vector<int> THREAD_COUNTS = {2, 4, 8};
static const std::vector<int> BATCH_SIZES   = {4, 8, 16};
// ============================================================

struct RunResult {
    int    threads;
    int    batch_size;
    double total_time_sec;
    double throughput;         // jobs / sec
    double avg_inference_ms;
    double avg_e2e_ms;
    double avg_queue_wait_ms;
    double ram_mb;
    bool   is_best_throughput  = false;
    bool   is_best_latency     = false;
    bool   is_best_combined    = false;
};

// -------------------------------------------------------
double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}

std::vector<InferenceJob> make_jobs(int count) {
    std::vector<InferenceJob> jobs;
    jobs.reserve(count);
    for (int i = 0; i < count; ++i) {
        InferenceJob j;
        j.jobid       = std::to_string(i);
        j.prompt      = "Benchmark prompt number " + std::to_string(i);
        j.tokens      = TOKENS_PER_JOB;
        j.temperature = TEMPERATURE;
        j.top_p       = TOP_P;
        jobs.push_back(j);
    }
    return jobs;
}

// -------------------------------------------------------
// Single run for one (threads, batch_size) configuration.
// Returns averaged metrics over NUM_REPEATS runs.
// -------------------------------------------------------
RunResult run_config(int num_threads, int batch_size,
                     ModelWrapper& llm, Logger& logger)
{
    // Accumulators for averaging across repeats
    double acc_time = 0, acc_infer = 0, acc_e2e = 0, acc_wait = 0, acc_ram = 0;

    for (int rep = 0; rep < NUM_REPEATS; ++rep) {

        // Fresh queue and result store each repeat
        ThreadSafeQueue<std::vector<InferenceJob>> queue(200);
        ResultsStorage results;

        auto all_jobs = make_jobs(NUM_JOBS);

        // Slice jobs into batches and enqueue
        for (int i = 0; i < NUM_JOBS; i += batch_size) {
            std::vector<InferenceJob> batch;
            int end = std::min((i + batch_size), NUM_JOBS);
            for (int j = i; j < end; ++j) {
                all_jobs[j].enqueue_time = std::chrono::high_resolution_clock::now();
                batch.push_back(all_jobs[j]);
            }
            queue.push(batch);
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        {
            WorkerPool pool(num_threads, queue, results, llm, logger);
            queue.shutdown();
            pool.join();

            auto stats = pool.get_latency_stats();
            acc_infer += stats.avg_inference_ms;
            acc_e2e   += stats.avg_e2e_ms;
            acc_wait  += stats.avg_queue_wait_ms;
            acc_ram   += get_ram_mb();
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        acc_time += std::chrono::duration<double>(t_end - t_start).count();
    }

    double avg_time = acc_time / NUM_REPEATS;

    return {
        num_threads,
        batch_size,
        avg_time,
        NUM_JOBS / avg_time,
        acc_infer / NUM_REPEATS,
        acc_e2e   / NUM_REPEATS,
        acc_wait  / NUM_REPEATS,
        acc_ram   / NUM_REPEATS
    };
}

// -------------------------------------------------------
// Print a formatted summary table to stdout
// -------------------------------------------------------
void print_table(const std::vector<RunResult>& results) {
    const int W = 10;
    std::cout << "\n" << std::string(88, '=') << "\n";
    std::cout << "  HYBRID BENCHMARK RESULTS  (" << NUM_JOBS << " jobs x "
              << NUM_REPEATS << " repeats averaged)\n";
    std::cout << std::string(88, '=') << "\n";

    std::cout << std::left
              << std::setw(W)   << "Threads"
              << std::setw(W)   << "Batch"
              << std::setw(W+2) << "Time(s)"
              << std::setw(W+4) << "Jobs/sec"
              << std::setw(W+6) << "Infer(ms)"
              << std::setw(W+4) << "E2E(ms)"
              << std::setw(W+6) << "QWait(ms)"
              << std::setw(W+2) << "RAM(MB)"
              << "Flags\n";
    std::cout << std::string(88, '-') << "\n";

    for (const auto& r : results) {
        std::string flags;
        if (r.is_best_throughput) flags += "[BEST THRUPUT] ";
        if (r.is_best_latency)    flags += "[BEST LATENCY] ";
        if (r.is_best_combined)   flags += "[BEST COMBINED]";

        std::cout << std::left  << std::fixed
                  << std::setw(W)   << r.threads
                  << std::setw(W)   << r.batch_size
                  << std::setw(W+2) << std::setprecision(2) << r.total_time_sec
                  << std::setw(W+4) << std::setprecision(3) << r.throughput
                  << std::setw(W+6) << std::setprecision(1) << r.avg_inference_ms
                  << std::setw(W+4) << std::setprecision(1) << r.avg_e2e_ms
                  << std::setw(W+6) << std::setprecision(1) << r.avg_queue_wait_ms
                  << std::setw(W+2) << std::setprecision(0) << r.ram_mb
                  << flags << "\n";
    }
    std::cout << std::string(88, '=') << "\n";
}

// -------------------------------------------------------
// Write CSV
// -------------------------------------------------------
void write_csv(const std::vector<RunResult>& results, const std::string& path) {
    std::ofstream f(path);
    f << "threads,batch_size,total_time_sec,throughput,avg_inference_ms,"
      << "avg_e2e_ms,avg_queue_wait_ms,ram_mb,best_throughput,best_latency,best_combined\n";
    for (const auto& r : results) {
        f << r.threads          << ","
          << r.batch_size       << ","
          << r.total_time_sec   << ","
          << r.throughput       << ","
          << r.avg_inference_ms << ","
          << r.avg_e2e_ms       << ","
          << r.avg_queue_wait_ms<< ","
          << r.ram_mb           << ","
          << r.is_best_throughput << ","
          << r.is_best_latency    << ","
          << r.is_best_combined   << "\n";
    }
    std::cout << "[CSV written to " << path << "]\n";
}

// -------------------------------------------------------
// Flag the best configs in the result set
// -------------------------------------------------------
void annotate_best(std::vector<RunResult>& results) {
    auto best_thruput = std::max_element(results.begin(), results.end(),
        [](const RunResult& a, const RunResult& b){ return a.throughput < b.throughput; });

    auto best_latency = std::min_element(results.begin(), results.end(),
        [](const RunResult& a, const RunResult& b){ return a.avg_e2e_ms < b.avg_e2e_ms; });

    // Combined score: normalised throughput (higher=better) minus normalised e2e (lower=better)
    double max_tp  = best_thruput->throughput;
    double min_e2e = best_latency->avg_e2e_ms;
    double max_e2e = std::max_element(results.begin(), results.end(),
        [](const RunResult& a, const RunResult& b){ return a.avg_e2e_ms < b.avg_e2e_ms; })->avg_e2e_ms;

    double best_score = -1e9;
    int    best_idx   = 0;
    for (int i = 0; i < (int)results.size(); ++i) {
        double norm_tp  = results[i].throughput / max_tp;
        double norm_e2e = (max_e2e - results[i].avg_e2e_ms) / (max_e2e - min_e2e + 1e-9);
        double score    = 0.6 * norm_tp + 0.4 * norm_e2e;  // weight throughput slightly more
        if (score > best_score) { best_score = score; best_idx = i; }
    }

    best_thruput->is_best_throughput = true;
    best_latency->is_best_latency    = true;
    results[best_idx].is_best_combined = true;
}

// -------------------------------------------------------
// Print winner summary
// -------------------------------------------------------
void print_winners(const std::vector<RunResult>& results) {
    std::cout << "\n--- RECOMMENDATIONS ---\n";
    for (const auto& r : results) {
        if (r.is_best_throughput)
            std::cout << "  Highest Throughput : threads=" << r.threads
                      << "  batch=" << r.batch_size
                      << "  (" << std::fixed << std::setprecision(3)
                      << r.throughput << " jobs/sec)\n";
        if (r.is_best_latency)
            std::cout << "  Lowest E2E Latency : threads=" << r.threads
                      << "  batch=" << r.batch_size
                      << "  (" << std::fixed << std::setprecision(1)
                      << r.avg_e2e_ms << " ms avg)\n";
        if (r.is_best_combined)
            std::cout << "  Best Combined Score: threads=" << r.threads
                      << "  batch=" << r.batch_size
                      << "  (60% throughput + 40% latency)\n";
    }
    std::cout << "\n";
}

// -------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: hybrid_benchmark <model_path> [output.csv]\n";
        return 1;
    }
    std::string model_path = argv[1];
    std::string csv_path   = (argc >= 3) ? argv[2] : "hybrid_results.csv";

    Logger logger;
    ModelWrapper llm(model_path, 512, 4);

    int total_configs = (int)(THREAD_COUNTS.size() * BATCH_SIZES.size());
    std::cout << "=== HYBRID BENCHMARK: THREADING + BATCHING ===\n";
    std::cout << "Configurations : " << total_configs
              << " (" << THREAD_COUNTS.size() << " thread counts x "
              << BATCH_SIZES.size()   << " batch sizes)\n";
    std::cout << "Jobs per run   : " << NUM_JOBS << "\n";
    std::cout << "Repeats        : " << NUM_REPEATS << " (results averaged)\n\n";

    // Warm-up — prevents cold-start skewing the first real config
    std::cout << "[Warm-up] threads=1  batch=1 ...\n";
    {
        Logger dummy_logger;
        run_config(1, 1, llm, dummy_logger);
    }
    std::cout << "[Warm-up done]\n\n";

    std::vector<RunResult> all_results;
    all_results.reserve(total_configs);

    int done = 0;
    for (int t : THREAD_COUNTS) {
        for (int b : BATCH_SIZES) {
            ++done;
            std::cout << "[" << std::setw(2) << done << "/" << total_configs << "]"
                      << "  threads=" << t << "  batch=" << b << " ...";
            std::cout.flush();

            auto r = run_config(t, b, llm, logger);
            all_results.push_back(r);

            std::cout << "  " << std::fixed << std::setprecision(3)
                      << r.throughput << " jobs/sec\n";
        }
    }

    annotate_best(all_results);
    print_table(all_results);
    print_winners(all_results);
    write_csv(all_results, csv_path);

    return 0;
}