#include "WorkerPool.h"
#include "ModelWrapper.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <psapi.h>

//hardcoded vals to see stuff
std::vector<InferenceJob> make_mt_jobs(int count) {
    std::vector<InferenceJob> jobs;
    for (int i = 0; i < count; i++) {
        jobs.push_back({ std::to_string(i), "Stress test prompt " + std::to_string(i), 50 });
    }
    return jobs;
}

struct BenchResult {
    int    num_threads;
    double total_time_sec;
    double throughput;
    double ram_usage_mb; //hard coded must add checks inside the model wrapper itself
};

double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}

BenchResult run_mt_benchmark(int num_threads, ModelWrapper& llm, Logger& logger) {
    ThreadSafeQueue<InferenceJob> queue(100);
    ResultsStorage results;
    auto jobs = make_mt_jobs(15);
    
    // Push jobs before starting timer to measure pure execution
    for (const auto& job : jobs) queue.push(job);
    double ram_usage;
    auto t_start = std::chrono::high_resolution_clock::now();
    {
        // This triggers your pin_to_core and thread_local context logic
        WorkerPool pool(num_threads, queue, results, llm, logger);
        queue.shutdown(); 
        // Destructor joins threads here
        ram_usage = get_ram_mb();
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    BenchResult br;
    br.num_threads = num_threads;
    br.total_time_sec = total_sec;
    br.throughput = 15.0 / total_sec;
    
    // Your calculation from earlier: Shared weights + private contexts
    //br.ram_usage_mb = 636.0 + (num_threads * 100.0); 
    br.ram_usage_mb = ram_usage;
    return br;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: m2_mt_benchmark <model_path>\n";
        return 1;
    }

    std::string model_path = argv[1];
    Logger logger;
    
    std::cout << "=== MILESTONE 2: MULTI-THREADING BENCHMARK (WINDOWS) ===\n";
    
    // Load model ONCE - this is your shared advantage
    ModelWrapper llm(model_path, 512, 4); 

    std::vector<BenchResult> all_results;
    run_mt_benchmark(1, llm, logger);
    for (int t : {1, 2, 4}) {
        all_results.push_back(run_mt_benchmark(t, llm, logger));
    }

    // --- PRINTING THE COMPARISON TABLE ---
    std::cout << "\n=== RESULTS TABLE ===\n";
    std::cout << std::left
              << std::setw(18) << "Strategy"
              << std::setw(10) << "Threads"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Throughput"
              << "RAM Usage\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto& r : all_results) {
        std::string label = std::to_string(r.num_threads) + " Threads";
        std::cout << std::left
                  << std::setw(18) << label
                  << std::setw(10) << r.num_threads
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(14) << std::setprecision(3) << r.throughput
                  << std::fixed << std::setprecision(0) << r.ram_usage_mb << " MB\n";
    }

    return 0;
}