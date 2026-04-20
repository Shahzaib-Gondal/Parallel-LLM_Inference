#include "ThreadSafeQueue.h"
#include "InferenceJob.h"
#include "ResultStorage.h"
#include "ModelWrapper.h"
#include "WorkerPool.h"
#include "JobDispatcher.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <psapi.h>

using namespace std;

// Generate dummy jobs for the benchmark
std::vector<InferenceJob> make_batch_jobs(int count, int batch_size) {
    std::vector<InferenceJob> jobs;
    for (int i = 0; i < count; i++) {
        // Unique IDs for every run
        std::string id = "b" + to_string(batch_size) + "_job_" + to_string(i);
        jobs.push_back({ id, "Tell me a fact about number " + std::to_string(i), 30, JobStatus::PENDING, 0.7f, 0.9f, {"", InferenceStatus::SUCCESS, "", 0} });
    }
    return jobs;
}

struct BenchResult {
    int    batch_size;
    double total_time_sec;
    double throughput;
    double avg_latency_ms;
    double ram_usage_mb; 
};

// Windows RAM Tracker
double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}

BenchResult run_batch_benchmark(int batch_size, ModelWrapper& llm) {
    // 1. Setup clean queues and database for this specific run
    ThreadSafeQueue<InferenceJob> api_queue;
    ThreadSafeQueue<vector<InferenceJob>> worker_queue;
    ResultsStorage results_db;

    int total_jobs = 16; // Use a multiple of your max batch size (e.g., 16)
    auto jobs = make_batch_jobs(total_jobs, batch_size);
    
    // 2. Pre-load jobs into the API queue to measure pure execution speed
    for (auto& job : jobs) {
        api_queue.push(std::move(job));
    }

    double ram_usage;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    {
        // 3. Spin up the Dispatcher with the specific batch size we are testing
        JobDispatcher dispatcher(api_queue, worker_queue, batch_size);
        
        // 4. Spin up ONE worker thread (to isolate batching metrics from MT metrics)
        WorkerPool pool(1, worker_queue, results_db, llm); 
        
        // 5. Polling Loop: Wait for all jobs to hit the database
        int completed_jobs = 0;
        while (completed_jobs < total_jobs) {
            completed_jobs = 0;
            for (const auto& job : jobs) {
                JobStatus status = results_db.get_status(job.jobid);
                if (status == JobStatus::COMPLETED || status == JobStatus::FAILED) {
                    completed_jobs++;
                }
            }
            this_thread::sleep_for(chrono::milliseconds(50));
        }

        // Measure RAM right before destroying the pool/dispatcher
        ram_usage = get_ram_mb();
        
        // The queues and dispatcher will be safely destroyed here due to scope
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    BenchResult br;
    br.batch_size = batch_size;
    br.total_time_sec = total_sec;
    br.throughput = total_jobs / total_sec;
    br.avg_latency_ms = (total_sec / total_jobs) * 1000.0;
    br.ram_usage_mb = ram_usage;
    
    return br;
}

int main(int argc, char* argv[]) {
    std::cout << "=== MILESTONE 2: BATCH INFERENCE BENCHMARK (WINDOWS) ===\n";
    
    // Load model ONCE to keep things fair
    cout << "Booting up TinyLlama backend...\n";
    ModelWrapper llm("models/tinyllama.gguf"); 

    std::vector<BenchResult> all_results;
    
    // Run an initial "warm-up" to load CUDA/CPU kernels into memory (optional but good practice)
    cout << "Running warm-up sequence...\n";
    run_batch_benchmark(1, llm);

    // Run the actual benchmark tests
    cout << "Starting automated benchmark suite...\n";
    for (int bs : {1, 2, 4, 8}) {
        cout << "Testing Batch Size: " << bs << "...\n";
        all_results.push_back(run_batch_benchmark(bs, llm));
    }

    // --- PRINTING THE COMPARISON TABLE ---
    std::cout << "\n=========================================================================\n";
    std::cout << std::left
              << std::setw(15) << "Batch Size"
              << std::setw(12) << "Time(s)"
              << std::setw(15) << "Throughput"
              << std::setw(15) << "Avg Latency"
              << "RAM Usage\n";
    std::cout << std::string(73, '-') << "\n";

    for (auto& r : all_results) {
        std::string label = "Size " + std::to_string(r.batch_size);
        std::cout << std::left
                  << std::setw(15) << label
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_sec
                  << std::setw(15) << std::setprecision(3) << std::to_string(r.throughput) + " req/s"
                  << std::setw(15) << std::setprecision(1) << std::to_string(r.avg_latency_ms) + " ms"
                  << std::fixed << std::setprecision(0) << r.ram_usage_mb << " MB\n";
    }
    std::cout << "=========================================================================\n";

    return 0;
}