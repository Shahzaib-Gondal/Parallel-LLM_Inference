// What this does:
//   1. Validates edge-case inputs still fail gracefully (empty prompts, bad tokens, etc.)
//   2. Fires concurrent bursts of requests through the InferenceEngine facade
//   3. Ramps up job counts to find where throughput plateaus / queue wait explodes
//   4. Tracks RAM usage at every stage so we can see memory pressure
//   5. Dumps everything into a nicely formatted CSV for Excel analysis

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

#include <windows.h>
#include <psapi.h>

#include "InferenceEngine.h"


static const int    TOKENS_PER_JOB  = 50;      // keep it short so runs finish in reasonable time
static const float  TEMPERATURE     = 0.7f;
static const int    WARMUP_JOBS     = 5;        // throwaway jobs to get the model warmed up
static const int    POLL_INTERVAL   = 150;      // ms between status checks

// job counts we ramp through during the scaling test
static const std::vector<int> RAMP_LOADS = { 5, 10, 20, 40, 60, 80, 100 };

// how many concurrent client threads for the burst test
static const int BURST_CLIENTS = 30;

// grab current process RAM in megabytes
static double get_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
}


//wait until every job in the list is either COMPLETED or FAILED
// return how many succeeded vs failed.
struct CompletionStats {
    int succeeded  = 0;
    int failed     = 0;
    double wall_ms = 0.0;
};

static CompletionStats wait_for_all(InferenceAPI* api,
                                    const std::vector<std::string>& ids)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int remaining = (int)ids.size();
    std::vector<bool> done(ids.size(), false);

    // keep polling until everything finishes
    while (remaining > 0) {
        for (int i = 0; i < (int)ids.size(); ++i) {
            if (done[i]) continue;
            JobStatus s = api->check_status(ids[i]);
            if (s == JobStatus::COMPLETED || s == JobStatus::FAILED) {
                done[i] = true;
                remaining--;
            }
        }
        if (remaining > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL));
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    CompletionStats cs;
    cs.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int i = 0; i < (int)ids.size(); ++i) {
        // fetch_result will return the stored result; status tells us what happened
        InferenceResult r = api->fetch_result(ids[i]);
        if (r.status == InferenceStatus::SUCCESS)
            cs.succeeded++;
        else
            cs.failed++;
    }
    return cs;
}


static void test_input_validation(InferenceAPI* api) {
    std::cout << "\n[TEST] Input Validation\n";
    std::cout << std::string(50, '-') << "\n";

    //empty prompt should be rejected immediately
    std::string id1 = api->submit_req("", 128, 0.7f);
    assert(api->check_status(id1) == JobStatus::FAILED);
    assert(api->fetch_result(id1).status == InferenceStatus::FAILURE_EMPTY_PROMPT);
    std::cout << "  [PASS] Empty prompt correctly rejected\n";

    //negative token count
    std::string id2 = api->submit_req("Hello world", -10, 0.7f);
    assert(api->check_status(id2) == JobStatus::FAILED);
    assert(api->fetch_result(id2).status == InferenceStatus::FAILURE_TOKEN_LIMIT);
    std::cout << "  [PASS] Negative token count rejected\n";

    //token count over the 2048 ceiling
    std::string id3 = api->submit_req("Hello world", 9999, 0.7f);
    assert(api->check_status(id3) == JobStatus::FAILED);
    assert(api->fetch_result(id3).status == InferenceStatus::FAILURE_TOKEN_LIMIT);
    std::cout << "  [PASS] Excessive token count rejected\n";

    //temperature out of range (above 2.0)
    std::string id4 = api->submit_req("Hello world", 128, 5.0f);
    assert(api->check_status(id4) == JobStatus::FAILED);
    assert(api->fetch_result(id4).status == InferenceStatus::FAILURE_RUNTIME_ERROR);
    std::cout << "  [PASS] Invalid temperature rejected\n";

    //negative temperature
    std::string id5 = api->submit_req("Hello world", 128, -1.0f);
    assert(api->check_status(id5) == JobStatus::FAILED);
    assert(api->fetch_result(id5).status == InferenceStatus::FAILURE_RUNTIME_ERROR);
    std::cout << "  [PASS] Negative temperature rejected\n";

    std::cout << "  >> All input validation tests PASSED\n";
}


//  TEST 2 — Concurrent Burst Submission
//  Many client threads fire requests at the same time.
//  This hammers the pending queue and dispatcher.
struct BurstResult {
    int    total_jobs;
    int    succeeded;
    int    failed;
    double submit_ms;     // time to submit all jobs
    double total_ms;      // end-to-end wall time
    double ram_after_mb;
};

static BurstResult test_concurrent_burst(InferenceAPI* api, int num_clients) {
    std::cout << "\n[TEST] Concurrent Burst (" << num_clients << " clients)\n";
    std::cout << std::string(50, '-') << "\n";

    std::vector<std::string> job_ids(num_clients);
    std::atomic<int> submit_errors{0};

    auto t_submit_start = std::chrono::high_resolution_clock::now();

    // each client is its own thread — they all slam submit_req simultaneously
    std::vector<std::thread> clients;
    clients.reserve(num_clients);

    for (int i = 0; i < num_clients; ++i) {
        clients.emplace_back([&, i]() {
            std::string prompt = "Concurrent burst prompt number " + std::to_string(i);
            try {
                job_ids[i] = api->submit_req(prompt, TOKENS_PER_JOB, TEMPERATURE);
            } catch (...) {
                submit_errors++;
            }
        });
    }

    // wait for all submit threads to finish
    for (auto& t : clients) t.join();

    auto t_submit_end = std::chrono::high_resolution_clock::now();
    double submit_ms = std::chrono::duration<double, std::milli>(
                           t_submit_end - t_submit_start).count();

    std::cout << "  Submitted " << num_clients << " jobs in "
              << std::fixed << std::setprecision(1) << submit_ms << " ms"
              << " (errors: " << submit_errors.load() << ")\n";

    // now wait for every job to finish
    CompletionStats cs = wait_for_all(api, job_ids);

    double ram_now = get_ram_mb();
    std::cout << "  Completed: " << cs.succeeded << " OK, " << cs.failed << " failed"
              << " in " << std::setprecision(1) << cs.wall_ms << " ms\n";
    std::cout << "  RAM after burst: " << std::setprecision(0) << ram_now << " MB\n";

    // basic sanity: at least some jobs should succeed
    assert(cs.succeeded > 0 && "Expected at least some jobs to succeed in burst test");
    std::cout << "  >> Burst test PASSED\n";

    return { num_clients, cs.succeeded, cs.failed, submit_ms,
             submit_ms + cs.wall_ms, ram_now };
}


//  TEST 3 — Ramp-Up Scaling / Bottleneck Detection
//  Gradually increase the number of jobs and measure how
//  throughput and latency change.  This is the meat of the
//  stress test — the CSV data comes from here.
struct ScalePoint {
    int    num_jobs;
    int    succeeded;
    int    failed;
    double wall_sec;       // total wall-clock time for this batch
    double throughput;     // jobs/sec
    double avg_latency_ms; // estimated per-job latency
    double ram_mb;
    double submit_ms;      // just the submission phase
};

static std::vector<ScalePoint> test_ramp_scaling(InferenceAPI* api) {
    std::cout << "\n[TEST] Ramp-Up Scaling\n";
    std::cout << std::string(50, '-') << "\n";

    std::vector<ScalePoint> points;
    points.reserve(RAMP_LOADS.size());

    for (int load : RAMP_LOADS) {
        std::cout << "  Load=" << std::setw(4) << load << " jobs ... ";
        std::cout.flush();

        std::vector<std::string> ids(load);

        // submit all jobs as fast as possible (single thread to keep it simple)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < load; ++i) {
            std::string prompt = "Scaling test job " + std::to_string(i)
                               + " at load level " + std::to_string(load);
            ids[i] = api->submit_req(prompt, TOKENS_PER_JOB, TEMPERATURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sub_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // wait for completion
        CompletionStats cs = wait_for_all(api, ids);

        double total_ms = sub_ms + cs.wall_ms;
        double wall_sec = total_ms / 1000.0;
        double tp       = (wall_sec > 0) ? load / wall_sec : 0;
        double avg_lat  = (load > 0) ? total_ms / load : 0;
        double ram      = get_ram_mb();

        std::cout << std::fixed << std::setprecision(3) << tp << " jobs/sec"
                  << "  |  latency=" << std::setprecision(1) << avg_lat << "ms"
                  << "  |  RAM=" << std::setprecision(0) << ram << "MB"
                  << "  |  OK=" << cs.succeeded << " FAIL=" << cs.failed
                  << "\n";

        points.push_back({ load, cs.succeeded, cs.failed,
                           wall_sec, tp, avg_lat, ram, sub_ms });
    }

    std::cout << "  >> Scaling test complete\n";
    return points;
}


//  CSV REPORT — formatted for easy Excel import
static void write_csv_report(const std::vector<ScalePoint>& scaling,
                             const BurstResult& burst,
                             double baseline_ram_mb,
                             const std::string& path)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Could not open " << path << " for writing\n";
        return;
    }

    // --- Sheet 1: Scaling data ---
    f << "=== SCALING TEST RESULTS ===\n";
    f << "num_jobs,succeeded,failed,wall_time_sec,throughput_jobs_per_sec,"
      << "avg_latency_ms,ram_mb,submit_phase_ms\n";

    for (auto& pt : scaling) {
        f << pt.num_jobs       << ","
          << pt.succeeded      << ","
          << pt.failed         << ","
          << std::fixed << std::setprecision(3) << pt.wall_sec     << ","
          << std::setprecision(4) << pt.throughput   << ","
          << std::setprecision(2) << pt.avg_latency_ms << ","
          << std::setprecision(1) << pt.ram_mb         << ","
          << std::setprecision(2) << pt.submit_ms      << "\n";
    }

    // --- Sheet 2: Burst test summary ---
    f << "\n=== BURST TEST RESULTS ===\n";
    f << "total_clients,succeeded,failed,submit_ms,total_wall_ms,ram_after_mb\n";
    f << burst.total_jobs  << ","
      << burst.succeeded   << ","
      << burst.failed      << ","
      << std::fixed << std::setprecision(2) << burst.submit_ms   << ","
      << std::setprecision(2) << burst.total_ms    << ","
      << std::setprecision(1) << burst.ram_after_mb << "\n";

    // --- Sheet 3: Bottleneck analysis ---
    f << "\n=== BOTTLENECK ANALYSIS ===\n";
    f << "metric,value,interpretation\n";

    // find peak throughput
    auto best_tp = std::max_element(scaling.begin(), scaling.end(),
        [](const ScalePoint& a, const ScalePoint& b) {
            return a.throughput < b.throughput;
        });

    // find the point where throughput starts dropping (saturation)
    int saturation_load = -1;
    for (int i = 1; i < (int)scaling.size(); ++i) {
        if (scaling[i].throughput < scaling[i-1].throughput * 0.95) {
            saturation_load = scaling[i].num_jobs;
            break;
        }
    }

    // memory growth rate: (last_ram - first_ram) / (last_jobs - first_jobs)
    double ram_growth = 0;
    if (scaling.size() >= 2) {
        ram_growth = (scaling.back().ram_mb - scaling.front().ram_mb)
                   / (scaling.back().num_jobs - scaling.front().num_jobs);
    }

    // queue pressure: ratio of submit time vs total time at highest load
    double queue_pressure = 0;
    if (!scaling.empty() && scaling.back().wall_sec > 0) {
        queue_pressure = (scaling.back().submit_ms / 1000.0) / scaling.back().wall_sec;
    }

    f << "peak_throughput_jobs_sec,"
      << std::setprecision(4) << best_tp->throughput << ","
      << "Best throughput at " << best_tp->num_jobs << " jobs\n";

    f << "peak_throughput_load," << best_tp->num_jobs << ","
      << "Job count that achieved max throughput\n";

    if (saturation_load > 0) {
        f << "saturation_point," << saturation_load << ","
          << "Throughput dropped >5% here - system saturated\n";
    } else {
        f << "saturation_point,N/A,"
          << "Throughput never dropped significantly - system not saturated\n";
    }

    f << "ram_baseline_mb,"
      << std::setprecision(1) << baseline_ram_mb << ","
      << "RAM right after model load (before any jobs)\n";

    f << "ram_peak_mb,"
      << std::setprecision(1) << scaling.back().ram_mb << ","
      << "RAM at highest load level\n";

    f << "ram_growth_mb_per_job,"
      << std::setprecision(4) << ram_growth << ","
      << "Extra MB consumed per additional job in queue\n";

    f << "queue_pressure_ratio,"
      << std::setprecision(4) << queue_pressure << ","
      << "Fraction of wall time spent just submitting (>0.3 = queue bottleneck)\n";

    f << "burst_submit_ms,"
      << std::setprecision(2) << burst.submit_ms << ","
      << burst.total_jobs << " simultaneous clients submission time\n";

    f.close();
    std::cout << "\n[CSV] Report written to: " << path << "\n";
}



static void print_summary_table(const std::vector<ScalePoint>& scaling) {
    const int W = 12;

    std::cout << "\n" << std::string(96, '=') << "\n";
    std::cout << "  STRESS TEST SCALING RESULTS  (M3 Hybrid: Threading + Batch Inference)\n";
    std::cout << std::string(96, '=') << "\n";

    std::cout << std::left
              << std::setw(W)   << "Jobs"
              << std::setw(W)   << "OK"
              << std::setw(W)   << "Fail"
              << std::setw(W+2) << "Wall(s)"
              << std::setw(W+4) << "Thruput"
              << std::setw(W+4) << "Latency(ms)"
              << std::setw(W)   << "RAM(MB)"
              << "Notes\n";
    std::cout << std::string(96, '-') << "\n";

    // figure out which row has best throughput so we can tag it
    double best_tp = 0;
    int best_idx = 0;
    for (int i = 0; i < (int)scaling.size(); ++i) {
        if (scaling[i].throughput > best_tp) {
            best_tp = scaling[i].throughput;
            best_idx = i;
        }
    }

    for (int i = 0; i < (int)scaling.size(); ++i) {
        auto& pt = scaling[i];
        std::string note;
        if (i == best_idx) note = "[PEAK THROUGHPUT]";
        if (i > 0 && pt.throughput < scaling[i-1].throughput * 0.95)
            note += " [SATURATING]";

        std::cout << std::left << std::fixed
                  << std::setw(W)   << pt.num_jobs
                  << std::setw(W)   << pt.succeeded
                  << std::setw(W)   << pt.failed
                  << std::setw(W+2) << std::setprecision(2) << pt.wall_sec
                  << std::setw(W+4) << std::setprecision(3) << pt.throughput
                  << std::setw(W+4) << std::setprecision(1) << pt.avg_latency_ms
                  << std::setw(W)   << std::setprecision(0) << pt.ram_mb
                  << note << "\n";
    }
    std::cout << std::string(96, '=') << "\n";
}


int main(int argc, char* argv[]) {

    const std::string model_path = (argc > 1)
        ? argv[1]
        : "models/tinyllama.gguf";

    const std::string csv_path = (argc > 2) ? argv[2] : "stress_test_m3_results.csv";

    std::cout << "============================================\n";
    std::cout << "  M3 STRESS TEST — Hybrid Inference Engine\n";
    std::cout << "============================================\n";
    std::cout << "Model : " << model_path << "\n";
    std::cout << "Output: " << csv_path   << "\n\n";

    // InferenceEngine wraps everything: model, dispatcher, workers, API
    // Using 2 workers, batch_size=4, 4 threads per context (matches project defaults)
    InferenceEngine engine(model_path, 2, 4, 4);
    InferenceAPI* api = engine.get_api();

    double baseline_ram = get_ram_mb();
    std::cout << "Engine online. Baseline RAM: " << std::fixed
              << std::setprecision(0) << baseline_ram << " MB\n";

    std::cout << "\n[WARMUP] Submitting " << WARMUP_JOBS << " throwaway jobs...\n";
    {
        std::vector<std::string> warmup_ids;
        warmup_ids.reserve(WARMUP_JOBS);
        for (int i = 0; i < WARMUP_JOBS; ++i) {
            warmup_ids.push_back(
                api->submit_req("Warmup prompt " + std::to_string(i),
                                TOKENS_PER_JOB, TEMPERATURE));
        }
        CompletionStats ws = wait_for_all(api, warmup_ids);
        std::cout << "[WARMUP] Done. " << ws.succeeded << " OK, "
                  << ws.failed << " failed.\n";
    }

    // RUN THE TESTS

    // Test 1: Input Validation (same style as M1_test.cpp)
    test_input_validation(api);

    // Test 2: Concurrent burst — many threads slamming the API at once
    BurstResult burst = test_concurrent_burst(api, BURST_CLIENTS);

    // Test 3: Ramp up the load and record scaling behavior
    std::vector<ScalePoint> scaling = test_ramp_scaling(api);

    print_summary_table(scaling);
    write_csv_report(scaling, burst, baseline_ram, csv_path);

    std::cout << "\n============================================\n";
    std::cout << "  ALL STRESS TESTS COMPLETED SUCCESSFULLY\n";
    std::cout << "============================================\n";

    return 0;
}
