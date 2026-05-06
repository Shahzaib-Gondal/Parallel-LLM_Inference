#pragma once
#include "InferenceResult.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>

struct llama_model;
struct llama_context;

class ModelWrapper {
public:
    explicit ModelWrapper(const std::string& model_path,
                          int n_ctx         = 512,
                          int n_threads     = 4,
                          unsigned int seed = 42);
    ~ModelWrapper();

    ModelWrapper(const ModelWrapper&)            = delete;
    ModelWrapper& operator=(const ModelWrapper&) = delete;

    void thread_context(int thread_id = 0);

    InferenceResult run_inference(const std::string& prompt,
                                  int   max_new_tokens = 128,
                                  float temperature    = 0.7f,
                                  float top_p          = 0.9f);

    std::vector<InferenceResult> run_batch_inference(
                                  const std::vector<std::string>& prompts,
                                  int   max_new_tokens = 128,
                                  float temperature    = 0.7f,
                                  float top_p          = 0.9f);

    bool is_loaded() const { return model_ != nullptr; }

private:
    llama_model*  model_     = nullptr;
    int           n_ctx_;
    int           n_threads_;
    unsigned int  seed_;
    std::unordered_map<std::thread::id, llama_context*> ctx_map_;
    mutable std::mutex ctx_mutex_;
    mutable std::mutex decode_mutex_;
};