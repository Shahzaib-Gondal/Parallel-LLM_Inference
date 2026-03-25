
#pragma once
#include "InferenceResult.h"
#include <string>

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

    InferenceResult run_inference(const std::string& prompt,
                                  int   max_new_tokens = 128,
                                  float temperature    = 0.7f,
                                  float top_p          = 0.9f);

    bool is_loaded() const { return model_ != nullptr && ctx_ != nullptr; }

private:
    llama_model*   model_    = nullptr;
    llama_context* ctx_      = nullptr;
    int            n_ctx_;
    int            n_threads_;
    unsigned int   seed_;
};