#include "ModelWrapper.h"
#include "D:/SSGC/Downloads/pdc project/llama.cpp/include/llama.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>

// NOTE: NO thread_local line here anymore

ModelWrapper::ModelWrapper(const std::string& model_path,
                           int n_ctx, int n_threads, unsigned int seed)
    : n_ctx_(n_ctx), n_threads_(n_threads), seed_(seed)
{
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_)
        throw std::runtime_error("Failed to load model: " + model_path);

    std::cerr << "[ModelWrapper] Loaded OK. ctx=" << n_ctx_
              << " threads=" << n_threads_ << " seed=" << seed_ << "\n";
}

ModelWrapper::~ModelWrapper() {
    {
        std::lock_guard<std::mutex> lock(ctx_mutex_);
        for (auto& [id, ctx] : ctx_map_)
            if (ctx) llama_free(ctx);
        ctx_map_.clear();
    }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
    llama_backend_free();
    std::cerr << "[ModelWrapper] Cleaned up.\n";
}

void ModelWrapper::thread_context(int thread_id) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx_;
    cparams.n_threads = n_threads_;

    llama_context* ctx = llama_init_from_model(model_, cparams);
    if (!ctx)
        throw std::runtime_error("Failed to create context for thread "
                                 + std::to_string(thread_id));

    auto tid = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock(ctx_mutex_);
        // free old context if this thread is re-registering
        auto it = ctx_map_.find(tid);
        if (it != ctx_map_.end() && it->second)
            llama_free(it->second);
        ctx_map_[tid] = ctx;
    }
    std::cerr << "[ModelWrapper] Context registered for thread "
              << tid << " (id=" << thread_id << ")\n";
}

InferenceResult ModelWrapper::run_inference(const std::string& prompt,
                                            int max_new_tokens,
                                            float temperature,
                                            float top_p)
{
    if (prompt.empty())
        return {"", InferenceStatus::FAILURE_EMPTY_PROMPT, "Prompt is empty", 0};

    // Look up context for THIS thread
    llama_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(ctx_mutex_);
        auto it = ctx_map_.find(std::this_thread::get_id());
        if (it != ctx_map_.end()) ctx = it->second;
    }

    if (!ctx) {
        std::cerr << "[ERROR] run_inference: no context for thread "
                  << std::this_thread::get_id() << "\n";
        return {"", InferenceStatus::FAILURE_MODEL_NOT_LOADED,
                "No context for this thread — call thread_context() first", 0};
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(model_);

        std::vector<llama_token> tokens(n_ctx_);
        int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(), (int)prompt.size(),
            tokens.data(), (int)tokens.size(),
            true, false
        );

        if (n_tokens < 0)
            return {"", InferenceStatus::FAILURE_TOKEN_LIMIT, "Prompt too long", 0};
        tokens.resize(n_tokens);

        // -------- FIRST DECODE (prompt) with MUTEX ----------
        {
            std::lock_guard<std::mutex> lock(decode_mutex_);
            llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
            if (llama_decode(ctx, batch) != 0)
                return {"", InferenceStatus::FAILURE_RUNTIME_ERROR, "Decode failed", 0};
        }

        std::string output;
        int generated = 0;

        auto* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed_));

        for (int i = 0; i < max_new_tokens; i++) {
            // Sample token (no decode yet)
            llama_token tok;
            {
                std::lock_guard<std::mutex> lock(decode_mutex_); // optional, but safe
                tok = llama_sampler_sample(sampler, ctx, -1);
            }

            if (llama_vocab_is_eog(vocab, tok)) break;

            char buf[256] = {};
            int  n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            if (n > 0) output.append(buf, n);

            // -------- PER‑TOKEN DECODE with MUTEX ----------
            {
                std::lock_guard<std::mutex> lock(decode_mutex_);
                llama_batch nb = llama_batch_get_one(&tok, 1);
                if (llama_decode(ctx, nb) != 0) break;
            }
            generated++;
        }

        llama_sampler_free(sampler);
        llama_memory_clear(llama_get_memory(ctx), true);

        return {output, InferenceStatus::SUCCESS, "", generated};

    } catch (const std::exception& e) {
        return {"", InferenceStatus::FAILURE_RUNTIME_ERROR,
                std::string("Exception: ") + e.what(), 0};
    }
}


std::vector<InferenceResult> ModelWrapper::run_batch_inference(
    const std::vector<std::string>& prompts,
    int max_new_tokens,
    float temperature,
    float top_p)
{
    std::vector<InferenceResult> results;
    results.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        results.push_back(run_inference(prompt, max_new_tokens, temperature, top_p));
    }
    return results;
}