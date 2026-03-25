#pragma once
#include <string>

enum class InferenceStatus {
    SUCCESS,
    FAILURE_MODEL_NOT_LOADED,
    FAILURE_EMPTY_PROMPT,
    FAILURE_TOKEN_LIMIT,
    FAILURE_RUNTIME_ERROR
};

struct InferenceResult {
    std::string output;
    InferenceStatus status;
    std::string error_message;
    int tokens_generated = 0;
};