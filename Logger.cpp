#include "Logger.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>

void Logger::log(const std::string& message, LogLevel level) {
    std::string time_str = get_current_time();
    std::string level_str = level_to_string(level);
    auto thread_id = std::this_thread::get_id();

    std::ostringstream oss;
    oss << "[" << time_str << "] " 
        << "[" << level_str << "] "
        << "[Thread " << thread_id << "] " 
        << message << "\n";

    // Lock the mutex so writes to cout are thread-safe and don't interleave
    std::lock_guard<std::mutex> lock(mtx_);
    
    // Write to standard output (or standard error if ERROR)
    if (level == LogLevel::LOG_ERROR) {
        std::cerr << oss.str();
    } else {
        std::cout << oss.str();
    }
}

std::string Logger::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&bt, &time_t_now);
#else
    localtime_r(&time_t_now, &bt);
#endif
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) const {
    switch(level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::LOG_ERROR: return "ERROR";
        case LogLevel::DEBUG: return "DEBUG";
        default: return "UNKNOWN";
    }
}
