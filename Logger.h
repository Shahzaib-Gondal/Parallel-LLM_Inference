#pragma once
#include <string>
#include <mutex>

enum class LogLevel {
    INFO,
    WARNING,
    LOG_ERROR,
    DEBUG
};

class Logger {
public:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    void log(const std::string& message, LogLevel level = LogLevel::INFO);
private:
    std::mutex mtx_;
    std::string get_current_time() const;
    std::string level_to_string(LogLevel level) const;
};