#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_capacity_;
    std::atomic<bool> shutdown_{false}; // The Kill Switch

public:
    explicit ThreadSafeQueue(size_t capacity) : max_capacity_(capacity) {}

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]{ return queue_.size() < max_capacity_ || shutdown_; });
        if (shutdown_) return false;
        
        queue_.push(std::move(value));
        cv_not_empty_.notify_one();
        return true;
    }

    // Notice this now returns a BOOL instead of VOID
    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]{ return !queue_.empty() || shutdown_; }); 
        
        if (shutdown_ && queue_.empty()) return false; // Queue is dead, wake up and exit

        value = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void shutdown() {
        shutdown_ = true;
        cv_not_empty_.notify_all(); // Wake up any sleeping consumers
        cv_not_full_.notify_all();  // Wake up any sleeping producers
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};