#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

using namespace std;

template <typename T>
class ThreadSafeQueue {
private:
    queue<T> queue_;
    mutable mutex mutex_;
    condition_variable cv_not_empty_;
    condition_variable cv_not_full_;
    size_t max_capacity_;

public:
    explicit ThreadSafeQueue(size_t capacity) : max_capacity_(capacity) {}

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    void push(T value) {
        unique_lock<mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]{ return queue_.size() < max_capacity_; });
        queue_.push(move(value));
        cv_not_empty_.notify_one();
    }

    bool try_pop(T& value) {
        lock_guard<mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void wait_and_pop(T& value) {
        unique_lock<mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]{ return !queue_.empty(); }); 
        value = move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
    }

    bool empty() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.size();
    }
};