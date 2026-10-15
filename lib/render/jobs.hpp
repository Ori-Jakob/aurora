#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::jobs {
enum class Priority : uint8_t { High, Normal, Low };
enum class JobStatus : uint8_t { Queued, Running, Done, Cancelled };

namespace detail {
struct JobState;
struct GroupState;
} // namespace detail

class JobContext {
public:
  explicit JobContext(const detail::JobState& state) noexcept : m_state(&state) {}
  [[nodiscard]] bool cancelled() const noexcept;

private:
  const detail::JobState* m_state;
};

using JobFunction = std::function<void(const JobContext&)>;

class JobHandle {
public:
  JobHandle() noexcept = default;
  explicit JobHandle(std::shared_ptr<detail::JobState> state) noexcept : m_state(std::move(state)) {}

  [[nodiscard]] bool valid() const noexcept { return m_state != nullptr; }
  [[nodiscard]] JobStatus status() const noexcept;
  bool cancel() const noexcept;
  void wait() const;

private:
  std::shared_ptr<detail::JobState> m_state;
};

class JobGroup {
public:
  JobGroup();
  JobHandle submit(JobFunction function, Priority priority = Priority::Normal, std::string_view label = {});
  void cancel() noexcept;
  void wait();
  [[nodiscard]] uint32_t pending() const noexcept;

private:
  std::shared_ptr<detail::GroupState> m_state;
};

template <typename T>
class ResultQueue {
public:
  void push(T value) {
    std::lock_guard lock{m_mutex};
    m_items.push_back(std::move(value));
    m_size.store(m_items.size(), std::memory_order_release);
  }

  template <typename F>
  size_t drain(F&& consume) {
    {
      std::lock_guard lock{m_mutex};
      m_draining.swap(m_items);
      m_size.store(0, std::memory_order_release);
    }
    const size_t count = m_draining.size();
    for (auto& item : m_draining) {
      consume(std::move(item));
    }
    m_draining.clear();
    return count;
  }

  [[nodiscard]] bool empty() const noexcept { return m_size.load(std::memory_order_acquire) == 0; }

private:
  std::mutex m_mutex;
  std::vector<T> m_items;
  std::vector<T> m_draining;
  std::atomic_size_t m_size = 0;
};

struct JobStats {
  uint32_t workers = 0;
  std::array<uint32_t, 3> queued{};
  uint32_t running = 0;
  uint64_t completed = 0;
  uint64_t cancelled = 0;
};

uint32_t worker_count_for(uint32_t hardwareThreads) noexcept;
uint32_t default_worker_count() noexcept;
void initialize(uint32_t workerCount);
void shutdown();
JobHandle submit(JobFunction function, Priority priority = Priority::Normal, std::string_view label = {});
void set_synchronous(bool synchronous) noexcept;
[[nodiscard]] bool synchronous() noexcept;
[[nodiscard]] bool is_job_thread() noexcept;
[[nodiscard]] JobStats stats() noexcept;
} // namespace aurora::render::jobs
