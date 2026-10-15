#include "jobs.hpp"

#include "../thread.hpp"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <stop_token>
#include <string>
#include <thread>

#include <tracy/Tracy.hpp>

namespace aurora::render::jobs {
namespace detail {
struct GroupState {
  std::atomic_uint32_t pending = 0;
  std::atomic_uint32_t cancelEpoch = 0;
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::weak_ptr<JobState>> members;
};

struct JobState {
  JobFunction function;
  std::string label;
  Priority priority = Priority::Normal;
  std::atomic<JobStatus> status = JobStatus::Queued;
  std::atomic_bool cancelRequested = false;
  std::mutex mutex;
  std::condition_variable cv;
  bool finished = false;
  std::shared_ptr<GroupState> group;
};
} // namespace detail

namespace {
constexpr size_t PriorityCount = 3;
using JobPtr = std::shared_ptr<detail::JobState>;

std::mutex g_queueMutex;
std::condition_variable_any g_queueCv;
std::array<std::deque<JobPtr>, PriorityCount> g_queues;
bool g_initialized = false;
std::atomic_uint32_t g_workerCount = 0;
std::atomic_bool g_stopping = false;
std::atomic_bool g_synchronous = false;
std::array<std::atomic_uint32_t, PriorityCount> g_queued{};
std::atomic_uint32_t g_startedWorkers = 0;
std::atomic_uint32_t g_running = 0;
std::atomic_uint64_t g_completed = 0;
std::atomic_uint64_t g_cancelled = 0;
// Declared after the queue state, so workers of a pool that was never shut down join before that state is destroyed
std::vector<thread::Thread> g_workers;
thread_local bool t_jobThread = false;

size_t priority_index(Priority priority) noexcept { return static_cast<size_t>(priority); }

bool is_finished(JobStatus status) noexcept { return status == JobStatus::Done || status == JobStatus::Cancelled; }

JobPtr make_job(JobFunction function, Priority priority, std::string_view label,
                std::shared_ptr<detail::GroupState> group) {
  auto state = std::make_shared<detail::JobState>();
  state->function = std::move(function);
  state->label = label;
  state->priority = priority;
  state->group = std::move(group);
  return state;
}

bool claim(detail::JobState& state, JobStatus next) noexcept {
  JobStatus expected = JobStatus::Queued;
  if (!state.status.compare_exchange_strong(expected, next, std::memory_order_acq_rel)) {
    return false;
  }
  g_queued[priority_index(state.priority)].fetch_sub(1, std::memory_order_relaxed);
  if (next == JobStatus::Running) {
    g_running.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

void execute(detail::JobState& state) {
  ZoneScopedN("Render job");
  if (!state.label.empty()) {
    ZoneText(state.label.data(), state.label.size());
  }
  state.function(JobContext{state});
}

void finish(detail::JobState& state, JobStatus status) {
  state.function = nullptr;
  if (status == JobStatus::Done) {
    g_running.fetch_sub(1, std::memory_order_relaxed);
    g_completed.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_cancelled.fetch_add(1, std::memory_order_relaxed);
  }
  const std::shared_ptr<detail::GroupState> group = std::move(state.group);
  {
    std::lock_guard lock{state.mutex};
    state.status.store(status, std::memory_order_release);
    state.finished = true;
  }
  state.cv.notify_all();
  if (group != nullptr && group->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard lock{group->mutex};
    group->cv.notify_all();
  }
}

void run_inline(detail::JobState& state) {
  state.status.store(JobStatus::Running, std::memory_order_release);
  g_running.fetch_add(1, std::memory_order_relaxed);
  execute(state);
  finish(state, JobStatus::Done);
}

bool has_queued() noexcept {
  return std::ranges::any_of(g_queues, [](const std::deque<JobPtr>& queue) { return !queue.empty(); });
}

JobPtr pop_job() {
  for (auto& queue : g_queues) {
    if (!queue.empty()) {
      JobPtr job = std::move(queue.front());
      queue.pop_front();
      return job;
    }
  }
  return nullptr;
}

void worker_main(std::stop_token token) {
  t_jobThread = true;
  while (true) {
    JobPtr job;
    {
      std::unique_lock lock{g_queueMutex};
      g_queueCv.wait(lock, token, [] { return g_stopping.load(std::memory_order_relaxed) || has_queued(); });
      if (token.stop_requested() || g_stopping.load(std::memory_order_relaxed)) {
        break;
      }
      job = pop_job();
    }
    // A failed claim means a waiter stole the job or it was cancelled while queued
    if (claim(*job, JobStatus::Running)) {
      execute(*job);
      finish(*job, JobStatus::Done);
    }
  }
}

void start_workers() {
  if (!g_workers.empty()) {
    return;
  }
  const uint32_t count = g_workerCount.load(std::memory_order_relaxed);
  g_workers.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    g_workers.emplace_back(
        thread::Options{.name = "Render job " + std::to_string(i), .priority = thread::Priority::Low}, &worker_main);
  }
  g_startedWorkers.store(count, std::memory_order_relaxed);
}

JobHandle dispatch(JobPtr state) {
  if (g_workerCount.load(std::memory_order_acquire) == 0 || g_synchronous.load(std::memory_order_acquire)) {
    run_inline(*state);
    return JobHandle{std::move(state)};
  }
  bool queued = false;
  bool stopping = false;
  {
    std::lock_guard lock{g_queueMutex};
    stopping = g_stopping.load(std::memory_order_relaxed);
    if (!stopping && g_workerCount.load(std::memory_order_relaxed) != 0) {
      start_workers();
      g_queues[priority_index(state->priority)].push_back(state);
      g_queued[priority_index(state->priority)].fetch_add(1, std::memory_order_relaxed);
      queued = true;
    }
  }
  if (queued) {
    g_queueCv.notify_one();
  } else if (stopping) {
    // No worker will take a job queued now, so a nested submit during shutdown must not leave a waiter behind
    finish(*state, JobStatus::Cancelled);
  } else {
    run_inline(*state);
  }
  return JobHandle{std::move(state)};
}
} // namespace

bool JobContext::cancelled() const noexcept {
  return m_state->cancelRequested.load(std::memory_order_acquire) || g_stopping.load(std::memory_order_acquire);
}

JobStatus JobHandle::status() const noexcept {
  return m_state != nullptr ? m_state->status.load(std::memory_order_acquire) : JobStatus::Done;
}

bool JobHandle::cancel() const noexcept {
  if (m_state == nullptr) {
    return false;
  }
  m_state->cancelRequested.store(true, std::memory_order_release);
  if (!claim(*m_state, JobStatus::Cancelled)) {
    return false;
  }
  finish(*m_state, JobStatus::Cancelled);
  return true;
}

void JobHandle::wait() const {
  // Jobs never wait on jobs: once every worker waits, nothing is left to run what they wait on
  assert(!is_job_thread());
  if (m_state == nullptr) {
    return;
  }
  if (claim(*m_state, JobStatus::Running)) {
    execute(*m_state);
    finish(*m_state, JobStatus::Done);
    return;
  }
  std::unique_lock lock{m_state->mutex};
  m_state->cv.wait(lock, [this] { return m_state->finished; });
}

JobGroup::JobGroup() : m_state(std::make_shared<detail::GroupState>()) {}

JobHandle JobGroup::submit(JobFunction function, Priority priority, std::string_view label) {
  m_state->pending.fetch_add(1, std::memory_order_acq_rel);
  const uint32_t epoch = m_state->cancelEpoch.load(std::memory_order_acquire);
  JobPtr state = make_job(std::move(function), priority, label, m_state);
  std::weak_ptr<detail::JobState> member = state;
  JobHandle handle = dispatch(std::move(state));
  {
    // Follows dispatch so a cancel can't claim a job that isn't queued yet; the epoch check covers the gap
    std::lock_guard lock{m_state->mutex};
    auto& members = m_state->members;
    // Pruning only when the vector would grow keeps long-lived groups linear in their submits
    if (members.size() == members.capacity()) {
      std::erase_if(members, [](const std::weak_ptr<detail::JobState>& weak) {
        const JobPtr job = weak.lock();
        return job == nullptr || is_finished(job->status.load(std::memory_order_acquire));
      });
    }
    members.push_back(std::move(member));
  }
  if (m_state->cancelEpoch.load(std::memory_order_acquire) != epoch) {
    handle.cancel();
  }
  return handle;
}

void JobGroup::cancel() noexcept {
  m_state->cancelEpoch.fetch_add(1, std::memory_order_acq_rel);
  std::vector<JobPtr> members;
  {
    std::lock_guard lock{m_state->mutex};
    members.reserve(m_state->members.size());
    for (const auto& weak : m_state->members) {
      if (JobPtr job = weak.lock()) {
        members.push_back(std::move(job));
      }
    }
  }
  for (JobPtr& job : members) {
    JobHandle{std::move(job)}.cancel();
  }
}

void JobGroup::wait() {
  assert(!is_job_thread());
  std::unique_lock lock{m_state->mutex};
  m_state->cv.wait(lock, [this] { return m_state->pending.load(std::memory_order_acquire) == 0; });
}

uint32_t JobGroup::pending() const noexcept { return m_state->pending.load(std::memory_order_acquire); }

uint32_t worker_count_for(uint32_t hardwareThreads) noexcept {
  // The game, FIFO, render worker and GX pipeline threads already keep four cores busy
  return static_cast<uint32_t>(std::clamp<int64_t>(static_cast<int64_t>(hardwareThreads) - 4, 2, 4));
}

uint32_t default_worker_count() noexcept { return worker_count_for(std::thread::hardware_concurrency()); }

void initialize(uint32_t workerCount) {
  std::lock_guard lock{g_queueMutex};
  assert(!g_initialized);
  g_initialized = true;
  g_workerCount.store(workerCount, std::memory_order_release);
  g_completed.store(0, std::memory_order_relaxed);
  g_cancelled.store(0, std::memory_order_relaxed);
}

void shutdown() {
  // A worker would wait to join itself
  assert(!is_job_thread());
  std::vector<JobPtr> dropped;
  std::vector<thread::Thread> workers;
  {
    std::lock_guard lock{g_queueMutex};
    if (!g_initialized) {
      return;
    }
    g_stopping.store(true, std::memory_order_release);
    for (auto& queue : g_queues) {
      std::ranges::move(queue, std::back_inserter(dropped));
      queue.clear();
    }
    workers.swap(g_workers);
  }
  for (const JobPtr& job : dropped) {
    if (claim(*job, JobStatus::Cancelled)) {
      finish(*job, JobStatus::Cancelled);
    }
  }
  dropped.clear();
  g_queueCv.notify_all();
  for (thread::Thread& worker : workers) {
    worker.request_stop();
  }
  for (thread::Thread& worker : workers) {
    worker.join();
  }
  workers.clear();
  {
    std::lock_guard lock{g_queueMutex};
    g_workerCount.store(0, std::memory_order_release);
    g_startedWorkers.store(0, std::memory_order_relaxed);
    g_initialized = false;
    g_stopping.store(false, std::memory_order_release);
  }
  g_synchronous.store(false, std::memory_order_release);
}

JobHandle submit(JobFunction function, Priority priority, std::string_view label) {
  return dispatch(make_job(std::move(function), priority, label, nullptr));
}

void set_synchronous(bool synchronous) noexcept { g_synchronous.store(synchronous, std::memory_order_release); }

bool synchronous() noexcept { return g_synchronous.load(std::memory_order_acquire); }

bool is_job_thread() noexcept { return t_jobThread; }

JobStats stats() noexcept {
  JobStats out;
  out.workers = g_startedWorkers.load(std::memory_order_relaxed);
  for (size_t i = 0; i < PriorityCount; ++i) {
    out.queued[i] = g_queued[i].load(std::memory_order_relaxed);
  }
  out.running = g_running.load(std::memory_order_relaxed);
  out.completed = g_completed.load(std::memory_order_relaxed);
  out.cancelled = g_cancelled.load(std::memory_order_relaxed);
  return out;
}
} // namespace aurora::render::jobs
