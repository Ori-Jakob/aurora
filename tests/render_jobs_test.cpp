#include <gtest/gtest.h>

#include "render/jobs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace std::chrono_literals;
namespace jobs = aurora::render::jobs;
using jobs::JobContext;
using jobs::JobFunction;
using jobs::JobGroup;
using jobs::JobHandle;
using jobs::JobStatus;
using jobs::Priority;

constexpr uint32_t DefaultWorkers = 4;

struct Gate {
  std::promise<void> started;
  std::shared_future<void> startedFuture = started.get_future().share();
  std::atomic_bool released = false;
  std::atomic_bool leftByCancel = false;

  void wait_started() const { startedFuture.wait(); }
  void release() { released.store(true); }
};

void wait_on_job_thread(const std::function<void()>& wait) {
  std::promise<void> done;
  std::future<void> future = done.get_future();
  const JobHandle handle = jobs::submit([&](const JobContext&) {
    wait();
    done.set_value();
  });
  future.wait();
  handle.wait();
}

class RenderJobs : public testing::Test {
protected:
  void SetUp() override { jobs::initialize(DefaultWorkers); }
  void TearDown() override { jobs::shutdown(); }

  static void restart(uint32_t workers) {
    jobs::shutdown();
    jobs::initialize(workers);
  }

  Gate& make_gate() { return *m_gates.emplace_back(std::make_unique<Gate>()); }

  static JobFunction gate_body(Gate& gate, JobFunction after = {}) {
    return [&gate, after = std::move(after)](const JobContext& ctx) {
      gate.started.set_value();
      while (!gate.released.load()) {
        if (ctx.cancelled()) {
          gate.leftByCancel.store(true);
          break;
        }
        std::this_thread::sleep_for(1ms);
      }
      if (after) {
        after(ctx);
      }
    };
  }

  JobHandle hold_worker(Gate& gate) {
    JobHandle handle = jobs::submit(gate_body(gate));
    gate.wait_started();
    return handle;
  }

private:
  // Owned here so TearDown's shutdown, which cancels any gate a failed test left polling, runs before they go away
  std::vector<std::unique_ptr<Gate>> m_gates;
};

TEST_F(RenderJobs, WorkerCountFormula) {
  EXPECT_EQ(jobs::worker_count_for(0), 2u);
  EXPECT_EQ(jobs::worker_count_for(4), 2u);
  EXPECT_EQ(jobs::worker_count_for(6), 2u);
  EXPECT_EQ(jobs::worker_count_for(7), 3u);
  EXPECT_EQ(jobs::worker_count_for(8), 4u);
  EXPECT_EQ(jobs::worker_count_for(64), 4u);
}

TEST_F(RenderJobs, RunsAllSubmittedJobs) {
  constexpr uint32_t Count = 10000;
  std::atomic_uint32_t value = 0;
  std::vector<JobHandle> handles;
  handles.reserve(Count);
  for (uint32_t i = 0; i < Count; ++i) {
    handles.push_back(jobs::submit([&value](const JobContext&) { value.fetch_add(1); }));
  }
  for (const JobHandle& handle : handles) {
    handle.wait();
  }
  EXPECT_EQ(value.load(), Count);
  const jobs::JobStats stats = jobs::stats();
  EXPECT_EQ(stats.completed, Count);
  EXPECT_EQ(stats.running, 0u);
  EXPECT_EQ(stats.queued, (std::array<uint32_t, 3>{}));
}

TEST_F(RenderJobs, WorkersStartLazily) {
  EXPECT_EQ(jobs::stats().workers, 0u);
  jobs::submit([](const JobContext&) {}).wait();
  EXPECT_EQ(jobs::stats().workers, DefaultWorkers);
}

TEST_F(RenderJobs, HigherPriorityRunsFirst) {
  restart(1);
  Gate& gate = make_gate();
  hold_worker(gate);
  std::string order;
  JobGroup group;
  group.submit([&order](const JobContext&) { order += 'A'; }, Priority::Low);
  group.submit([&order](const JobContext&) { order += 'B'; }, Priority::Normal);
  group.submit([&order](const JobContext&) { order += 'C'; }, Priority::High);
  const jobs::JobStats queued = jobs::stats();
  EXPECT_EQ(queued.queued, (std::array<uint32_t, 3>{1, 1, 1}));
  gate.release();
  group.wait();
  EXPECT_EQ(order, "CBA");
}

TEST_F(RenderJobs, CancelQueuedJob) {
  restart(1);
  Gate& gate = make_gate();
  hold_worker(gate);
  std::atomic_bool ran = false;
  const JobHandle queued = jobs::submit([&ran](const JobContext&) { ran.store(true); });
  EXPECT_TRUE(queued.cancel());
  EXPECT_EQ(queued.status(), JobStatus::Cancelled);
  EXPECT_FALSE(queued.cancel());
  EXPECT_EQ(jobs::stats().cancelled, 1u);
  gate.release();
  queued.wait();
  jobs::shutdown();
  EXPECT_FALSE(ran.load());
  EXPECT_EQ(jobs::stats().cancelled, 1u);
}

TEST_F(RenderJobs, CancelRunningJobSetsFlag) {
  Gate& gate = make_gate();
  const JobHandle running = hold_worker(gate);
  EXPECT_FALSE(running.cancel());
  running.wait();
  EXPECT_TRUE(gate.leftByCancel.load());
  EXPECT_EQ(running.status(), JobStatus::Done);
  EXPECT_FALSE(running.cancel());
}

TEST_F(RenderJobs, WaitStealsQueuedJob) {
  restart(1);
  Gate& gate = make_gate();
  hold_worker(gate);
  std::thread::id ranOn;
  const JobHandle queued = jobs::submit([&ranOn](const JobContext&) { ranOn = std::this_thread::get_id(); });
  queued.wait();
  EXPECT_EQ(ranOn, std::this_thread::get_id());
  EXPECT_EQ(queued.status(), JobStatus::Done);
  gate.release();
}

TEST_F(RenderJobs, ZeroWorkersRunInline) {
  restart(0);
  std::thread::id ranOn;
  const JobHandle handle = jobs::submit([&ranOn](const JobContext&) { ranOn = std::this_thread::get_id(); });
  EXPECT_EQ(handle.status(), JobStatus::Done);
  EXPECT_EQ(ranOn, std::this_thread::get_id());
  EXPECT_EQ(jobs::stats().workers, 0u);
}

TEST_F(RenderJobs, SynchronousModeRunsInline) {
  jobs::set_synchronous(true);
  EXPECT_TRUE(jobs::synchronous());
  std::thread::id ranOn;
  const JobHandle inlined = jobs::submit([&ranOn](const JobContext&) { ranOn = std::this_thread::get_id(); });
  EXPECT_EQ(inlined.status(), JobStatus::Done);
  EXPECT_EQ(ranOn, std::this_thread::get_id());

  jobs::set_synchronous(false);
  std::promise<bool> onJobThread;
  std::future<bool> result = onJobThread.get_future();
  const JobHandle queued =
      jobs::submit([&onJobThread](const JobContext&) { onJobThread.set_value(jobs::is_job_thread()); });
  EXPECT_TRUE(result.get());
  queued.wait();
  EXPECT_FALSE(jobs::is_job_thread());
}

TEST_F(RenderJobs, GroupWaitAndCancel) {
  constexpr uint32_t Count = 100;
  std::atomic_bool released = false;
  std::atomic_uint32_t executed = 0;
  std::atomic_uint32_t sawCancel = 0;
  JobGroup group;
  std::vector<JobHandle> handles;
  for (uint32_t i = 0; i < Count; ++i) {
    handles.push_back(group.submit([&](const JobContext& ctx) {
      executed.fetch_add(1);
      while (!released.load()) {
        std::this_thread::sleep_for(1ms);
      }
      if (ctx.cancelled()) {
        sawCancel.fetch_add(1);
      }
    }));
  }
  group.cancel();
  released.store(true);
  group.wait();
  EXPECT_EQ(group.pending(), 0u);
  const auto cancelled = static_cast<uint32_t>(
      std::ranges::count_if(handles, [](const JobHandle& handle) { return handle.status() == JobStatus::Cancelled; }));
  EXPECT_LE(executed.load(), DefaultWorkers);
  EXPECT_GE(cancelled, Count - DefaultWorkers);
  EXPECT_EQ(executed.load() + cancelled, Count);
  EXPECT_EQ(sawCancel.load(), executed.load());
  EXPECT_EQ(jobs::stats().cancelled, cancelled);
}

TEST_F(RenderJobs, GroupCancelReachesMemberStillBeingSubmitted) {
  // The member cancels its own group on a worker, often before submit has registered it
  for (int i = 0; i < 200; ++i) {
    JobGroup group;
    std::atomic_bool submitted = false;
    std::atomic_bool flagged = false;
    const JobHandle handle = group.submit([&](const JobContext& ctx) {
      group.cancel();
      while (!submitted.load()) {
        std::this_thread::yield();
      }
      flagged.store(ctx.cancelled());
    });
    submitted.store(true);
    handle.wait();
    ASSERT_TRUE(flagged.load()) << "iteration " << i;
  }
}

TEST_F(RenderJobs, ResultQueueKeepsPerProducerOrder) {
  constexpr uint32_t Producers = 4;
  constexpr uint32_t PerProducer = 10000;
  jobs::ResultQueue<std::pair<uint32_t, uint32_t>> results;
  std::vector<JobHandle> handles;
  for (uint32_t producer = 0; producer < Producers; ++producer) {
    handles.push_back(jobs::submit([&results, producer](const JobContext&) {
      for (uint32_t i = 0; i < PerProducer; ++i) {
        results.push({producer, i});
      }
    }));
  }
  std::array<int64_t, Producers> last;
  last.fill(-1);
  bool ordered = true;
  size_t received = 0;
  while (received < Producers * PerProducer) {
    const size_t drained = results.drain([&](std::pair<uint32_t, uint32_t> item) {
      int64_t& previous = last[item.first];
      ordered = ordered && static_cast<int64_t>(item.second) > previous;
      previous = item.second;
    });
    if (drained == 0) {
      std::this_thread::yield();
    }
    received += drained;
  }
  for (const JobHandle& handle : handles) {
    handle.wait();
  }
  EXPECT_TRUE(ordered);
  EXPECT_EQ(received, Producers * PerProducer);
  EXPECT_TRUE(results.empty());
}

TEST_F(RenderJobs, ShutdownCancelsQueuedJobs) {
  restart(1);
  Gate& gate = make_gate();
  const JobHandle held = hold_worker(gate);
  std::atomic_uint32_t ran = 0;
  std::vector<JobHandle> queued;
  for (int i = 0; i < 10; ++i) {
    queued.push_back(jobs::submit([&ran](const JobContext&) { ran.fetch_add(1); }));
  }
  std::thread stopper{[] { jobs::shutdown(); }};
  stopper.join();
  EXPECT_TRUE(gate.leftByCancel.load());
  EXPECT_EQ(held.status(), JobStatus::Done);
  EXPECT_EQ(ran.load(), 0u);
  for (const JobHandle& handle : queued) {
    EXPECT_EQ(handle.status(), JobStatus::Cancelled);
  }
  EXPECT_EQ(jobs::stats().cancelled, 10u);
  EXPECT_EQ(jobs::stats().workers, 0u);
}

TEST_F(RenderJobs, NestedSubmitDuringShutdownIsCancelled) {
  restart(1);
  Gate& gate = make_gate();
  JobGroup group;
  JobHandle nested;
  std::atomic_bool nestedRan = false;
  group.submit(gate_body(gate, [group, &nested, &nestedRan](const JobContext&) mutable {
    nested = group.submit([&nestedRan](const JobContext&) { nestedRan.store(true); });
  }));
  gate.wait_started();
  std::thread stopper{[] { jobs::shutdown(); }};
  stopper.join();
  EXPECT_TRUE(gate.leftByCancel.load());
  ASSERT_TRUE(nested.valid());
  EXPECT_EQ(nested.status(), JobStatus::Cancelled);
  EXPECT_FALSE(nestedRan.load());
  group.wait();
  EXPECT_EQ(group.pending(), 0u);
}

TEST_F(RenderJobs, SubmitAfterShutdownRunsInline) {
  jobs::shutdown();
  std::thread::id ranOn;
  const JobHandle handle = jobs::submit([&ranOn](const JobContext&) { ranOn = std::this_thread::get_id(); });
  EXPECT_EQ(handle.status(), JobStatus::Done);
  EXPECT_EQ(ranOn, std::this_thread::get_id());
}

TEST_F(RenderJobs, WaitOnJobThreadAsserts) {
  // Fork-style death tests would drop the worker threads the statement needs
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const JobHandle finished = jobs::submit([](const JobContext&) {});
  finished.wait();
  JobGroup group;
  EXPECT_DEBUG_DEATH(wait_on_job_thread([&] { finished.wait(); }), "is_job_thread");
  EXPECT_DEBUG_DEATH(wait_on_job_thread([&] { group.wait(); }), "is_job_thread");
}
} // namespace
