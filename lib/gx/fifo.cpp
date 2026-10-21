#include "fifo.hpp"

#include "../thread.hpp"
#include "command_processor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

#include <SDL3/SDL_atomic.h>
#include <SDL3/SDL_stdinc.h>
#include <tracy/Tracy.hpp>

namespace aurora::gx::fifo {
namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

namespace {
constexpr Module Log{"aurora::gx::fifo"};
constexpr auto kProcessingMode = ProcessingMode::Thread;
constexpr uint32_t kDrawBatchSize = 1;
constexpr auto kSpinBeforeSleep = std::chrono::microseconds{0};
static_assert(kSpinBeforeSleep <= std::chrono::microseconds{100}, "the FIFO spin must stay far below a frame");

// Publisher-written and worker-written words get separate lines so a spinning worker costs one transfer per publish
struct alignas(64) PublishLine {
  bool frameActive = false;
  bool synchronousPublish = false;
  uint32_t pendingDraws = 0;
  std::atomic<uint64_t> published{0};
  std::atomic<bool> workerParked{false};
};
struct alignas(64) ProcessLine {
  std::atomic<uint64_t> processed{0};
  std::atomic<uint32_t> waiters{0};
};
struct alignas(64) GameCounters {
  std::atomic<uint64_t> frames{0}, publishes{0}, wakeCalls{0};
  std::array<std::atomic<uint64_t>, 3> drains{};
  std::atomic<uint64_t> drainWaitNs{0}, drainWaitMaxNs{0};
  uint64_t frameDrainNs = 0;
  uint64_t processNsBase = 0;
};
struct alignas(64) WorkerCounters {
  std::atomic<uint64_t> parks{0}, chunks{0}, notifies{0};
};
PublishLine sPublish;
ProcessLine sProcess;
GameCounters sGame;
WorkerCounters sWorker;
// Game thread only; 1 us buckets, the last one catches everything longer
std::array<uint32_t, 2048> sDrainWaitUs{};
std::array<uint32_t, 2048> sFrameDrainUs{};
std::atomic<testing::ParkHook> sParkHook{nullptr};
uint64_t sStreamBase = 0;
std::mutex sBufferMutex;
std::atomic<uint32_t> sWorkerWake{0};
thread::Thread sWorkerThread;
std::atomic<DrawDoneCallback> sDrawDoneCallback{nullptr};
std::atomic<const ProcessObserver*> sProcessObserver{nullptr};

inline void bump(std::atomic<uint64_t>& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void dispatch_draw_done() noexcept {
  if (const auto callback = sDrawDoneCallback.load(std::memory_order_acquire); callback != nullptr) {
    callback();
  }
}

void wake_worker() noexcept {
  sWorkerWake.fetch_add(1, std::memory_order_release);
  sWorkerWake.notify_all();
}

enum class WaitOutcome : uint8_t { Ready, Spun, Slept };

void run_park_hook(testing::ParkStep step) noexcept {
  if (const auto hook = sParkHook.load(std::memory_order_acquire); hook != nullptr)
    UNLIKELY { hook(step); }
}

void advance_published(uint64_t target) noexcept {
  // seq_cst pairs with the worker's park flag, so a publish skips the wake while the worker is awake
  sPublish.published.store(target, std::memory_order_seq_cst);
  // A cleared flag always belongs to a publisher whose exchange saw it set and woke the worker, so no wake is lost
  if (sPublish.workerParked.load(std::memory_order_seq_cst) &&
      sPublish.workerParked.exchange(false, std::memory_order_seq_cst)) {
    bump(sGame.wakeCalls);
    wake_worker();
  }
}

template <typename Ready>
bool spin_until(Ready&& ready) noexcept {
  if constexpr (kSpinBeforeSleep.count() == 0) {
    return false;
  } else {
    const auto deadline = std::chrono::steady_clock::now() + kSpinBeforeSleep;
    do {
      for (uint32_t i = 0; i < 64; ++i) {
        SDL_CPUPauseInstruction();
        if (ready()) {
          return true;
        }
      }
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }
}

WaitOutcome wait_for_processed(uint64_t target) noexcept {
  if (sProcess.processed.load(std::memory_order_acquire) >= target) {
    return WaitOutcome::Ready;
  }
  if (spin_until([target] { return sProcess.processed.load(std::memory_order_acquire) >= target; })) {
    return WaitOutcome::Spun;
  }
  sProcess.waiters.fetch_add(1, std::memory_order_seq_cst);
  uint64_t processed = sProcess.processed.load(std::memory_order_seq_cst);
  while (processed < target) {
    sProcess.processed.wait(processed, std::memory_order_acquire);
    processed = sProcess.processed.load(std::memory_order_acquire);
  }
  sProcess.waiters.fetch_sub(1, std::memory_order_relaxed);
  return WaitOutcome::Slept;
}

void process_to(uint64_t target) noexcept {
  uint64_t processed = sProcess.processed.load(std::memory_order_relaxed);
  while (processed < target) {
    bump(sWorker.chunks);
    ProcessResult result{};
    {
      std::lock_guard lock{sBufferMutex};
      AURORA_ASSERT(processed >= sStreamBase && target <= sStreamBase + detail::sBufferSize,
                    "FIFO processing range [{}, {}) is outside buffered range [{}, {})", processed, target, sStreamBase,
                    sStreamBase + detail::sBufferSize);
      const auto start = static_cast<uint32_t>(processed - sStreamBase);
      const auto size = static_cast<uint32_t>(target - processed);
      result = process(detail::sBufferData + start, size);
      if (const auto* observer = sProcessObserver.load(std::memory_order_acquire); observer != nullptr)
        UNLIKELY { observer->processed(observer->userData, detail::sBufferData + start, result.bytesProcessed); }
    }
    AURORA_ASSERT(result.bytesProcessed > 0 && result.bytesProcessed <= target - processed,
                  "FIFO processor made invalid progress: processed {} of {} remaining bytes", result.bytesProcessed,
                  target - processed);
    if (result.drawDone) {
      dispatch_draw_done();
    }
    processed += result.bytesProcessed;
    sProcess.processed.store(processed, std::memory_order_seq_cst);
    if (sProcess.waiters.load(std::memory_order_seq_cst) != 0) {
      bump(sWorker.notifies);
      sProcess.processed.notify_all();
    }
  }
}

void worker_main(std::stop_token token) noexcept {
  std::stop_callback wakeOnStop{token, wake_worker};
  while (true) {
    const uint32_t event = sWorkerWake.load(std::memory_order_acquire);
    const uint64_t processed = sProcess.processed.load(std::memory_order_relaxed);
    const uint64_t published = sPublish.published.load(std::memory_order_acquire);
    if (published != processed) {
      process_to(published);
      continue;
    }

    if (token.stop_requested()) {
      break;
    }
    // A sleeping worker costs the publisher a kernel wake, and frames publish every few microseconds
    if (spin_until([&] {
          return sPublish.published.load(std::memory_order_acquire) != processed || token.stop_requested();
        })) {
      continue;
    }
    run_park_hook(testing::ParkStep::BeforeFlag);
    sPublish.workerParked.store(true, std::memory_order_seq_cst);
    run_park_hook(testing::ParkStep::AfterFlag);
    if (sPublish.published.load(std::memory_order_seq_cst) == processed) {
      bump(sWorker.parks);
      sWorkerWake.wait(event, std::memory_order_acquire);
    }
    sPublish.workerParked.store(false, std::memory_order_seq_cst);
  }
}

void start_worker() {
  if (kProcessingMode != ProcessingMode::Thread || sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread = thread::Thread{{
                                     .name = "Aurora FIFO processor",
                                     .affinity = thread::Affinity::SharedCache,
                                 },
                                 worker_main};
}

void stop_worker() {
  if (!sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread.request_stop();
  sWorkerThread.join();
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void add_to_histogram(std::array<uint32_t, 2048>& histogram, uint64_t ns) noexcept {
  ++histogram[std::min<uint64_t>(ns / 1000, histogram.size() - 1)];
}

uint32_t histogram_p95(const std::array<uint32_t, 2048>& histogram) noexcept {
  uint64_t count = 0;
  for (const uint32_t bucket : histogram) {
    count += bucket;
  }
  if (count == 0) {
    return 0;
  }
  const uint64_t rank = (count * 95 + 99) / 100 - 1;
  uint64_t seen = 0;
  for (uint32_t us = 0; us < histogram.size(); ++us) {
    seen += histogram[us];
    if (seen > rank) {
      return us;
    }
  }
  return static_cast<uint32_t>(histogram.size() - 1);
}

void log_stats() noexcept {
  const FifoStats s = stats();
  if (s.frames == 0) {
    return;
  }
  const auto perFrame = [&s](uint64_t value) { return static_cast<double>(value) / static_cast<double>(s.frames); };
  const uint64_t drains = s.drainsReady + s.drainsSpun + s.drainsSlept;
  const double drainMeanUs = drains != 0 ? static_cast<double>(s.drainWaitNs) / static_cast<double>(drains) / 1e3 : 0.0;
  const uint64_t processNs = cpu_zone_stats().ns[CpuZoneProcess] - sGame.processNsBase;
  const std::string process =
      processNs != 0 ? fmt::format(", process {:.3f} ms/frame", perFrame(processNs) / 1e6) : std::string{};
  Log.info(
      "FIFO stats: {} frames; per frame publishes {:.1f}, wake calls {:.1f}, parks {:.1f}, chunks {:.1f}, "
      "notifies {:.1f}; drains ready {} spun {} slept {}; drain wait mean {:.1f} us p95 {} us max {:.1f} us; "
      "frame drain p95 {} us{}",
      s.frames, perFrame(s.publishes), perFrame(s.wakeCalls), perFrame(s.parks), perFrame(s.chunks),
      perFrame(s.notifies), s.drainsReady, s.drainsSpun, s.drainsSlept, drainMeanUs, histogram_p95(sDrainWaitUs),
      static_cast<double>(s.drainWaitMaxNs) / 1e3, histogram_p95(sFrameDrainUs), process);
}
} // namespace

ProcessingMode processing_mode() noexcept { return kProcessingMode; }

void init() {
  stop_worker();

  constexpr uint32_t initialCapacity = 64 * 1024;
  free(detail::sBufferData);
  detail::sBufferData = static_cast<uint8_t*>(malloc(initialCapacity));
  AURORA_ASSERT(detail::sBufferData != nullptr, "fifo::init: failed to allocate {} bytes", initialCapacity);
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;

  sPublish.frameActive = false;
  sPublish.pendingDraws = 0;
  sStreamBase = 0;
  sPublish.published.store(0, std::memory_order_relaxed);
  sPublish.workerParked.store(false, std::memory_order_relaxed);
  sProcess.processed.store(0, std::memory_order_relaxed);
  sProcess.waiters.store(0, std::memory_order_relaxed);
  sWorkerWake.store(0, std::memory_order_relaxed);
  for (auto* counter : {&sGame.frames, &sGame.publishes, &sGame.wakeCalls, &sGame.drainWaitNs, &sGame.drainWaitMaxNs,
                        &sWorker.parks, &sWorker.chunks, &sWorker.notifies}) {
    counter->store(0, std::memory_order_relaxed);
  }
  for (auto& drains : sGame.drains) {
    drains.store(0, std::memory_order_relaxed);
  }
  sGame.frameDrainNs = 0;
  // Zone totals run since startup, and the stats line covers only this session's frames
  sGame.processNsBase = cpu_zone_stats().ns[CpuZoneProcess];
  sDrainWaitUs.fill(0);
  sFrameDrainUs.fill(0);
  // Live runs have no replay flag, so an environment switch times process()
  if (const char* value = SDL_getenv("AURORA_CPU_ZONES"); value != nullptr && std::strcmp(value, "1") == 0) {
    set_cpu_zone_timing(CpuZoneTimingFifo);
  }

  reset_immutable_arrays();
  init_immutable_arrays();
  start_worker();
}

void shutdown() {
  stop_worker();
  log_stats();
}

void begin_frame() noexcept { sPublish.frameActive = true; }

void end_frame() noexcept {
  sPublish.frameActive = false;
  clear_draw_cache(); // command_processor
  bump(sGame.frames);
  add_to_histogram(sFrameDrainUs, sGame.frameDrainNs);
  sGame.frameDrainNs = 0;
}

void write_data_grow(const void* data, uint32_t length) {
  const uint64_t needed64 = static_cast<uint64_t>(detail::sBufferSize) + length;
  AURORA_ASSERT(needed64 <= std::numeric_limits<uint32_t>::max(), "fifo::write_data: buffer size overflow");
  const auto needed = static_cast<uint32_t>(needed64);
  const auto doubledCapacity = static_cast<uint64_t>(detail::sBufferCapacity) * 2;
  const auto newCapacity = static_cast<uint32_t>(
      std::min<uint64_t>(std::max(doubledCapacity, needed64), std::numeric_limits<uint32_t>::max()));
  const auto grow = [newCapacity] {
    auto* resized = static_cast<uint8_t*>(realloc(detail::sBufferData, newCapacity));
    AURORA_ASSERT(resized != nullptr, "fifo::write_data: failed to allocate {} bytes", newCapacity);
    detail::sBufferData = resized;
  };
  if (sWorkerThread.joinable()) {
    std::lock_guard lock{sBufferMutex};
    grow();
  } else {
    grow();
  }
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCapacity;
}

void publish() noexcept {
  if (!sPublish.frameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }

  const uint64_t target = sStreamBase + detail::sBufferSize;
  if (target > sPublish.published.load(std::memory_order_relaxed)) {
    sPublish.pendingDraws = 0;
    bump(sGame.publishes);
    if (kProcessingMode == ProcessingMode::Thread) {
      advance_published(target);
      if (sPublish.synchronousPublish)
        UNLIKELY { wait_for_processed(target); }
    } else {
      sPublish.published.store(target, std::memory_order_release);
      process_to(target);
    }
  }
}

DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept {
  return sDrawDoneCallback.exchange(callback, std::memory_order_acq_rel);
}

void set_process_observer(const ProcessObserver* observer) noexcept {
  sProcessObserver.store(observer, std::memory_order_release);
}

const ProcessObserver* process_observer() noexcept { return sProcessObserver.load(std::memory_order_acquire); }

void set_synchronous_publish(bool synchronous) noexcept { sPublish.synchronousPublish = synchronous; }

void finish_draw() noexcept {
  if (!sPublish.frameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }
  if (++sPublish.pendingDraws >= kDrawBatchSize) {
    publish();
  }
}

void patch_u32(uint32_t offset, uint32_t val) {
  AURORA_ASSERT(!detail::sInDisplayList && offset <= detail::sBufferSize &&
                    sizeof(uint32_t) <= detail::sBufferSize - offset,
                "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  AURORA_ASSERT(sStreamBase + offset >= sPublish.published.load(std::memory_order_relaxed),
                "fifo::patch_u32: offset {} is below the published watermark", offset);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  const uint32_t bytesWritten = detail::sDlWritePos;
  const uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (detail::sBufferSize == 0) {
    return;
  }

  ZoneScoped;
  const uint64_t target = sStreamBase + detail::sBufferSize;

  switch (kProcessingMode) {
  case ProcessingMode::Drain:
  case ProcessingMode::Inline:
    sPublish.published.store(target, std::memory_order_relaxed);
    process_to(target);
    break;
  case ProcessingMode::Thread: {
    if (target > sPublish.published.load(std::memory_order_relaxed)) {
      advance_published(target);
    }
    const auto start = std::chrono::steady_clock::now();
    const WaitOutcome outcome = wait_for_processed(target);
    const uint64_t waitNs = elapsed_ns(start);
    bump(sGame.drains[static_cast<size_t>(outcome)]);
    bump(sGame.drainWaitNs, waitNs);
    if (waitNs > sGame.drainWaitMaxNs.load(std::memory_order_relaxed)) {
      sGame.drainWaitMaxNs.store(waitNs, std::memory_order_relaxed);
    }
    sGame.frameDrainNs += waitNs;
    add_to_histogram(sDrainWaitUs, waitNs);
    break;
  }
  }

  {
    std::lock_guard lock{sBufferMutex};
    sStreamBase = target;
    detail::sBufferSize = 0;
  }
  sPublish.pendingDraws = 0;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }

void clear_buffer() {
  const uint64_t processed = sProcess.processed.load(std::memory_order_acquire);
  AURORA_ASSERT(sPublish.published.load(std::memory_order_acquire) == processed,
                "fifo::clear_buffer: published commands are still pending");
  std::lock_guard lock{sBufferMutex};
  sStreamBase = processed;
  detail::sBufferSize = 0;
  sPublish.pendingDraws = 0;
}

FifoStats stats() noexcept {
  return {
      .frames = sGame.frames.load(std::memory_order_relaxed),
      .publishes = sGame.publishes.load(std::memory_order_relaxed),
      .wakeCalls = sGame.wakeCalls.load(std::memory_order_relaxed),
      .parks = sWorker.parks.load(std::memory_order_relaxed),
      .chunks = sWorker.chunks.load(std::memory_order_relaxed),
      .notifies = sWorker.notifies.load(std::memory_order_relaxed),
      .drainsReady = sGame.drains[static_cast<size_t>(WaitOutcome::Ready)].load(std::memory_order_relaxed),
      .drainsSpun = sGame.drains[static_cast<size_t>(WaitOutcome::Spun)].load(std::memory_order_relaxed),
      .drainsSlept = sGame.drains[static_cast<size_t>(WaitOutcome::Slept)].load(std::memory_order_relaxed),
      .drainWaitNs = sGame.drainWaitNs.load(std::memory_order_relaxed),
      .drainWaitMaxNs = sGame.drainWaitMaxNs.load(std::memory_order_relaxed),
  };
}

namespace testing {
void set_park_hook(ParkHook hook) noexcept { sParkHook.store(hook, std::memory_order_release); }

bool worker_parked() noexcept { return sPublish.workerParked.load(std::memory_order_seq_cst); }

uint32_t process_waiters() noexcept { return sProcess.waiters.load(std::memory_order_seq_cst); }

uint32_t spin_before_sleep_ns() noexcept {
  return static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(kSpinBeforeSleep).count());
}
} // namespace testing

} // namespace aurora::gx::fifo
