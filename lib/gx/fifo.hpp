#pragma once

#include "../internal.hpp"

#include <cstring>

namespace aurora::gx::fifo {

namespace detail {
extern uint8_t* sBufferData;
extern uint32_t sBufferSize;
extern uint32_t sBufferCapacity;
extern bool sInDisplayList;
extern uint8_t* sDlBuffer;
extern uint32_t sDlSize;
extern uint32_t sDlWritePos;
} // namespace detail

enum class ProcessingMode : uint8_t {
  // Process FIFO synchronously at drain()
  Drain,
  // Process FIFO synchronously at publish() (useful for profiling)
  Inline,
  // Process FIFO asynchronously on a worker thread; drain() synchronizes
  Thread
};
ProcessingMode processing_mode() noexcept;

void init();
void shutdown();

void begin_frame() noexcept;
void end_frame() noexcept;

// Out-of-line slow path: grows internal buffer then appends data
void write_data_grow(const void* data, uint32_t length);

inline void write_data(const void* data, const uint32_t length) {
  if (!detail::sInDisplayList)
    LIKELY {
      if (length <= detail::sBufferCapacity - detail::sBufferSize)
        LIKELY {
          std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
          detail::sBufferSize += length;
          return;
        }
      write_data_grow(data, length);
    }
  else if (length <= detail::sDlSize - detail::sDlWritePos) {
    std::memcpy(detail::sDlBuffer + detail::sDlWritePos, data, length);
    detail::sDlWritePos += length;
  }
}

inline void write_u8(const uint8_t val) {
  if (!detail::sInDisplayList)
    LIKELY {
      if (detail::sBufferSize < detail::sBufferCapacity)
        LIKELY {
          detail::sBufferData[detail::sBufferSize++] = val;
          return;
        }
      write_data_grow(&val, 1);
    }
  else if (detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = val;
  }
}

inline void write_u16(const uint16_t val) {
  const auto out = bswap(val);
  write_data(&out, sizeof(out));
}

inline void write_u32(const uint32_t val) {
  const auto out = bswap(val);
  write_data(&out, sizeof(out));
}

inline void write_u64(const uint64_t val) {
  const auto out = bswap(val);
  write_data(&out, sizeof(out));
}

inline void write_f32(const float val) {
  const auto out = bswap(val);
  write_data(&out, sizeof(out));
}

// Overwrites an unpublished u32 previously written at the given offset.
void patch_u32(uint32_t offset, uint32_t val);

// Marks a complete draw and publishes when the configured draw batch is full.
void finish_draw() noexcept;

// Makes commands written so far available to the FIFO processor.
void publish() noexcept;

using DrawDoneCallback = void (*)();
DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept;

struct ProcessObserver {
  void (*processed)(void* userData, const uint8_t* data, uint32_t size) noexcept = nullptr;
  void* userData = nullptr;
};
// Swapped only while the FIFO is drained, like guest_memory::set_hooks.
void set_process_observer(const ProcessObserver* observer) noexcept;
const ProcessObserver* process_observer() noexcept;
// Trace capture needs guest memory frozen while published commands decode.
void set_synchronous_publish(bool synchronous) noexcept;

// Display list recording
void begin_display_list(uint8_t* buf, uint32_t size);
uint32_t end_display_list();
bool in_display_list();

// Ensure all buffered commands have been processed.
void drain();

// Internal buffer inspection
const uint8_t* get_buffer_data();
uint32_t get_buffer_size();
void clear_buffer();

struct FifoStats {
  uint64_t frames = 0, publishes = 0, wakeCalls = 0, parks = 0, chunks = 0, notifies = 0;
  uint64_t drainsReady = 0, drainsSpun = 0, drainsSlept = 0, drainWaitNs = 0, drainWaitMaxNs = 0;
};
FifoStats stats() noexcept;

namespace testing {
enum class ParkStep : uint8_t { BeforeFlag, AfterFlag };
using ParkHook = void (*)(ParkStep step);
void set_park_hook(ParkHook hook) noexcept;
bool worker_parked() noexcept;
uint32_t process_waiters() noexcept;
uint32_t spin_before_sleep_ns() noexcept;
} // namespace testing

} // namespace aurora::gx::fifo
