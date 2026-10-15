#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace aurora::webgpu::gpu_prof {

inline constexpr uint32_t PassTimingKeepAliveFrames = 120;

struct PassTime {
  const char* name = nullptr;
  float ms = 0.f;
};

struct FrameTimes {
  uint64_t sequence = 0;
  float frameMs = 0.f;
  uint32_t dropped = 0;
  std::vector<PassTime> passes;
};

namespace detail {
enum class EventKind : uint8_t {
  ZoneBegin,
  PassBegin,
  End,
};

struct Event {
  const char* name; // static lifetime; nullptr for End
  uint32_t query;
  EventKind kind;
};

void aggregate(std::span<const Event> events, std::span<const uint64_t> timestamps, uint64_t frameBegin,
               uint64_t frameEnd, FrameTimes& out);
} // namespace detail

void initialize();
void shutdown();

void frame_begin(const wgpu::CommandEncoder& encoder);
void frame_end(const wgpu::CommandEncoder& encoder);
void after_submit();
const wgpu::PassTimestampWrites* pass_writes(std::string_view name);

bool supported() noexcept;
void request(uint32_t frames) noexcept;
void set_always(bool always) noexcept;
bool latest(FrameTimes& out);
uint64_t last_frame_sequence() noexcept;
bool wait_idle(uint64_t timeoutNs);

class Zone {
public:
  Zone(const wgpu::CommandEncoder& encoder, std::string_view name);
  ~Zone();
  Zone(const Zone&) = delete;
  Zone& operator=(const Zone&) = delete;

private:
  const wgpu::CommandEncoder* m_encoder = nullptr;
  uint32_t m_endQuery = 0;
};

namespace testing {
uint64_t resolve_count() noexcept;
bool resources_created() noexcept;
std::span<const detail::Event> recording_events() noexcept;
} // namespace testing

} // namespace aurora::webgpu::gpu_prof
