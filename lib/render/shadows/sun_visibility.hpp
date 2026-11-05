#pragma once

#include "vsm_filter.hpp"
#include "vsm_pages.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"
#include "../stage_camera.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::detail {
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::shadows::sun_visibility {
inline constexpr std::string_view Pass = "sun.visibility";
inline constexpr std::string_view Shader = "shadows/sun_visibility";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RG8Unorm;
inline constexpr uint32_t Order = 80;
inline constexpr uint32_t SampleStride = 8;
inline constexpr uint32_t CounterFirst = vsm::stat::SunUnshadowed;
inline constexpr uint32_t CounterWords = vsm::stat::SunSampledOutOfRange + 1 - CounterFirst;
inline constexpr uint32_t ReadbackSlots = 4;

namespace binding {
inline constexpr uint32_t Depth = 0;
inline constexpr uint32_t Normal = 1;
inline constexpr uint32_t Pool = 2;
inline constexpr uint32_t Table = 3;
inline constexpr uint32_t Views = 4;
inline constexpr uint32_t Stats = 5;
inline constexpr uint32_t Target = 6;
} // namespace binding

namespace flag {
inline constexpr uint32_t Producing = 1u << 0;
inline constexpr uint32_t Counters = 1u << 1;
} // namespace flag

struct Params {
  detail::StageCamera camera;
  detail::ViewRotation rotation;
  uint32_t flags = 0;
  uint32_t taps = 0;
  uint32_t frame = 0;
  float normalScale = 1.f;
  float depthBias = vsm::DefaultDepthBias;
  uint32_t reserved[3]{};
};
static_assert(sizeof(Params) == 128 && sizeof(Params) <= MaxPassParamsSize);

struct SunShadowState {
  bool producing = false;
  bool consumers = false;
  uint32_t lightId = 0;
  uint32_t lightFlags = 0;
  double elevationDeg = 0.0;
  uint32_t levels = 0;
  bool unshadowed = false;
  float unshadowedShare = 0.f;
  float fallbackShare = 0.f;
  float staleShare = 0.f;
  float outOfRangeShare = 0.f;
  uint64_t statsFrame = 0;
  vsm::Idle idle = vsm::Idle::Inactive;
};

struct Counters {
  uint64_t producing = 0;
  uint64_t clearOnly = 0;
  uint64_t readbacks = 0;
  uint64_t unshadowedFrames = 0;
  uint64_t samples = 0;
  uint64_t sampledUnshadowed = 0;
  uint64_t sampledFallback = 0;
  uint64_t sampledStale = 0;
  uint64_t sampledOutOfRange = 0;
  uint64_t firstUnshadowed = 0;
};

void initialize();
void shutdown() noexcept;
void on_latch(const Settings* settings) noexcept;
[[nodiscard]] bool frame_enabled(const detail::StageData& stage) noexcept;
[[nodiscard]] uint32_t light_index(const detail::StageData& stage) noexcept;
[[nodiscard]] SunShadowState state() noexcept;
[[nodiscard]] detail::PassDesc pass_desc();
[[nodiscard]] shaders::RenderPipelineDesc pipeline_desc();
[[nodiscard]] Params latched_params(const Settings* settings) noexcept;

namespace testing {
void reset() noexcept;
void set_light_index(std::optional<uint32_t> index) noexcept;
[[nodiscard]] Counters counters() noexcept;
} // namespace testing
} // namespace aurora::render::shadows::sun_visibility
