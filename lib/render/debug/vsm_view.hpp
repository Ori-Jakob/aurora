#pragma once

#include "views.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"
#include "../stage_camera.hpp"
#include "../shadows/vsm_filter.hpp"
#include "../shadows/vsm_pages.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::detail {
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::debug::vsm_view {
inline constexpr std::string_view Pass = "vsm.view";
inline constexpr std::string_view Shader = "debug/vsm_view";
inline constexpr uint16_t Order = 90;
inline constexpr uint32_t Workgroup = 8;
inline constexpr uint32_t StorageBuffers = 5;
inline constexpr uint32_t ReadbackSlots = 3;

namespace binding {
inline constexpr uint32_t Depth = 0;
inline constexpr uint32_t Normal = 1;
inline constexpr uint32_t Pool = 2;
inline constexpr uint32_t Table = 3;
inline constexpr uint32_t Phys = 4;
inline constexpr uint32_t Requests = 5;
inline constexpr uint32_t RenderList = 6;
inline constexpr uint32_t Views = 7;
inline constexpr uint32_t Stats = 8;
inline constexpr uint32_t Target = 9;
} // namespace binding

namespace flag {
inline constexpr uint32_t Producing = 1u << 0;
} // namespace flag

struct Params {
  detail::StageCamera camera;
  detail::ViewRotation rotation;
  uint32_t flags = 0;
  uint32_t taps = shadows::vsm::MaxFilterTaps;
  float normalScale = 1.f;
  float depthBias = shadows::vsm::DefaultDepthBias;
  uint32_t reserved[4]{};
};
static_assert(sizeof(Params) == 128 && sizeof(Params) <= MaxPassParamsSize);

struct Shown {
  bool valid = false;
  uint64_t frame = 0;
  std::array<uint32_t, shadows::vsm::StatCount> stats{};
};

struct Counters {
  uint64_t producing = 0;
  uint64_t noSun = 0;
};

// Rows 49-52 map to the sub-modes in id order
[[nodiscard]] std::optional<VsmViewMode> row_mode(const GbufView& row) noexcept;
[[nodiscard]] std::optional<VsmViewMode> selected_mode(uint16_t view) noexcept;
[[nodiscard]] std::string reason_text(const shadows::vsm::SunPlan& plan, bool consumed, bool keyRoomOnly);

void initialize();
void shutdown() noexcept;
void on_latch(const Settings* settings) noexcept;
[[nodiscard]] detail::PassDesc pass_desc();
[[nodiscard]] shaders::ComputePipelineDesc pipeline_desc(VsmViewMode mode);
[[nodiscard]] Shown shown() noexcept;
void draw_stats();

namespace testing {
void reset() noexcept;
[[nodiscard]] Counters counters() noexcept;
} // namespace testing
} // namespace aurora::render::debug::vsm_view
