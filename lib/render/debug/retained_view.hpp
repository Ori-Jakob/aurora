#pragma once

#include "views.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"
#include "../scene/streamout.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
class PassContext;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::debug::retained_view {
inline constexpr std::string_view Shader = "debug/retained_view";
inline constexpr std::string_view DepthResource = "debug.retained_depth";
inline constexpr std::string_view InstancesResource = "debug.retained_instances";
inline constexpr std::string_view PairsResource = "debug.retained_pairs";
inline constexpr uint32_t MaxViewInstances = 16384;
inline constexpr uint32_t MaxViewPairs = 131072;
inline constexpr uint32_t InstanceWords = 4;
inline constexpr uint32_t PairWords = 2;
inline constexpr uint32_t HeaderWords = 64;
inline constexpr uint64_t InstancesBytes = uint64_t{HeaderWords + MaxViewInstances * InstanceWords} * 4;
inline constexpr uint64_t PairsBytes = uint64_t{MaxViewPairs} * PairWords * 4;
inline constexpr uint32_t StorageBindings = 7;
inline constexpr float MinCoverageExtent = 1000.f;
inline constexpr float MinChosenExtent = 10.f;
inline constexpr float MaxCoverageExtent = 50000.f;

inline constexpr uint32_t PipelineEntries = 5;

namespace header {
inline constexpr uint32_t Proj = 0;
inline constexpr uint32_t View = 16;
inline constexpr uint32_t Mode = 28;
inline constexpr uint32_t Flags = 29;
inline constexpr uint32_t Instances = 30;
inline constexpr uint32_t Pairs = 31;
inline constexpr uint32_t Marker = 32;
inline constexpr uint32_t Target = 44;
inline constexpr uint32_t Ortho = 1u << 0;
} // namespace header

namespace instance_flag {
inline constexpr uint32_t Clustered = 1u << 0;
} // namespace instance_flag

struct ViewInstance {
  uint32_t slot = 0;
  uint32_t color = 0;
  uint32_t indexCount = 0;
  uint32_t flags = 0;
};
static_assert(sizeof(ViewInstance) == InstanceWords * 4);

struct Packed {
  std::vector<ViewInstance> instances;
  std::vector<std::array<uint32_t, PairWords>> pairs;
  uint32_t overInstances = 0;
  uint32_t overPairs = 0;
};

struct Shown {
  uint32_t instances = 0;
  uint32_t clusters = 0;
  uint32_t overInstances = 0;
  uint32_t overPairs = 0;
  float extent = 0.f;
};

struct Limits {
  uint32_t instances = MaxViewInstances;
  uint32_t pairs = MaxViewPairs;
};

[[nodiscard]] RetainedClass classify(const scene::streamout::LiveInstance& instance) noexcept;
[[nodiscard]] uint32_t pack_color(Rgb8 color, bool seen) noexcept;
[[nodiscard]] Packed pack(std::span<const scene::streamout::LiveInstance> live, std::span<const uint64_t> seen,
                          RetainedMode mode, const Limits& limits = {});
[[nodiscard]] float fit_extent(std::span<const scene::streamout::LiveInstance> live,
                               const std::array<double, 3>& eye) noexcept;

[[nodiscard]] bool supported(const Caps& caps) noexcept;
void initialize();
void shutdown() noexcept;
[[nodiscard]] bool declarable(const detail::PassSetup& setup) noexcept;
void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GbufView& row);
[[nodiscard]] bool enabled(const detail::StageData& stage) noexcept;
[[nodiscard]] shaders::RenderPipelineDesc pipeline_desc(uint32_t entry);
[[nodiscard]] std::span<const uint32_t> mode_entries(RetainedMode mode) noexcept;
void pipelines(const detail::PipelineSetup& setup, const GbufView& row,
               std::span<uint32_t, detail::MaxPassPipelines> ids);
void encode(detail::PassContext& ctx, const GbufView& row);

// Game thread; 0 fits the extent at each frame
void set_coverage_extent(float extent) noexcept;
[[nodiscard]] float coverage_extent() noexcept;
[[nodiscard]] Shown shown() noexcept;
} // namespace aurora::render::debug::retained_view
