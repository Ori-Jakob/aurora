#pragma once

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>
#include <aurora/render_caps.hpp>
#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace aurora::render::compact {
inline constexpr uint32_t WorkgroupSize = 128;
inline constexpr uint32_t ItemsPerThread = 8;
inline constexpr uint32_t TileSize = WorkgroupSize * ItemsPerThread;
inline constexpr uint32_t MaxCapacity = 65535u * TileSize;
inline constexpr uint32_t ImmediateSize = 64;
inline constexpr uint32_t StorageBindings = 5;
inline constexpr uint32_t HeaderWords = 16;
inline constexpr uint32_t TotalWord = 0;
inline constexpr uint32_t LiveWord = 1;
inline constexpr uint32_t MaxCachedBindings = 256;

enum class Op : uint8_t { Scan, Compact };
enum class ArgsKind : uint8_t { None, Dispatch, Draw, DrawIndexed };
enum class Reject : uint8_t {
  None,
  Unsupported,
  Device,
  Pipeline,
  Plan,
  KeepMask,
  Args,
  Usage,
  Alignment,
  Range,
  Size,
  Overlap
};

struct Plan {
  Op op = Op::Scan;
  uint32_t capacity = 0;
  uint8_t levels = 0;
  uint8_t dispatches = 0;
  uint64_t scratchBytes = 0;
  [[nodiscard]] bool valid() const noexcept { return levels != 0; }
  bool operator==(const Plan&) const = default;
};

struct ArgsSpec {
  ArgsKind kind = ArgsKind::None;
  uint32_t offset = 0;
  uint8_t field = 0;
  uint32_t scale = 1;
  uint32_t limit = 0;
  std::array<uint32_t, 5> record{};
};

class Binding {
public:
  [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(m_group); }

private:
  friend struct BindingAccess;
  wgpu::BindGroup m_group;
  std::array<uint64_t, 3 * StorageBindings> m_key{};
};

struct Job {
  Plan plan;
  BufferView input, output, scratch, count;
  uint32_t countWord = 0;
  uint32_t keepMask = UINT32_MAX;
  ArgsSpec args;
  BufferView argsTarget;
  const Binding* binding = nullptr;
};

struct GraphJob {
  Op op = Op::Scan;
  uint32_t capacity = 0;
  std::string_view input, output, scratch, count;
  uint32_t countWord = 0;
  uint32_t keepMask = UINT32_MAX;
  std::string_view args;
  ArgsSpec argsSpec;
};

struct Stats {
  uint64_t jobs = 0, passes = 0, dispatches = 0, rejected = 0, backstops = 0, bindGroupsCreated = 0,
           bindGroupsLastFrame = 0;
};

[[nodiscard]] bool supported(const Caps& caps) noexcept;
[[nodiscard]] uint32_t max_capacity(const Caps& caps) noexcept;
[[nodiscard]] Plan make_plan(Op op, uint32_t capacity, const Caps& caps) noexcept;
[[nodiscard]] uint32_t args_record_words(ArgsKind kind) noexcept;
[[nodiscard]] Reject validate(const Job& job, const Caps& caps) noexcept;
[[nodiscard]] std::string_view reject_name(Reject reason) noexcept;
void initialize();
void shutdown();
void worker_frame_begin(uint64_t frameId);
shaders::PipelineId pipeline();
[[nodiscard]] bool ready() noexcept;
void pass_pipelines(const detail::PipelineSetup& setup, const void* state,
                    std::span<uint32_t, detail::MaxPassPipelines> ids);
bool declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GraphJob& job);
void encode_pass(detail::PassContext& ctx, const void* state);
[[nodiscard]] detail::PassDesc graph_pass(std::string name, Stage stage, detail::PassDeclareFn declare,
                                          const void* state);
[[nodiscard]] Binding make_binding(const Job& job);
bool encode(const wgpu::CommandEncoder& encoder, std::span<const Job> jobs, std::string_view label);
bool encode(const wgpu::CommandEncoder& encoder, const Job& job, std::string_view label);
bool encode(const detail::PassContext& ctx, std::span<const Job> jobs);
bool encode(const wgpu::ComputePassEncoder& pass, std::span<const Job> jobs);
[[nodiscard]] Stats stats() noexcept;

namespace testing {
struct Options {
  bool releaseRejections = false;
  bool skipSizeChecks = false;
};
void set_options(const Options& options);
void force_pipeline(shaders::PipelineId id);
} // namespace testing
} // namespace aurora::render::compact
