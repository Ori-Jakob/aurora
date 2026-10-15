#pragma once

#include <aurora/render.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aurora::render::detail {
enum class PassKind : uint8_t { Compute, Render, Copy, Encoder, Native };
enum class PassClass : uint8_t { Required, Optional, Diagnostic };

struct StageData;
class PassContext;
using PassEnabledFn = bool (*)(const StageData& stage, const void* state);
using PassEncodeFn = void (*)(PassContext& ctx, const void* state);

struct PassDesc {
  std::string name;
  Stage stage = Stage::SceneBegin;
  PassKind kind = PassKind::Encoder;
  PassClass cls = PassClass::Required;
  uint16_t order = 0;
  bool runsWithoutMarker = false;
  PassEnabledFn enabled = nullptr;
  PassEncodeFn encode = nullptr;
  const void* state = nullptr;
};

using PassId = uint64_t;
inline constexpr PassId InvalidPassId = 0;

PassId register_pass(const PassDesc& desc);
void unregister_pass(PassId id) noexcept;

struct CompiledPass {
  PassDesc desc;
  PassId id = InvalidPassId;
  uint64_t sequence = 0;
};

struct StageRange {
  uint32_t first = 0;
  uint32_t count = 0;
};

struct CompiledGraph {
  uint64_t graphHash = 0;
  std::vector<CompiledPass> passes;
  std::array<StageRange, StageCount> stages{};
  bool hasNativePasses = false;

  [[nodiscard]] bool empty() const noexcept { return passes.empty(); }
  [[nodiscard]] const StageRange& range(Stage stage) const noexcept { return stages[static_cast<size_t>(stage)]; }
};

inline const std::shared_ptr<const CompiledGraph>& empty_graph() noexcept {
  static const std::shared_ptr<const CompiledGraph> graph = std::make_shared<const CompiledGraph>();
  return graph;
}

std::shared_ptr<const CompiledGraph> compile_graph(const Settings& settings, const Caps& caps);
bool consume_graph_dirty() noexcept;

namespace testing {
void clear_passes() noexcept;
} // namespace testing
} // namespace aurora::render::detail
