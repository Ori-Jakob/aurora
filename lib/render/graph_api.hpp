#pragma once

#include "graph.hpp"

#include <aurora/render_graph.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
inline constexpr size_t MaxExternalPasses = 64;

struct ExternalPassRecord {
  std::string name;
  std::string prefix;
  ExternalPassDesc desc;
  PassId id = InvalidPass;
  uint64_t sequence = 0;
  std::array<std::byte, render::MaxPassParamsSize> pending{};
};

struct ExternalNodeState {
  std::string name;
  std::string prefix;
  ExternalPassDesc desc;
  PassId id = InvalidPass;
  uint64_t sequence = 0;
};

struct ExternalPassSet {
  uint64_t epoch = 0;
  std::vector<ExternalNodeState> nodes;
};

struct LatchedExternalPass {
  PassId id = InvalidPass;
  PassEnabledCallback enabledFn = nullptr;
  void* userdata = nullptr;
  uint32_t paramsSize = 0;
  bool enabled = false;
  std::array<std::byte, render::MaxPassParamsSize> params{};
};

bool valid_external_name(std::string_view name) noexcept;
std::string_view external_prefix(std::string_view name) noexcept;
bool reserved_prefix(std::string_view prefix) noexcept;

uint64_t external_pass_epoch() noexcept;
std::shared_ptr<const ExternalPassSet> snapshot_external_passes();
void append_external_passes(const ExternalPassSet& set, PassEncodeFn encode, std::vector<PassDesc>& out);
std::span<const LatchedExternalPass> latch_external_passes();
const LatchedExternalPass* latched_external_pass(PassId id) noexcept;
bool external_pass_alive(PassId id) noexcept;
bool external_enabled(const StageData& stage, const void* state);
std::string untraceable_external_pass();
bool log_once(std::string_view pass, std::string_view reason, std::string_view detail);
void shutdown_external_passes() noexcept;

namespace testing {
void reset_external_passes() noexcept;
} // namespace testing
} // namespace aurora::render::detail
