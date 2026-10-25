#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aurora::webgpu::gpu_prof {
struct FrameTimes;
} // namespace aurora::webgpu::gpu_prof

namespace aurora::render::debug {
struct InspectRead {
  std::string resource;
  std::string access;
  std::string missing;
  std::string binding;
};

struct InspectWrite {
  std::string resource;
  std::string access;
};

struct InspectPass {
  std::optional<uint32_t> frameSlot;
  std::string name;
  std::string stage;
  std::string nodeKind;
  std::optional<std::string> kind;
  std::optional<std::string> cls;
  std::optional<uint32_t> order;
  std::optional<uint32_t> registration;
  bool active = true;
  std::optional<std::string> cull;
  std::optional<std::string> detail;
  bool sideEffect = false;
  bool external = false;
  std::vector<InspectRead> reads;
  std::vector<InspectWrite> writes;
};

struct InspectResource {
  std::string name;
  std::string kind;
  std::string type;
  std::string format;
  std::string size;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depthOrLayers = 1;
  uint32_t mips = 1;
  uint32_t samples = 1;
  std::vector<std::string> usage;
  uint64_t bytes = 0;
  std::optional<uint32_t> slot;
  std::optional<uint32_t> firstPass;
  std::optional<uint32_t> lastPass;
};

struct InspectEdge {
  uint32_t from = 0;
  uint32_t to = 0;
  std::string kind;
  std::optional<std::string> resource;
};

struct InspectStage {
  std::string name;
  uint32_t firstPass = 0;
  uint32_t passCount = 0;
};

struct GraphInspection {
  uint64_t graphHash = 0;
  std::string sceneDepth = "unavailable";
  std::vector<InspectStage> stages;
  std::vector<InspectPass> passes;
  std::vector<InspectResource> resources;
  std::vector<InspectEdge> edges;
  uint32_t poolSlots = 0;
  uint64_t logicalBytes = 0;
  uint64_t slotBytes = 0;
  std::optional<uint64_t> claimedBytes;
  std::optional<uint64_t> idleBytes;
};

struct FrameCullSnapshot {
  uint64_t frameId = 0;
  uint64_t graphHash = 0;
  std::vector<std::optional<std::string>> reasons;
};

std::string format_graph_json(const GraphInspection& graph, const FrameCullSnapshot* frame = nullptr,
                              const webgpu::gpu_prof::FrameTimes* timings = nullptr);
std::string format_graph_dot(const GraphInspection& graph, const FrameCullSnapshot* frame = nullptr);
} // namespace aurora::render::debug
