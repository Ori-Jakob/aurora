#pragma once

#include "../json.hpp"

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aurora::render::material {
enum SummaryBit : uint32_t {
  Stochastic = 1u << 0,
  NoPbr = 1u << 1,
  Cutout = 1u << 2,
  Unlit = 1u << 3,
  FakeShadow = 1u << 8,
  Grass = 1u << 9,
  Displacement = 1u << 10,
  ShadowScope = 1u << 11,
  StochasticParams = 1u << 12,
  Class = 1u << 13,
};

inline constexpr uint32_t MapRoleCount = 7;
inline constexpr std::array<std::string_view, MapRoleCount> MapRoleNames{"albedo", "delight",   "nrm", "orm",
                                                                         "emit",   "grassmask", "mat"};
inline constexpr uint32_t MapRoleShift = 16;
inline constexpr uint32_t VertexColorShift = 24;
inline constexpr uint32_t NeutralShift = 26;
inline constexpr uint32_t GateShift = 28;

constexpr uint32_t map_role_bit(uint32_t role) noexcept { return 1u << (MapRoleShift + role); }
constexpr bool has_map_role(uint32_t summary, uint32_t role) noexcept { return (summary & map_role_bit(role)) != 0; }
constexpr uint32_t vertex_color(uint32_t summary) noexcept { return summary >> VertexColorShift & 3u; }
constexpr uint32_t shadow_neutral(uint32_t summary) noexcept { return summary >> NeutralShift & 3u; }
constexpr uint32_t shadow_gate(uint32_t summary) noexcept { return summary >> GateShift & 3u; }

enum StochasticField : uint8_t {
  StochasticContrast = 1u << 0,
  StochasticRotation = 1u << 1,
  StochasticScale = 1u << 2,
  StochasticExponent = 1u << 3,
};

struct MaterialRecord {
  uint32_t summary = 0;
  uint16_t id = 0;
  uint16_t classId = 0;
  uint16_t shadowScope = 0;
  uint8_t stochasticSet = 0;
  uint8_t reserved0 = 0;
  float contrast = 0.f;
  float rotation = 0.f;
  float scale = 0.f;
  float exponent = 0.f;
  uint32_t reserved1 = 0;
};
static_assert(sizeof(MaterialRecord) == 32);
static_assert(std::is_trivially_copyable_v<MaterialRecord>);

struct KeyTable {
  std::vector<texture::TextureSourceKey> keys;
  std::vector<std::string> names;
};

struct Contributor {
  uint64_t registration = 0;
  std::string label;
  std::string base;
  texture::MaterialLayer layer = texture::MaterialLayer::Pack;
  int32_t priority = 0;
  uint64_t sequence = 0;
  std::shared_ptr<const json::Doc> doc;
};

struct KeyIndex;

struct Snapshot {
  uint64_t generation = 0;
  uint64_t bindEpoch = 0;
  std::shared_ptr<const KeyTable> keyTable;
  std::vector<MaterialRecord> records;
  std::vector<yyjson_val*> entries;
  std::vector<std::array<uint32_t, MapRoleCount>> mapSupplier;
  std::vector<std::string> classes;
  std::shared_ptr<const json::Doc> merged;
  std::string mergedText;
  std::vector<Contributor> contributors;
  std::array<uint32_t, 32> summaryCount{};
  uint32_t liveCount = 0;
  uint32_t materialCount = 0;
  uint32_t grassTypeCount = 0;
  std::shared_ptr<const KeyIndex> index;

  uint16_t find(const texture::TextureSourceKey& key) const noexcept;
  const MaterialRecord& record(uint32_t id) const noexcept;
  bool live(uint32_t id) const noexcept;
  const texture::TextureSourceKey* key(uint32_t id) const noexcept;
  std::string_view key_name(uint32_t id) const noexcept;
  std::string_view map_base(uint32_t id, uint32_t role) const noexcept;
};

std::shared_ptr<const Snapshot> snapshot() noexcept;
uint64_t generation() noexcept;
using PublishObserver = void (*)(const std::shared_ptr<const Snapshot>&) noexcept;
void set_publish_observer(PublishObserver observer) noexcept;
bool latch() noexcept;
std::shared_ptr<const Snapshot> latched_snapshot() noexcept;
const Snapshot* latched_raw() noexcept;

namespace testing {
void set_check_hook(void (*hook)(uint64_t)) noexcept;
} // namespace testing
} // namespace aurora::render::material
