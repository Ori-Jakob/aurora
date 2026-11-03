#pragma once

#include <aurora/render.hpp>
#include <dolphin/gx/GXEnum.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace aurora::render::material {
struct Snapshot;
} // namespace aurora::render::material

namespace aurora::render::material::classes {
inline constexpr uint8_t None = 0;
inline constexpr uint8_t Default = 1;
inline constexpr uint8_t FirstAurora = 9;
inline constexpr uint8_t FirstGame = 128;
inline constexpr uint8_t Reserved = 255;
inline constexpr size_t MaxNameLength = 64;
inline constexpr std::string_view FakeShadowName = "fake_shadow";

struct BuiltIn {
  uint8_t id = 0;
  std::string_view name;
  OpenPbrParams params;
};
[[nodiscard]] std::span<const BuiltIn> built_ins() noexcept;

struct GpuRow {
  std::array<float, 32> v{};

  bool operator==(const GpuRow&) const = default;
};
static_assert(sizeof(GpuRow) == 128);
using GpuRows = std::array<GpuRow, 256>;
inline constexpr uint32_t RowThin = 1;
inline constexpr uint32_t RowTwoSided = 2;
inline constexpr uint32_t RowNoSsr = 4;

enum class Source : uint8_t { Default, Manifest, Hint, Force };

enum Field : uint8_t {
  FieldClass = 1u << 0,
  FieldRoughness = 1u << 1,
  FieldMetalness = 1u << 2,
};
inline constexpr size_t FieldCount = 3;

struct Stats {
  uint32_t builtIn = 0;
  uint32_t registered = 0;
  uint32_t rejected = 0;
  uint32_t unknownNames = 0;
  uint32_t unregisteredHints = 0;
};

[[nodiscard]] GpuRow gpu_row(const OpenPbrParams& params) noexcept;
[[nodiscard]] OpenPbrParams sanitised(const OpenPbrParams& params) noexcept;

// Game thread
bool register_class(uint8_t id, std::string_view name, const OpenPbrParams& params) noexcept;
void set_force(std::string_view name) noexcept;
void apply_frame_state(const Snapshot& snapshot) noexcept;
[[nodiscard]] uint64_t epoch() noexcept;
[[nodiscard]] std::shared_ptr<const GpuRows> gpu_rows() noexcept;
[[nodiscard]] Stats stats() noexcept;
void shutdown() noexcept;

// Latched table: game thread while the FIFO is idle, or the FIFO
[[nodiscard]] uint8_t resolve(std::string_view name) noexcept;
[[nodiscard]] bool present(uint8_t id) noexcept;
[[nodiscard]] const OpenPbrParams& params(uint8_t id) noexcept;
[[nodiscard]] uint8_t supplied(uint32_t materialId) noexcept;
[[nodiscard]] uint8_t manifest_class(uint32_t materialId) noexcept;

// FIFO thread
[[nodiscard]] uint32_t material_word(const uint16_t idPerField[FieldCount], uint8_t hint, uint32_t categories,
                                     GXCullMode cull, bool doubleSided, Source& source) noexcept;
[[nodiscard]] constexpr uint32_t default_word(bool ambient) noexcept { return ambient ? 0x800100D9u : 0x000100D9u; }

namespace testing {
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::material::classes
