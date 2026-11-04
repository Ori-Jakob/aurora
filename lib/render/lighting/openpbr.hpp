#pragma once

#include "../graph.hpp"
#include "../graph_bind.hpp"
#include "../material/classes.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <bitset>
#include <cstdint>
#include <string_view>
#include <vector>

namespace aurora::render::detail::lighting::openpbr {
inline constexpr std::string_view Classes = "light.classes";
inline constexpr std::string_view Lut3d = "light.lut3d";
inline constexpr std::string_view Lut2d = "light.lut2d";
inline constexpr std::string_view Library = "aurora/openpbr";
inline constexpr std::string_view LayeredDefine = "OPENPBR_LAYERED";

inline constexpr uint32_t TableSize = 32;
inline constexpr uint32_t Lut2dLayers = 3;
inline constexpr uint32_t LayerAverages = 0;
inline constexpr uint32_t LayerMetal = 1;
inline constexpr uint32_t LayerLtc = 2;
inline constexpr wgpu::TextureFormat Lut3dFormat = wgpu::TextureFormat::RG16Float;
inline constexpr wgpu::TextureFormat Lut2dFormat = wgpu::TextureFormat::RGBA16Float;
inline constexpr uint32_t ClassRows = 256;
inline constexpr uint64_t ClassBytes = uint64_t{ClassRows} * sizeof(material::classes::GpuRow);
inline constexpr size_t Lut3dTexels = size_t{TableSize} * TableSize * TableSize;
inline constexpr size_t Lut2dTexels = size_t{TableSize} * TableSize * Lut2dLayers;
static_assert(ClassBytes == 32768);

inline constexpr double IorMax = 2.5;
inline constexpr double MinRoughness = 0.045;
inline constexpr double MinMuV = 1e-4;
inline constexpr double MinAlphaWithEnergyLoss = 0.04 * 0.04;

[[nodiscard]] TexDesc lut3d_desc() noexcept;
[[nodiscard]] TexDesc lut2d_desc() noexcept;

struct LtcEntry {
  float a = 0.f;
  float b = 0.f;
  float c = 0.f;
};

namespace data {
extern const std::array<uint16_t, Lut3dTexels> IdealDielectricEc;
extern const std::array<uint16_t, size_t{TableSize} * TableSize> IdealDielectricAverageEc;
extern const std::array<uint16_t, size_t{TableSize} * TableSize> IdealDielectricReflectionRatio;
extern const std::array<uint16_t, Lut3dTexels> OpaqueDielectricEc;
extern const std::array<uint16_t, size_t{TableSize} * TableSize> OpaqueDielectricAverageEc;
extern const std::array<uint16_t, size_t{TableSize} * TableSize> IdealMetalEc;
extern const std::array<uint16_t, TableSize> IdealMetalAverageEc;
extern const std::array<LtcEntry, size_t{TableSize} * TableSize> Ltc;
} // namespace data

[[nodiscard]] uint16_t to_half(float value) noexcept;
[[nodiscard]] float from_half(uint16_t bits) noexcept;

// f16 texel words exactly as uploaded: lut3d RG per texel, lut2d RGBA per texel, layer-major
struct Tables {
  std::vector<uint16_t> lut3d;
  std::vector<uint16_t> lut2d;
};
[[nodiscard]] Tables build_tables();
inline constexpr uint32_t DfgSamples = 256;
[[nodiscard]] std::array<double, 3> dfg(double muV, double alpha) noexcept;

// Index-space coordinates of the pinned tables, before the texel-centre remap
[[nodiscard]] double ior_index(double ior) noexcept;
[[nodiscard]] double alpha_index(double alpha) noexcept;
[[nodiscard]] double cos_index(double cosTheta) noexcept;

void initialize();
void shutdown() noexcept;
[[nodiscard]] bool resident() noexcept;
[[nodiscard]] PersistentHandle classes_handle() noexcept;
[[nodiscard]] PersistentHandle lut3d_handle() noexcept;
[[nodiscard]] PersistentHandle lut2d_handle() noexcept;
// Render worker; returns true when the rows were rewritten
bool refresh_classes(const wgpu::CommandEncoder& encoder);

using ClassSet = std::bitset<ClassRows>;
// Coat and fuzz are the lobes OPENPBR_LAYERED adds
[[nodiscard]] bool layered(const material::classes::GpuRow& row) noexcept;
[[nodiscard]] ClassSet layered_classes(const material::classes::GpuRows& rows) noexcept;
// Render worker, after refresh_classes; every class while the tables are not resident
[[nodiscard]] ClassSet uploaded_layered() noexcept;

// Double mirror of aurora/openpbr, filtering the same f16 values in texel space
using Vec3 = std::array<double, 3>;

struct Surface {
  Vec3 baseColor{1.0, 1.0, 1.0};
  double roughness = 0.85;
  double metalness = 0.0;
};

struct Prepared {
  Vec3 n{};
  Vec3 v{};
  double muV = 1.0;
  double alpha = 0.0;
  double eta = 1.5;
  Vec3 specTint{};
  Vec3 metalF0{};
  Vec3 metalTint{};
  double metalWeight = 0.0;
  Vec3 metalMs{};
  double ecmV = 0.0;
  double ecmAvg = 1.0;
  Vec3 diffuse{};
  Vec3 transmit{};
  double sigma = 0.0;
  double ecV = 1.0;
  double invEcAvg = 1.0;
  double wrap = 0.0;
  bool thin = false;
  double coatWeight = 0.0;
  double coatEta = 1.6;
  double coatAlpha = 0.0;
  Vec3 coatTint{1.0, 1.0, 1.0};
  Vec3 coatBase{1.0, 1.0, 1.0};
  double fuzzWeight = 0.0;
  Vec3 fuzzColour{};
  Vec3 fuzzLtc{};
  double fuzzBase = 1.0;
  Vec3 t{};
  Vec3 b{};
  Vec3 ambientDiffuse{};
  Vec3 ambientSpecular{};
};

[[nodiscard]] Prepared prepare(const Tables& tables, const material::classes::GpuRow& row, const Surface& surface,
                               const Vec3& n, const Vec3& v);
[[nodiscard]] Vec3 eval_light(const Tables& tables, const Prepared& p, const Vec3& l, bool specular = true);
[[nodiscard]] Vec3 eval_ambient(const Prepared& p, const Vec3& irradiance, const Vec3& radiance);
[[nodiscard]] Vec3 directional_albedo(const Prepared& p);
// Emission sits under the coat and fuzz, so the deferred mirror dims it by what they let through
[[nodiscard]] Vec3 emission_scale(const Prepared& p);

// Filtered lookups in texel space; z is the IOR index for lut3d, layer selects lut2d's slice
[[nodiscard]] std::array<double, 2> sample_lut3d(const Tables& tables, double x, double y, double z);
[[nodiscard]] std::array<double, 4> sample_lut2d(const Tables& tables, double x, double y, uint32_t layer);

namespace testing {
[[nodiscard]] const Tables& tables();
void drop_resident() noexcept;
[[nodiscard]] uint64_t uploaded_epoch() noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting::openpbr
