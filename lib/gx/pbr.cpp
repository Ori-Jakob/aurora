#include "pbr.hpp"

#include "../gfx/dds_io.hpp"
#include "../gfx/texture_replacement.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static aurora::Module PbrLog("aurora::gx::pbr");

namespace aurora::gx {
using webgpu::g_device;
using webgpu::g_queue;

bool enablePbrMaterialOverride = false;
bool pbrMaterialOverrideActive = false;
Vec4<float> pbrParams{0.30f, 0.04f, 0.20f, 0.0f};
Vec4<float> pbrScales{1.0f, 1.0f, static_cast<float>(AURORA_PBR_DEBUG_OFF), 0.0f};
Vec4<float> pbrNormalParams{1.0f, 1.0f, 1.0f, 0.0f};
Vec4<float> pbrAmbientGradient{1.15f, 0.45f, 0.80f, 1.0f};
Vec4<float> pbrIblParams{1.0f, 1.0f, 1.0f, 5.0f};
Vec4<float> pbrFillDir{0.39f, -0.44f, -0.81f, 0.0f};
Vec4<float> pbrMaterialFactors{0.5f, 0.0f, 1.0f, 0.5f};
Vec4<float> pbrMaterialEmissive{0.0f, 0.0f, 0.0f, 0.0f};
gfx::TextureHandle pbrMaterialRmaos;
gfx::TextureHandle pbrMaterialRoughness;
gfx::TextureHandle pbrMaterialMetallic;
gfx::TextureHandle pbrMaterialAo;
gfx::TextureHandle pbrMaterialSpecular;
gfx::TextureHandle pbrMaterialNormal;
gfx::TextureHandle pbrMaterialEmissiveMap;

namespace pbr_internal {
constexpr u32 PbrIrradianceCubeSize = 16;
constexpr u32 PbrPrefilterCubeSize = 64;
constexpr u32 PbrPrefilterMipCount = 6;
constexpr u32 PbrRuntimeProbeCubeSize = 32;
constexpr u32 PbrRuntimeProbePrefilterMipCount = 6;
constexpr u32 PbrProbeCaptureFramesPerFace = 6;
constexpr u32 PbrProbePeriodicRefreshFrames = 180;
constexpr float PbrProbeDirtyDistance = 48.0f;
constexpr float PbrProbeDirtyRotationSum = 0.08f;
constexpr u32 PbrBrdfLutSize = 128;
constexpr float PbrPi = 3.14159265358979323846f;
constexpr std::array<std::string_view, 6> PbrCubeFaceNames{"px", "nx", "py", "ny", "pz", "nz"};
constexpr u32 PbrProbeIrradianceFilterPassCount = 6;
constexpr u32 PbrProbePrefilterPassCount = 6 * PbrRuntimeProbePrefilterMipCount;
constexpr u32 PbrProbeFilterPassCount = PbrProbeIrradianceFilterPassCount + PbrProbePrefilterPassCount;

wgpu::Sampler sPbrMaterialSampler;
wgpu::Sampler sPbrIblSampler;
wgpu::Sampler sPbrBrdfLutSampler;

struct PbrIblTextureSet {
  wgpu::Texture irradianceCubeTexture;
  wgpu::TextureView irradianceCubeView;
  wgpu::Texture prefilterCubeTexture;
  wgpu::TextureView prefilterCubeView;
  wgpu::Texture brdfLutTexture;
  wgpu::TextureView brdfLutView;
  u32 prefilterMipCount = 1;
  bool available = false;
};

PbrIblTextureSet sFallbackPbrIbl;
PbrIblTextureSet sAuthoredPbrIbl;
PbrIblTextureSet sProbePbrIbl;
PbrIblTextureSet sProbeCapturePbrIbl;
PbrIblTextureSet* sActivePbrIbl = &sFallbackPbrIbl;
AuroraPbrIblSource sRequestedPbrIblSource = AURORA_PBR_IBL_SOURCE_PROBE;
std::string sActivePbrIblSceneKey;
std::string sAuthoredPbrIblRootPath;
std::string sAuthoredPbrIblStage;
int sAuthoredPbrIblRoom = -1;
bool sAuthoredPbrIblGlobalLoaded = false;
bool sAuthoredPbrIblStageLoaded = false;
bool sAuthoredPbrIblRoomLoaded = false;
std::array<wgpu::TextureView, 6> sProbeIrradianceFaceViews;
std::array<std::array<wgpu::TextureView, 6>, PbrRuntimeProbePrefilterMipCount> sProbePrefilterFaceViews;
std::array<wgpu::TextureView, 6> sProbeCaptureFaceViews;
wgpu::Texture sProbeCaptureMsaaColorTexture;
wgpu::TextureView sProbeCaptureMsaaColorView;
wgpu::Texture sProbeCaptureDepthTexture;
wgpu::TextureView sProbeCaptureDepthView;
wgpu::BindGroupLayout sProbeFilterBindGroupLayout;
wgpu::RenderPipeline sProbeFilterPipeline;
std::array<wgpu::Buffer, PbrProbeFilterPassCount> sProbeFilterUniforms;
std::array<wgpu::BindGroup, PbrProbeFilterPassCount> sProbeFilterBindGroups;
wgpu::TextureFormat sProbeCaptureColorFormat = wgpu::TextureFormat::Undefined;
wgpu::TextureFormat sProbeCaptureDepthFormat = wgpu::TextureFormat::Undefined;
u32 sProbeCaptureMsaaSamples = 0;
u32 sProbeCaptureFace = 0;
u32 sProbeCaptureDelayFrames = 0;
bool sProbeCaptureResourcesReady = false;
bool sProbeFilterPending = false;

struct PbrProbeCamera {
  Mat3x4<float> sourceView;
  Mat3x4<float> sourceInvView;
  Mat4x4<float> projection;
  std::array<Mat3x4<float>, 6> viewFromSource;
  std::array<Mat3x4<float>, 6> normalFromSource;
  bool valid = false;
};

PbrProbeCamera sProbeCamera;
PbrProbeCamera sProbeCaptureCamera;
PbrProbeCamera sProbeLastCompletedCamera;
u32 sProbeFramesSinceRefresh = PbrProbePeriodicRefreshFrames;
bool sProbeRefreshPending = true;
bool sProbeAutoRefresh = false;

struct PbrProbeFilterParams {
  u32 face = 0;
  u32 mode = 0;
  u32 mip = 0;
  u32 sampleCount = 0;
  float roughness = 0.0f;
  float sourceMip = 0.0f;
  float padding0 = 0.0f;
  float padding1 = 0.0f;
};

struct PbrVec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

float pbr_saturate(float v) noexcept { return std::clamp(v, 0.0f, 1.0f); }

float pbr_dot(PbrVec3 a, PbrVec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }

PbrVec3 pbr_normalize(PbrVec3 v) noexcept {
  const float len = std::sqrt(std::max(pbr_dot(v, v), 1e-12f));
  return {v.x / len, v.y / len, v.z / len};
}

PbrVec3 pbr_cross(PbrVec3 a, PbrVec3 b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float pbr_mtx(const Mat3x4<float>& mtx, u32 row, u32 col) noexcept {
  return (&mtx.m0)[row][col];
}

float& pbr_mtx(Mat3x4<float>& mtx, u32 row, u32 col) noexcept {
  return (&mtx.m0)[row][col];
}

Mat3x4<float> pbr_mul_affine(const Mat3x4<float>& lhs, const Mat3x4<float>& rhs) noexcept {
  Mat3x4<float> out;
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 4; ++col) {
      pbr_mtx(out, row, col) = pbr_mtx(lhs, row, 0) * pbr_mtx(rhs, 0, col) +
                               pbr_mtx(lhs, row, 1) * pbr_mtx(rhs, 1, col) +
                               pbr_mtx(lhs, row, 2) * pbr_mtx(rhs, 2, col) +
                               (col == 3 ? pbr_mtx(lhs, row, 3) : 0.0f);
    }
  }
  return out;
}

Mat3x4<float> pbr_mul_linear(const Mat3x4<float>& lhs, const Mat3x4<float>& rhs) noexcept {
  Mat3x4<float> out{};
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 3; ++col) {
      pbr_mtx(out, row, col) = pbr_mtx(lhs, row, 0) * pbr_mtx(rhs, 0, col) +
                               pbr_mtx(lhs, row, 1) * pbr_mtx(rhs, 1, col) +
                               pbr_mtx(lhs, row, 2) * pbr_mtx(rhs, 2, col);
    }
  }
  return out;
}

Mat3x4<float> pbr_make_look_at(PbrVec3 eye, PbrVec3 forward, PbrVec3 up) noexcept {
  const PbrVec3 f = pbr_normalize(forward);
  const PbrVec3 s = pbr_normalize(pbr_cross(f, up));
  const PbrVec3 u = pbr_cross(s, f);
  return {
      {s.x, s.y, s.z, -pbr_dot(s, eye)},
      {u.x, u.y, u.z, -pbr_dot(u, eye)},
      {-f.x, -f.y, -f.z, pbr_dot(f, eye)},
  };
}

bool pbr_probe_camera_changed(const PbrProbeCamera& reference, const PbrProbeCamera& camera) noexcept {
  if (!reference.valid || !camera.valid) {
    return true;
  }

  const float dx = reference.sourceInvView.m0[3] - camera.sourceInvView.m0[3];
  const float dy = reference.sourceInvView.m1[3] - camera.sourceInvView.m1[3];
  const float dz = reference.sourceInvView.m2[3] - camera.sourceInvView.m2[3];
  if (dx * dx + dy * dy + dz * dz >= PbrProbeDirtyDistance * PbrProbeDirtyDistance) {
    return true;
  }

  float rotationDiff = 0.0f;
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 3; ++col) {
      rotationDiff += std::abs(pbr_mtx(reference.sourceInvView, row, col) - pbr_mtx(camera.sourceInvView, row, col));
    }
  }
  return rotationDiff >= PbrProbeDirtyRotationSum;
}

PbrVec3 pbr_cube_direction(u32 face, u32 x, u32 y, u32 size) noexcept {
  const float u = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 1.0f;
  const float v = 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 1.0f;
  switch (face) {
  case 0:
    return pbr_normalize({1.0f, -v, -u});
  case 1:
    return pbr_normalize({-1.0f, -v, u});
  case 2:
    return pbr_normalize({u, 1.0f, v});
  case 3:
    return pbr_normalize({u, -1.0f, -v});
  case 4:
    return pbr_normalize({u, -v, 1.0f});
  default:
    return pbr_normalize({-u, -v, -1.0f});
  }
}

float pbr_smoothstep(float a, float b, float v) noexcept {
  const float t = pbr_saturate((v - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}

float pbr_fallback_environment_luma(PbrVec3 dir) noexcept {
  const float sky = pbr_smoothstep(0.0f, 1.0f, std::max(dir.y, 0.0f));
  const float ground = pbr_smoothstep(0.0f, 1.0f, std::max(-dir.y, 0.0f));
  const float horizon = std::max(1.0f - std::max(sky, ground), 0.0f);
  return sky * 1.15f + ground * 0.45f + horizon * 0.80f;
}

uint8_t pbr_to_unorm8(float v) noexcept { return static_cast<uint8_t>(std::round(pbr_saturate(v) * 255.0f)); }

wgpu::Extent3D pbr_physical_size(wgpu::Extent3D size, gfx::TextureFormatInfo info) noexcept {
  return {
      .width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth,
      .height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight,
      .depthOrArrayLayers = size.depthOrArrayLayers,
  };
}

void pbr_write_texture_layer_mip(wgpu::Texture& texture, u32 mip, u32 layer,
                                 const gfx::ConvertedTexture& converted) noexcept {
  const wgpu::Extent3D logicalSize{
      .width = converted.width,
      .height = converted.height,
      .depthOrArrayLayers = 1,
  };
  const auto info = gfx::format_info(converted.format);
  const auto physicalSize = pbr_physical_size(logicalSize, info);
  const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
  const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
  const uint32_t bytesPerRow = widthBlocks * info.blockSize;
  const uint32_t dataSize = bytesPerRow * heightBlocks;
  if (converted.data.size() < dataSize) {
    PbrLog.warn("PBR IBL texture upload expected {} bytes, got {}", dataSize, converted.data.size());
    return;
  }

  const wgpu::TexelCopyTextureInfo dst{
      .texture = texture,
      .mipLevel = mip,
      .origin = wgpu::Origin3D{.z = layer},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferLayout layout{
      .bytesPerRow = bytesPerRow,
      .rowsPerImage = heightBlocks,
  };
  g_queue.WriteTexture(&dst, converted.data.data(), dataSize, &layout, &physicalSize);
}

void pbr_write_texture_layer_mip(wgpu::Texture& texture, u32 mip, u32 layer, u32 width, u32 height,
                                 const std::vector<uint8_t>& data) noexcept {
  const wgpu::TexelCopyTextureInfo dst{
      .texture = texture,
      .mipLevel = mip,
      .origin = wgpu::Origin3D{.z = layer},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferLayout layout{
      .bytesPerRow = width * 4,
      .rowsPerImage = height,
  };
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  g_queue.WriteTexture(&dst, data.data(), data.size(), &layout, &size);
}

void pbr_fill_cube_texture(wgpu::Texture& texture, u32 baseSize, u32 mipCount, bool prefiltered) noexcept {
  for (u32 mip = 0; mip < mipCount; ++mip) {
    const u32 size = std::max(baseSize >> mip, 1u);
    const float roughness = mipCount > 1 ? static_cast<float>(mip) / static_cast<float>(mipCount - 1) : 0.0f;
    for (u32 face = 0; face < 6; ++face) {
      std::vector<uint8_t> pixels(static_cast<size_t>(size) * static_cast<size_t>(size) * 4);
      for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
          const PbrVec3 dir = pbr_cube_direction(face, x, y, size);
          float luma = pbr_fallback_environment_luma(dir);
          if (prefiltered) {
            luma = luma * (1.0f - roughness) + 0.80f * roughness;
          }
          const uint8_t value = pbr_to_unorm8(luma);
          const size_t offset = (static_cast<size_t>(y) * size + x) * 4;
          pixels[offset + 0] = value;
          pixels[offset + 1] = value;
          pixels[offset + 2] = value;
          pixels[offset + 3] = 255;
        }
      }
      pbr_write_texture_layer_mip(texture, mip, face, size, size, pixels);
    }
  }
}

float pbr_radical_inverse_vdc(u32 bits) noexcept {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

PbrVec3 pbr_importance_sample_ggx(float xi0, float xi1, float roughness) noexcept {
  const float a = roughness * roughness;
  const float phi = 2.0f * PbrPi * xi0;
  const float cosTheta = std::sqrt((1.0f - xi1) / (1.0f + (a * a - 1.0f) * xi1));
  const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
  return {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

float pbr_geometry_schlick_ggx(float nDotV, float roughness) noexcept {
  const float a = roughness;
  const float k = (a * a) * 0.5f;
  return nDotV / (nDotV * (1.0f - k) + k);
}

std::array<float, 2> pbr_integrate_brdf(float nDotV, float roughness) noexcept {
  constexpr u32 SampleCount = 128;
  const PbrVec3 v{std::sqrt(std::max(1.0f - nDotV * nDotV, 0.0f)), 0.0f, nDotV};
  float a = 0.0f;
  float b = 0.0f;

  for (u32 i = 0; i < SampleCount; ++i) {
    const float xi0 = static_cast<float>(i) / static_cast<float>(SampleCount);
    const float xi1 = pbr_radical_inverse_vdc(i);
    const PbrVec3 h = pbr_importance_sample_ggx(xi0, xi1, roughness);
    const float vDotH = std::max(pbr_dot(v, h), 0.0f);
    const PbrVec3 l =
        pbr_normalize({2.0f * vDotH * h.x - v.x, 2.0f * vDotH * h.y - v.y, 2.0f * vDotH * h.z - v.z});
    const float nDotL = std::max(l.z, 0.0f);
    const float nDotH = std::max(h.z, 0.0f);

    if (nDotL > 0.0f) {
      const float g = pbr_geometry_schlick_ggx(nDotL, roughness) * pbr_geometry_schlick_ggx(nDotV, roughness);
      const float gVis = (g * vDotH) / std::max(nDotH * nDotV, 1e-6f);
      const float fc = std::pow(1.0f - vDotH, 5.0f);
      a += (1.0f - fc) * gVis;
      b += fc * gVis;
    }
  }

  return {a / static_cast<float>(SampleCount), b / static_cast<float>(SampleCount)};
}

std::vector<uint8_t> pbr_generate_brdf_lut() {
  std::vector<uint8_t> pixels(static_cast<size_t>(PbrBrdfLutSize) * static_cast<size_t>(PbrBrdfLutSize) * 4);
  for (u32 y = 0; y < PbrBrdfLutSize; ++y) {
    const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(PbrBrdfLutSize);
    for (u32 x = 0; x < PbrBrdfLutSize; ++x) {
      const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(PbrBrdfLutSize);
      const auto integrated = pbr_integrate_brdf(nDotV, roughness);
      const size_t offset = (static_cast<size_t>(y) * PbrBrdfLutSize + x) * 4;
      pixels[offset + 0] = pbr_to_unorm8(integrated[0]);
      pixels[offset + 1] = pbr_to_unorm8(integrated[1]);
      pixels[offset + 2] = 0;
      pixels[offset + 3] = 255;
    }
  }
  return pixels;
}

std::filesystem::path pbr_ibl_root_path() {
  if (g_config.configPath == nullptr) {
    return {};
  }
  return std::filesystem::path{reinterpret_cast<const char8_t*>(g_config.configPath)} / "texture_replacements" /
         "pbr_ibl";
}

bool pbr_is_safe_scene_component(std::string_view value) noexcept {
  if (value.empty()) {
    return false;
  }
  for (const char ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                    ch == '_' || ch == '-';
    if (!ok) {
      return false;
    }
  }
  return true;
}

std::optional<std::filesystem::path> pbr_find_existing_file(
    const std::filesystem::path& dir, std::initializer_list<std::filesystem::path> names) noexcept {
  std::error_code ec;
  for (const auto& name : names) {
    const auto path = dir / name;
    if (std::filesystem::is_regular_file(path, ec)) {
      return path;
    }
  }
  return std::nullopt;
}

std::optional<gfx::ConvertedTexture> pbr_load_face_dds(const std::filesystem::path& path) noexcept {
  auto loaded = gfx::dds::load_dds_file(path);
  if (!loaded.has_value()) {
    PbrLog.warn("Failed to load PBR IBL DDS {}", path.string());
    return std::nullopt;
  }
  return loaded;
}

std::optional<std::array<gfx::ConvertedTexture, 6>> pbr_load_cube_mip_faces(
    const std::filesystem::path& dir, std::string_view stem, u32 mip, bool allowPlainMip0) noexcept {
  std::array<gfx::ConvertedTexture, 6> faces;
  for (u32 face = 0; face < faces.size(); ++face) {
    const auto faceName = PbrCubeFaceNames[face];
    std::optional<std::filesystem::path> path;
    if (mip == 0 && allowPlainMip0) {
      path = pbr_find_existing_file(dir, {fmt::format("{}_{}.dds", stem, faceName),
                                          fmt::format("{}_mip0_{}.dds", stem, faceName),
                                          fmt::format("{}_{}_mip0.dds", stem, faceName)});
    } else {
      path = pbr_find_existing_file(dir, {fmt::format("{}_mip{}_{}.dds", stem, mip, faceName),
                                          fmt::format("{}_{}_mip{}.dds", stem, faceName, mip)});
    }
    if (!path.has_value()) {
      return std::nullopt;
    }

    auto loaded = pbr_load_face_dds(*path);
    if (!loaded.has_value()) {
      return std::nullopt;
    }
    faces[face] = std::move(*loaded);
  }

  const auto format = faces[0].format;
  const auto width = faces[0].width;
  const auto height = faces[0].height;
  for (u32 face = 1; face < faces.size(); ++face) {
    if (faces[face].format != format || faces[face].width != width || faces[face].height != height) {
      PbrLog.warn("PBR IBL cube faces for {} mip {} must have matching dimensions and format", stem, mip);
      return std::nullopt;
    }
  }
  return faces;
}

bool pbr_load_cube_from_directory(const std::filesystem::path& dir, std::string_view stem, bool prefiltered,
                                  wgpu::Texture& texture, wgpu::TextureView& view, u32& loadedMipCount) noexcept {
  std::vector<std::array<gfx::ConvertedTexture, 6>> mips;
  for (u32 mip = 0; mip < 12; ++mip) {
    auto faces = pbr_load_cube_mip_faces(dir, stem, mip, true);
    if (!faces.has_value()) {
      break;
    }
    if (!mips.empty()) {
      const uint32_t expectedWidth = std::max(mips[0][0].width >> mip, 1u);
      const uint32_t expectedHeight = std::max(mips[0][0].height >> mip, 1u);
      if ((*faces)[0].width != expectedWidth || (*faces)[0].height != expectedHeight) {
        PbrLog.warn("PBR IBL {} mip {} expected {}x{}, got {}x{}", stem, mip, expectedWidth, expectedHeight,
                    (*faces)[0].width, (*faces)[0].height);
        break;
      }
    }
    mips.push_back(std::move(*faces));
    if (!prefiltered) {
      break;
    }
  }

  if (mips.empty()) {
    return false;
  }

  const auto& first = mips[0][0];
  const wgpu::TextureDescriptor desc{
      .label = prefiltered ? "PBR authored prefiltered specular cube" : "PBR authored irradiance cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = first.width, .height = first.height, .depthOrArrayLayers = 6},
      .format = first.format,
      .mipLevelCount = static_cast<uint32_t>(mips.size()),
      .sampleCount = 1,
  };
  texture = g_device.CreateTexture(&desc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = prefiltered ? "PBR authored prefiltered specular cube view" : "PBR authored irradiance cube view",
      .format = first.format,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = static_cast<uint32_t>(mips.size()),
      .arrayLayerCount = 6,
  };
  view = texture.CreateView(&viewDesc);

  for (u32 mip = 0; mip < mips.size(); ++mip) {
    for (u32 face = 0; face < 6; ++face) {
      pbr_write_texture_layer_mip(texture, mip, face, mips[mip][face]);
    }
  }

  loadedMipCount = static_cast<uint32_t>(mips.size());
  return true;
}

bool pbr_load_brdf_lut_from_directory(const std::filesystem::path& dir, PbrIblTextureSet& set) noexcept {
  const auto path = pbr_find_existing_file(dir, {"brdf_lut.dds", "brdf.dds"});
  if (!path.has_value()) {
    return false;
  }
  auto lut = pbr_load_face_dds(*path);
  if (!lut.has_value()) {
    return false;
  }

  const wgpu::TextureDescriptor desc{
      .label = "PBR authored BRDF LUT",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = lut->width, .height = lut->height, .depthOrArrayLayers = 1},
      .format = lut->format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.brdfLutTexture = g_device.CreateTexture(&desc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = "PBR authored BRDF LUT view",
      .format = lut->format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  set.brdfLutView = set.brdfLutTexture.CreateView(&viewDesc);
  pbr_write_texture_layer_mip(set.brdfLutTexture, 0, 0, *lut);
  return true;
}

void pbr_update_ibl_max_mip(const PbrIblTextureSet& set) noexcept {
  const float maxMip = static_cast<float>(std::max(set.prefilterMipCount, 1u) - 1u);
  pbrIblParams = {pbrIblParams.x(), pbrIblParams.y(), pbrIblParams.z(), maxMip};
}

void load_fallback_pbr_ibl_textures(PbrIblTextureSet& set) noexcept {
  const wgpu::TextureDescriptor irradianceDesc{
      .label = "PBR fallback irradiance cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrIrradianceCubeSize, .height = PbrIrradianceCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.irradianceCubeTexture = g_device.CreateTexture(&irradianceDesc);
  const wgpu::TextureViewDescriptor irradianceViewDesc{
      .label = "PBR fallback irradiance cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.irradianceCubeView = set.irradianceCubeTexture.CreateView(&irradianceViewDesc);
  pbr_fill_cube_texture(set.irradianceCubeTexture, PbrIrradianceCubeSize, 1, false);

  const wgpu::TextureDescriptor prefilterDesc{
      .label = "PBR fallback prefiltered specular cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrPrefilterCubeSize, .height = PbrPrefilterCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = PbrPrefilterMipCount,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&prefilterDesc);
  const wgpu::TextureViewDescriptor prefilterViewDesc{
      .label = "PBR fallback prefiltered specular cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = PbrPrefilterMipCount,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&prefilterViewDesc);
  pbr_fill_cube_texture(set.prefilterCubeTexture, PbrPrefilterCubeSize, PbrPrefilterMipCount, true);
  set.prefilterMipCount = PbrPrefilterMipCount;

  const wgpu::TextureDescriptor brdfDesc{
      .label = "PBR BRDF LUT",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrBrdfLutSize, .height = PbrBrdfLutSize, .depthOrArrayLayers = 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.brdfLutTexture = g_device.CreateTexture(&brdfDesc);
  const wgpu::TextureViewDescriptor brdfViewDesc{
      .label = "PBR BRDF LUT view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  set.brdfLutView = set.brdfLutTexture.CreateView(&brdfViewDesc);

  const auto brdfLut = pbr_generate_brdf_lut();
  pbr_write_texture_layer_mip(set.brdfLutTexture, 0, 0, PbrBrdfLutSize, PbrBrdfLutSize, brdfLut);
  set.available = true;
}

void select_active_pbr_ibl_set() noexcept;

void create_probe_capture_target(PbrIblTextureSet& set, std::array<wgpu::TextureView, 6>& faceViews,
                                 wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::TextureDescriptor cubeDesc{
      .label = "PBR runtime probe cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&cubeDesc);
  set.irradianceCubeTexture = set.prefilterCubeTexture;

  const wgpu::TextureViewDescriptor cubeViewDesc{
      .label = "PBR runtime probe cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&cubeViewDesc);
  set.irradianceCubeView = set.prefilterCubeTexture.CreateView(&cubeViewDesc);

  for (u32 face = 0; face < faceViews.size(); ++face) {
    const wgpu::TextureViewDescriptor faceViewDesc{
        .label = "PBR runtime probe face view",
        .format = colorFormat,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = face,
        .arrayLayerCount = 1,
    };
    faceViews[face] = set.prefilterCubeTexture.CreateView(&faceViewDesc);
  }

  set.brdfLutTexture = sFallbackPbrIbl.brdfLutTexture;
  set.brdfLutView = sFallbackPbrIbl.brdfLutView;
  set.prefilterMipCount = 1;
  set.available = false;
}

void create_processed_probe_target(PbrIblTextureSet& set, wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::TextureDescriptor irradianceDesc{
      .label = "PBR runtime probe irradiance cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrIrradianceCubeSize, .height = PbrIrradianceCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.irradianceCubeTexture = g_device.CreateTexture(&irradianceDesc);
  const wgpu::TextureViewDescriptor irradianceViewDesc{
      .label = "PBR runtime probe irradiance cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.irradianceCubeView = set.irradianceCubeTexture.CreateView(&irradianceViewDesc);
  for (u32 face = 0; face < sProbeIrradianceFaceViews.size(); ++face) {
    const wgpu::TextureViewDescriptor faceViewDesc{
        .label = "PBR runtime probe irradiance face view",
        .format = colorFormat,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = face,
        .arrayLayerCount = 1,
    };
    sProbeIrradianceFaceViews[face] = set.irradianceCubeTexture.CreateView(&faceViewDesc);
  }

  const wgpu::TextureDescriptor prefilterDesc{
      .label = "PBR runtime probe prefiltered specular cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = PbrRuntimeProbePrefilterMipCount,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&prefilterDesc);
  const wgpu::TextureViewDescriptor prefilterViewDesc{
      .label = "PBR runtime probe prefiltered specular cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = PbrRuntimeProbePrefilterMipCount,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&prefilterViewDesc);
  for (u32 mip = 0; mip < sProbePrefilterFaceViews.size(); ++mip) {
    for (u32 face = 0; face < sProbePrefilterFaceViews[mip].size(); ++face) {
      const wgpu::TextureViewDescriptor faceViewDesc{
          .label = "PBR runtime probe prefiltered specular face view",
          .format = colorFormat,
          .dimension = wgpu::TextureViewDimension::e2D,
          .baseMipLevel = mip,
          .mipLevelCount = 1,
          .baseArrayLayer = face,
          .arrayLayerCount = 1,
      };
      sProbePrefilterFaceViews[mip][face] = set.prefilterCubeTexture.CreateView(&faceViewDesc);
    }
  }

  set.brdfLutTexture = sFallbackPbrIbl.brdfLutTexture;
  set.brdfLutView = sFallbackPbrIbl.brdfLutView;
  set.prefilterMipCount = PbrRuntimeProbePrefilterMipCount;
  set.available = false;
}

wgpu::RenderPipeline create_probe_filter_pipeline(wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{
      .code = R"""(
struct Params {
  face: u32,
  mode: u32,
  mip: u32,
  sample_count: u32,
  roughness: f32,
  source_mip: f32,
  padding0: f32,
  padding1: f32,
};

@group(0) @binding(0) var source_sampler: sampler;
@group(0) @binding(1) var source_cube: texture_cube<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, -3.0),
  vec2<f32>(3.0, 1.0)
);
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
  vec2<f32>(0.0, 0.0),
  vec2<f32>(0.0, 2.0),
  vec2<f32>(2.0, 0.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var out: VertexOutput;
  out.pos = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  out.uv = uvs[vertex_index];
  return out;
}

fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
  let p = uv * 2.0 - vec2<f32>(1.0);
  let u = p.x;
  let v = p.y;
  switch face {
  case 0u: { return normalize(vec3<f32>(1.0, -v, -u)); }
  case 1u: { return normalize(vec3<f32>(-1.0, -v, u)); }
  case 2u: { return normalize(vec3<f32>(u, 1.0, v)); }
  case 3u: { return normalize(vec3<f32>(u, -1.0, -v)); }
  case 4u: { return normalize(vec3<f32>(u, -v, 1.0)); }
  default: { return normalize(vec3<f32>(-u, -v, -1.0)); }
  }
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn build_tangent(n: vec3<f32>) -> mat3x3<f32> {
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) {
    up = vec3<f32>(0.0, 0.0, 1.0);
  }
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  return mat3x3<f32>(tangent, bitangent, n);
}

fn sample_env(dir: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(source_cube, source_sampler, dir, 0.0).rgb;
}

fn filter_irradiance(n: vec3<f32>) -> vec3<f32> {
  let basis = build_tangent(n);
  var color = vec3<f32>(0.0);
  let count = max(params.sample_count, 1u);
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= count) {
      break;
    }
    let xi = vec2<f32>((f32(i) + 0.5) / f32(count), radical_inverse_vdc(i));
    let phi = 6.28318530718 * xi.x;
    let cos_theta = sqrt(1.0 - xi.y);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    color += sample_env(normalize(basis * local));
  }
  return color / f32(count);
}

fn filter_prefilter(n: vec3<f32>) -> vec3<f32> {
  if (params.roughness <= 0.001) {
    return sample_env(n);
  }

  let basis = build_tangent(n);
  let count = max(params.sample_count, 1u);
  let a = max(params.roughness * params.roughness, 0.001);
  var color = vec3<f32>(0.0);
  var total_weight = 0.0;
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= count) {
      break;
    }
    let xi = vec2<f32>((f32(i) + 0.5) / f32(count), radical_inverse_vdc(i));
    let phi = 6.28318530718 * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = normalize(basis * vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
    let l = normalize(2.0 * dot(n, h) * h - n);
    let ndotl = max(dot(n, l), 0.0);
    if (ndotl > 0.0) {
      color += sample_env(l) * ndotl;
      total_weight += ndotl;
    }
  }
  return color / max(total_weight, 0.0001);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let n = cube_dir(params.face, in.uv);
  var color: vec3<f32>;
  if (params.mode == 0u) {
    color = filter_irradiance(n);
  } else {
    color = filter_prefilter(n);
  }
  return vec4<f32>(color, 1.0);
}
)""",
  }};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &wgslSource,
      .label = "PBR runtime probe filter shader",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  const std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::Cube,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform},
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{
      .label = "PBR runtime probe filter bind group layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  sProbeFilterBindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDesc);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{
      .label = "PBR runtime probe filter pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &sProbeFilterBindGroupLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDesc);
  const std::array colorTargets{
      wgpu::ColorTargetState{
          .format = colorFormat,
          .writeMask = wgpu::ColorWriteMask::All,
      },
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor pipelineDesc{
      .label = "PBR runtime probe filter pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&pipelineDesc);
}

void create_probe_filter_bind_groups() noexcept {
  for (u32 pass = 0; pass < PbrProbeFilterPassCount; ++pass) {
    const wgpu::BufferDescriptor bufferDesc{
        .label = "PBR runtime probe filter params",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(PbrProbeFilterParams),
    };
    sProbeFilterUniforms[pass] = g_device.CreateBuffer(&bufferDesc);

    const bool irradiance = pass < PbrProbeIrradianceFilterPassCount;
    const u32 localPass = irradiance ? pass : pass - PbrProbeIrradianceFilterPassCount;
    const u32 face = irradiance ? localPass : localPass % 6;
    const u32 mip = irradiance ? 0 : localPass / 6;
    const float roughness = PbrRuntimeProbePrefilterMipCount > 1
                                ? static_cast<float>(mip) / static_cast<float>(PbrRuntimeProbePrefilterMipCount - 1)
                                : 0.0f;
    const PbrProbeFilterParams params{
        .face = face,
        .mode = irradiance ? 0u : 1u,
        .mip = mip,
        .sampleCount = irradiance ? 24u : 32u,
        .roughness = roughness,
        .sourceMip = 0.0f,
    };
    g_queue.WriteBuffer(sProbeFilterUniforms[pass], 0, &params, sizeof(params));

    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .sampler = sPbrIblSampler,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .textureView = sProbeCapturePbrIbl.prefilterCubeView,
        },
        wgpu::BindGroupEntry{
            .binding = 2,
            .buffer = sProbeFilterUniforms[pass],
            .offset = 0,
            .size = sizeof(PbrProbeFilterParams),
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDesc{
        .label = "PBR runtime probe filter bind group",
        .layout = sProbeFilterBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    sProbeFilterBindGroups[pass] = g_device.CreateBindGroup(&bindGroupDesc);
  }
}

void ensure_probe_capture_resources() noexcept {
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  const u32 msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  if (sProbeCaptureResourcesReady && sProbeCaptureColorFormat == colorFormat &&
      sProbeCaptureDepthFormat == depthFormat && sProbeCaptureMsaaSamples == msaaSamples) {
    return;
  }

  create_processed_probe_target(sProbePbrIbl, colorFormat);
  create_probe_capture_target(sProbeCapturePbrIbl, sProbeCaptureFaceViews, colorFormat);
  sProbeFilterPipeline = create_probe_filter_pipeline(colorFormat);
  create_probe_filter_bind_groups();

  if (msaaSamples > 1) {
    const wgpu::TextureDescriptor msaaDesc{
        .label = "PBR runtime probe capture MSAA color",
        .usage = wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 1},
        .format = colorFormat,
        .mipLevelCount = 1,
        .sampleCount = msaaSamples,
    };
    sProbeCaptureMsaaColorTexture = g_device.CreateTexture(&msaaDesc);
    sProbeCaptureMsaaColorView = sProbeCaptureMsaaColorTexture.CreateView();
  } else {
    sProbeCaptureMsaaColorTexture = {};
    sProbeCaptureMsaaColorView = {};
  }

  const wgpu::TextureDescriptor depthDesc{
      .label = "PBR runtime probe capture depth",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 1},
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = msaaSamples,
  };
  sProbeCaptureDepthTexture = g_device.CreateTexture(&depthDesc);
  sProbeCaptureDepthView = sProbeCaptureDepthTexture.CreateView();

  sProbeCaptureColorFormat = colorFormat;
  sProbeCaptureDepthFormat = depthFormat;
  sProbeCaptureMsaaSamples = msaaSamples;
  sProbeCaptureFace = 0;
  sProbeCaptureDelayFrames = 0;
  sProbeFramesSinceRefresh = PbrProbePeriodicRefreshFrames;
  sProbeRefreshPending = true;
  sProbeCaptureCamera.valid = false;
  sProbeLastCompletedCamera.valid = false;
  sProbeCaptureResourcesReady = true;
  select_active_pbr_ibl_set();
}

bool load_authored_pbr_ibl_from_directory(const std::filesystem::path& dir, PbrIblTextureSet& set) noexcept {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return false;
  }

  bool loadedAny = false;
  u32 prefilterMipCount = set.prefilterMipCount;
  loadedAny |= pbr_load_cube_from_directory(dir, "irradiance", false, set.irradianceCubeTexture,
                                            set.irradianceCubeView, prefilterMipCount);
  if (pbr_load_cube_from_directory(dir, "prefilter", true, set.prefilterCubeTexture, set.prefilterCubeView,
                                   prefilterMipCount)) {
    set.prefilterMipCount = prefilterMipCount;
    loadedAny = true;
  }
  loadedAny |= pbr_load_brdf_lut_from_directory(dir, set);
  return loadedAny;
}

PbrIblTextureSet& active_pbr_ibl_set() noexcept { return sActivePbrIbl != nullptr ? *sActivePbrIbl : sFallbackPbrIbl; }

AuroraPbrIblSource active_pbr_ibl_source() noexcept {
  if (sActivePbrIbl == &sProbePbrIbl) {
    return AURORA_PBR_IBL_SOURCE_PROBE;
  }
  if (sActivePbrIbl == &sAuthoredPbrIbl) {
    return AURORA_PBR_IBL_SOURCE_AUTHORED;
  }
  return AURORA_PBR_IBL_SOURCE_FALLBACK;
}

template <size_t N>
void copy_status_string(char (&dst)[N], std::string_view value) noexcept {
  static_assert(N > 0);
  const size_t len = std::min(value.size(), N - 1);
  std::memcpy(dst, value.data(), len);
  dst[len] = '\0';
}

void select_active_pbr_ibl_set() noexcept {
  PbrIblTextureSet* selected = &sFallbackPbrIbl;
  switch (sRequestedPbrIblSource) {
  case AURORA_PBR_IBL_SOURCE_PROBE:
    if (sProbePbrIbl.available) {
      selected = &sProbePbrIbl;
    }
    break;
  case AURORA_PBR_IBL_SOURCE_AUTHORED:
    if (sAuthoredPbrIbl.available) {
      selected = &sAuthoredPbrIbl;
    }
    break;
  case AURORA_PBR_IBL_SOURCE_FALLBACK:
  default:
    break;
  }

  sActivePbrIbl = selected;
  pbr_update_ibl_max_mip(*sActivePbrIbl);
  g_gxState.stateDirty = true;
}

Mat3x4<float> pbr_load_mtx3x4(const float* data) noexcept {
  if (data == nullptr) {
    return {};
  }
  return {
      {data[0], data[1], data[2], data[3]},
      {data[4], data[5], data[6], data[7]},
      {data[8], data[9], data[10], data[11]},
  };
}

Mat4x4<float> pbr_projection_from_gx_mtx(const float* data) noexcept {
  if (data == nullptr) {
    return {};
  }
  const float p0 = data[0];
  const float p1 = data[2];
  const float p2 = data[5];
  const float p3 = data[6];
  const float p4 = data[10];
  const float p5 = data[11];
  Mat4x4<float> proj{};
  proj.m0[0] = p0;
  proj.m0[2] = p1;
  proj.m1[1] = p2;
  proj.m1[2] = p3;
  proj.m2[2] = p4;
  proj.m2[3] = p5;
  proj.m3[2] = -1.0f;
  return proj;
}

void pbr_update_probe_face_matrices() noexcept {
  const PbrVec3 eye{sProbeCamera.sourceInvView.m0[3], sProbeCamera.sourceInvView.m1[3],
                    sProbeCamera.sourceInvView.m2[3]};
  constexpr std::array<PbrVec3, 6> directions{{
      {1.0f, 0.0f, 0.0f},
      {-1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, -1.0f},
  }};
  constexpr std::array<PbrVec3, 6> ups{{
      {0.0f, -1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
  }};

  for (u32 face = 0; face < 6; ++face) {
    const Mat3x4<float> faceView = pbr_make_look_at(eye, directions[face], ups[face]);
    sProbeCamera.viewFromSource[face] = pbr_mul_affine(faceView, sProbeCamera.sourceInvView);
    sProbeCamera.normalFromSource[face] = pbr_mul_linear(faceView, sProbeCamera.sourceInvView);
  }
}

gfx::TextureHandle find_named_pbr_sidecar(std::string_view materialName,
                                          std::initializer_list<std::string_view> suffixes) {
  if (materialName.empty()) {
    return {};
  }

  for (std::string_view suffix : suffixes) {
    std::string candidate{materialName};
    candidate += suffix;
    if (auto texture = gfx::texture_replacement::find_named_pbr_texture(candidate); texture.has_value()) {
      return *texture;
    }
  }

  return {};
}

void clear_pbr_material_maps() {
  pbrMaterialRmaos.reset();
  pbrMaterialRoughness.reset();
  pbrMaterialMetallic.reset();
  pbrMaterialAo.reset();
  pbrMaterialSpecular.reset();
  pbrMaterialNormal.reset();
  pbrMaterialEmissiveMap.reset();
}
} // namespace pbr_internal

using namespace pbr_internal;

void initialize_pbr_resources() noexcept {
  const wgpu::SamplerDescriptor materialSamplerDesc{
      .label = "PBR material sampler",
      .addressModeU = wgpu::AddressMode::Repeat,
      .addressModeV = wgpu::AddressMode::Repeat,
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.0f,
      .lodMaxClamp = 1000.0f,
  };
  sPbrMaterialSampler = gfx::sampler_ref(materialSamplerDesc);

  const wgpu::SamplerDescriptor iblSamplerDesc{
      .label = "PBR IBL sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.0f,
      .lodMaxClamp = static_cast<float>(PbrPrefilterMipCount - 1),
  };
  sPbrIblSampler = gfx::sampler_ref(iblSamplerDesc);

  const wgpu::SamplerDescriptor brdfSamplerDesc{
      .label = "PBR BRDF LUT sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Undefined,
  };
  sPbrBrdfLutSampler = gfx::sampler_ref(brdfSamplerDesc);

  load_fallback_pbr_ibl_textures(sFallbackPbrIbl);
  sActivePbrIbl = &sFallbackPbrIbl;
  select_active_pbr_ibl_set();
}

void add_pbr_texture_layout_entries(
    std::array<wgpu::BindGroupLayoutEntry, TextureBindGroupEntryCount>& textureEntries) noexcept {
  const auto addPbrTextureLayout = [&](u32 textureBinding, u32 samplerBinding,
                                       wgpu::TextureViewDimension viewDimension = wgpu::TextureViewDimension::e2D) {
    textureEntries[textureBinding] = {
        .binding = textureBinding,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            {
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = viewDimension,
            },
    };
    textureEntries[samplerBinding] = {
        .binding = samplerBinding,
        .visibility = wgpu::ShaderStage::Fragment,
        .sampler = {.type = wgpu::SamplerBindingType::Filtering},
    };
  };

  addPbrTextureLayout(pbr_rmaos_texture_binding(0), pbr_rmaos_sampler_binding(0));
  addPbrTextureLayout(pbr_roughness_texture_binding(0), pbr_roughness_sampler_binding(0));
  addPbrTextureLayout(pbr_metallic_texture_binding(0), pbr_metallic_sampler_binding(0));
  addPbrTextureLayout(pbr_ao_texture_binding(0), pbr_ao_sampler_binding(0));
  addPbrTextureLayout(pbr_specular_texture_binding(0), pbr_specular_sampler_binding(0));
  addPbrTextureLayout(pbr_normal_texture_binding(0), pbr_normal_sampler_binding(0));
  addPbrTextureLayout(pbr_emissive_texture_binding(0), pbr_emissive_sampler_binding(0));
  addPbrTextureLayout(pbr_ibl_irradiance_texture_binding(0), pbr_ibl_irradiance_sampler_binding(0),
                      wgpu::TextureViewDimension::Cube);
  addPbrTextureLayout(pbr_ibl_prefilter_texture_binding(0), pbr_ibl_prefilter_sampler_binding(0),
                      wgpu::TextureViewDimension::Cube);
  addPbrTextureLayout(pbr_ibl_brdf_lut_texture_binding(0), pbr_ibl_brdf_lut_sampler_binding(0));
}

void add_pbr_empty_bind_group_entries(std::array<wgpu::BindGroupEntry, TextureBindGroupEntryCount>& entries,
                                      const wgpu::TextureView& emptyTextureView,
                                      const wgpu::Sampler& emptySampler) noexcept {
  const auto addPbrEmptyBinding = [&](u32 textureBinding, u32 samplerBinding) {
    entries[textureBinding] = {
        .binding = textureBinding,
        .textureView = emptyTextureView,
    };
    entries[samplerBinding] = {
        .binding = samplerBinding,
        .sampler = emptySampler,
    };
  };

  addPbrEmptyBinding(pbr_rmaos_texture_binding(0), pbr_rmaos_sampler_binding(0));
  addPbrEmptyBinding(pbr_roughness_texture_binding(0), pbr_roughness_sampler_binding(0));
  addPbrEmptyBinding(pbr_metallic_texture_binding(0), pbr_metallic_sampler_binding(0));
  addPbrEmptyBinding(pbr_ao_texture_binding(0), pbr_ao_sampler_binding(0));
  addPbrEmptyBinding(pbr_specular_texture_binding(0), pbr_specular_sampler_binding(0));
  addPbrEmptyBinding(pbr_normal_texture_binding(0), pbr_normal_sampler_binding(0));
  addPbrEmptyBinding(pbr_emissive_texture_binding(0), pbr_emissive_sampler_binding(0));
  const auto& ibl = active_pbr_ibl_set();
  entries[pbr_ibl_irradiance_texture_binding(0)] = {
      .binding = pbr_ibl_irradiance_texture_binding(0),
      .textureView = ibl.irradianceCubeView,
  };
  entries[pbr_ibl_irradiance_sampler_binding(0)] = {
      .binding = pbr_ibl_irradiance_sampler_binding(0),
      .sampler = sPbrIblSampler,
  };
  entries[pbr_ibl_prefilter_texture_binding(0)] = {
      .binding = pbr_ibl_prefilter_texture_binding(0),
      .textureView = ibl.prefilterCubeView,
  };
  entries[pbr_ibl_prefilter_sampler_binding(0)] = {
      .binding = pbr_ibl_prefilter_sampler_binding(0),
      .sampler = sPbrIblSampler,
  };
  entries[pbr_ibl_brdf_lut_texture_binding(0)] = {
      .binding = pbr_ibl_brdf_lut_texture_binding(0),
      .textureView = ibl.brdfLutView,
  };
  entries[pbr_ibl_brdf_lut_sampler_binding(0)] = {
      .binding = pbr_ibl_brdf_lut_sampler_binding(0),
      .sampler = sPbrBrdfLutSampler,
  };
}

void bind_pbr_texture_entries(std::array<WGPUBindGroupEntry, TextureBindGroupEntryCount>& textureEntries,
                              const ShaderInfo& info, const wgpu::TextureView& emptyTextureView,
                              const wgpu::Sampler& emptySampler) noexcept {
  const auto bindPbrTexture = [&](u32 textureBinding, u32 samplerBinding, const gfx::TextureHandle& handle,
                                  const gfx::TextureBind* samplerSource) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    WGPUBindGroupEntry& pbrSamplerEntry = textureEntries[samplerBinding];
    pbrTextureEntry.binding = textureBinding;
    pbrSamplerEntry.binding = samplerBinding;
    if (handle && samplerSource != nullptr && *samplerSource) {
      pbrTextureEntry.textureView = handle->sampleTextureView.Get();
      pbrSamplerEntry.sampler = gfx::sampler_ref(samplerSource->get_descriptor()).Get();
    } else {
      pbrTextureEntry.textureView = emptyTextureView.Get();
      pbrSamplerEntry.sampler = emptySampler.Get();
    }
  };
  const auto bindPbrView = [&](u32 textureBinding, u32 samplerBinding, const wgpu::TextureView& textureView,
                               const wgpu::Sampler& sampler) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    WGPUBindGroupEntry& pbrSamplerEntry = textureEntries[samplerBinding];
    pbrTextureEntry.binding = textureBinding;
    pbrSamplerEntry.binding = samplerBinding;
    pbrTextureEntry.textureView = textureView.Get();
    pbrSamplerEntry.sampler = sampler.Get();
  };
  const auto bindPbrStandaloneTexture = [&](u32 textureBinding, u32 samplerBinding, const gfx::TextureHandle& handle) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    WGPUBindGroupEntry& pbrSamplerEntry = textureEntries[samplerBinding];
    pbrTextureEntry.binding = textureBinding;
    pbrSamplerEntry.binding = samplerBinding;
    if (handle) {
      pbrTextureEntry.textureView = handle->sampleTextureView.Get();
      pbrSamplerEntry.sampler = sPbrMaterialSampler.Get();
    } else {
      pbrTextureEntry.textureView = emptyTextureView.Get();
      pbrSamplerEntry.sampler = emptySampler.Get();
    }
  };

  const gfx::TextureBind* pbrTex = nullptr;
  if ((info.pbrFlags & PbrMaterialEnabled) != 0 && info.pbrTexMapId < MaxTextures) {
    pbrTex = &g_gxState.textures[info.pbrTexMapId];
  }
  const bool useGlobalMaps = (info.pbrFlags & PbrMaterialUseGlobalMaps) != 0;
  if (useGlobalMaps) {
    bindPbrStandaloneTexture(pbr_rmaos_texture_binding(0), pbr_rmaos_sampler_binding(0), pbrMaterialRmaos);
    bindPbrStandaloneTexture(pbr_roughness_texture_binding(0), pbr_roughness_sampler_binding(0), pbrMaterialRoughness);
    bindPbrStandaloneTexture(pbr_metallic_texture_binding(0), pbr_metallic_sampler_binding(0), pbrMaterialMetallic);
    bindPbrStandaloneTexture(pbr_ao_texture_binding(0), pbr_ao_sampler_binding(0), pbrMaterialAo);
    bindPbrStandaloneTexture(pbr_specular_texture_binding(0), pbr_specular_sampler_binding(0), pbrMaterialSpecular);
    bindPbrStandaloneTexture(pbr_normal_texture_binding(0), pbr_normal_sampler_binding(0), pbrMaterialNormal);
    bindPbrStandaloneTexture(pbr_emissive_texture_binding(0), pbr_emissive_sampler_binding(0),
                             pbrMaterialEmissiveMap);
  } else {
    bindPbrTexture(pbr_rmaos_texture_binding(0), pbr_rmaos_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrRmaos : nullptr, pbrTex);
    bindPbrTexture(pbr_roughness_texture_binding(0), pbr_roughness_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrRoughness : nullptr, pbrTex);
    bindPbrTexture(pbr_metallic_texture_binding(0), pbr_metallic_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrMetallic : nullptr, pbrTex);
    bindPbrTexture(pbr_ao_texture_binding(0), pbr_ao_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrAo : nullptr, pbrTex);
    bindPbrTexture(pbr_specular_texture_binding(0), pbr_specular_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrSpecular : nullptr, pbrTex);
    bindPbrTexture(pbr_normal_texture_binding(0), pbr_normal_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrNormal : nullptr, pbrTex);
    bindPbrTexture(pbr_emissive_texture_binding(0), pbr_emissive_sampler_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrEmissive : nullptr, pbrTex);
  }
  const auto& ibl = active_pbr_ibl_set();
  bindPbrView(pbr_ibl_irradiance_texture_binding(0), pbr_ibl_irradiance_sampler_binding(0),
              ibl.irradianceCubeView, sPbrIblSampler);
  bindPbrView(pbr_ibl_prefilter_texture_binding(0), pbr_ibl_prefilter_sampler_binding(0), ibl.prefilterCubeView,
              sPbrIblSampler);
  bindPbrView(pbr_ibl_brdf_lut_texture_binding(0), pbr_ibl_brdf_lut_sampler_binding(0), ibl.brdfLutView,
              sPbrBrdfLutSampler);
}

bool pbr_probe_capture_requested() noexcept {
  using namespace pbr_internal;
  if (!enablePbrMaterialOverride || pbrIblParams.x() <= 0.0f ||
      sRequestedPbrIblSource != AURORA_PBR_IBL_SOURCE_PROBE || !sProbeCamera.valid) {
    sProbeCaptureDelayFrames = 0;
    return false;
  }

  if (sProbeCaptureDelayFrames > 0) {
    --sProbeCaptureDelayFrames;
    return false;
  }

  if (sProbeCaptureFace != 0 || !sProbePbrIbl.available || sProbeRefreshPending) {
    return true;
  }

  if (sProbeAutoRefresh && sProbeFramesSinceRefresh >= PbrProbePeriodicRefreshFrames) {
    sProbeRefreshPending = true;
    return true;
  }

  ++sProbeFramesSinceRefresh;
  return false;
}

uint32_t pbr_probe_cube_size() noexcept { return pbr_internal::PbrRuntimeProbeCubeSize; }

uint32_t pbr_probe_capture_face() noexcept { return pbr_internal::sProbeCaptureFace; }

void begin_pbr_probe_capture() noexcept {
  using namespace pbr_internal;
  ensure_probe_capture_resources();
  if (sProbeCaptureFace == 0) {
    sProbeCaptureCamera = sProbeCamera;
    sProbeRefreshPending = false;
  }
}

void finish_pbr_probe_capture() noexcept {
  using namespace pbr_internal;
  if (!sProbeCaptureResourcesReady) {
    return;
  }

  sProbeCaptureDelayFrames = PbrProbeCaptureFramesPerFace > 0 ? PbrProbeCaptureFramesPerFace - 1 : 0;
  sProbeCaptureFace = (sProbeCaptureFace + 1) % 6;
  if (sProbeCaptureFace != 0) {
    return;
  }

  sProbePbrIbl.available = true;
  sProbeFilterPending = true;
  sProbeFramesSinceRefresh = 0;
  sProbeRefreshPending = pbr_probe_camera_changed(sProbeCaptureCamera, sProbeCamera);
  sProbeLastCompletedCamera = sProbeCaptureCamera;
  sProbeCaptureCamera.valid = false;
  select_active_pbr_ibl_set();
}

const wgpu::TextureView& pbr_probe_capture_face_view(uint32_t face) noexcept {
  return pbr_internal::sProbeCaptureFaceViews[std::min<uint32_t>(face, 5)];
}

const wgpu::TextureView& pbr_probe_capture_color_view(uint32_t face) noexcept {
  if (webgpu::g_graphicsConfig.msaaSamples > 1) {
    return pbr_internal::sProbeCaptureMsaaColorView;
  }
  return pbr_probe_capture_face_view(face);
}

const wgpu::TextureView& pbr_probe_capture_depth_view() noexcept { return pbr_internal::sProbeCaptureDepthView; }

void run_pbr_probe_filter(const wgpu::CommandEncoder& cmd) noexcept {
  using namespace pbr_internal;
  if (!sProbeFilterPending || !sProbeCaptureResourcesReady || !sProbeFilterPipeline) {
    return;
  }

  const auto runPass = [&](u32 passIndex, const wgpu::TextureView& targetView, u32 size, std::string_view label) {
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = targetView,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {0.0, 0.0, 0.0, 1.0},
        },
    };
    const wgpu::RenderPassDescriptor passDesc{
        .label = label.data(),
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    auto pass = cmd.BeginRenderPass(&passDesc);
    pass.SetPipeline(sProbeFilterPipeline);
    pass.SetBindGroup(0, sProbeFilterBindGroups[passIndex]);
    pass.SetViewport(0.0f, 0.0f, static_cast<float>(size), static_cast<float>(size), 0.0f, 1.0f);
    pass.SetScissorRect(0, 0, size, size);
    pass.Draw(3);
    pass.End();
  };

  for (u32 face = 0; face < 6; ++face) {
    runPass(face, sProbeIrradianceFaceViews[face], PbrIrradianceCubeSize, "PBR probe irradiance filter");
  }

  for (u32 mip = 0; mip < PbrRuntimeProbePrefilterMipCount; ++mip) {
    const u32 size = std::max(PbrRuntimeProbeCubeSize >> mip, 1u);
    for (u32 face = 0; face < 6; ++face) {
      const u32 passIndex = PbrProbeIrradianceFilterPassCount + mip * 6 + face;
      runPass(passIndex, sProbePrefilterFaceViews[mip][face], size, "PBR probe prefilter");
    }
  }

  sProbeFilterPending = false;
  select_active_pbr_ibl_set();
}

void patch_pbr_probe_uniform(uint8_t* uniformData, const ProbeUniformPatchInfo& patch, uint32_t face) noexcept {
  using namespace pbr_internal;
  const auto& probeCamera = sProbeCaptureCamera.valid ? sProbeCaptureCamera : sProbeCamera;
  if (uniformData == nullptr || !patch.eligible || !probeCamera.valid || face >= 6) {
    return;
  }
  if (patch.nrmMtxOffset + sizeof(Mat3x4<float>) * MaxPnMtx > patch.uniformSize ||
      patch.projectionOffset + sizeof(Mat4x4<float>) > patch.uniformSize) {
    return;
  }

  auto* viewport = reinterpret_cast<float*>(uniformData + 8);
  const float size = static_cast<float>(PbrRuntimeProbeCubeSize);
  viewport[0] = size;
  viewport[1] = size;
  viewport[2] = size;
  viewport[3] = size;
  std::memcpy(uniformData + patch.projectionOffset, &probeCamera.projection, sizeof(probeCamera.projection));

  auto* posMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.pnMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    posMtx[i] = pbr_mul_affine(probeCamera.viewFromSource[face], posMtx[i]);
  }

  auto* nrmMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.nrmMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    nrmMtx[i] = pbr_mul_linear(probeCamera.normalFromSource[face], nrmMtx[i]);
  }
}

void configure_pbr_material_override(ShaderConfig& config, const ShaderInfo& info) noexcept {
  config.pbrFlags = 0;
  config.pbrTexMapId = GX_TEXMAP_NULL;
  config.pbrTexCoordId = GX_TEXCOORD_NULL;
  config.pbrChannelId = GX_COLOR_NULL;

  if (!enablePbrMaterialOverride || config.lineMode != 0) {
    return;
  }

  const auto choosePbrChannel = [&]() -> GXChannelID {
    for (u32 i = 0; i < config.tevStageCount; ++i) {
      const auto channel = config.tevStages[i].channelId;
      if (channel != GX_COLOR_NULL && channel != GX_COLOR_ZERO && channel != GX_ALPHA_BUMP &&
          channel != GX_ALPHA_BUMPN) {
        return channel;
      }
    }
    if (config.colorChannels[GX_COLOR0].lightingEnabled || config.colorChannels[GX_ALPHA0].lightingEnabled) {
      return GX_COLOR0A0;
    }
    if (config.colorChannels[GX_COLOR1].lightingEnabled || config.colorChannels[GX_ALPHA1].lightingEnabled) {
      return GX_COLOR1A1;
    }
    return GX_COLOR0A0;
  };

  const auto choosePbrTexCoord = [&]() -> GXTexCoordID {
    for (u32 i = 0; i < config.tcgs.size(); ++i) {
      if (config.tcgs[i].src != GX_MAX_TEXGENSRC) {
        return static_cast<GXTexCoordID>(i);
      }
    }
    return GX_TEXCOORD_NULL;
  };

  if (pbrMaterialOverrideActive) {
    config.pbrFlags =
        PbrMaterialEnabled | PbrMaterialUsePrevAlbedo | PbrMaterialPrevAlbedoIsLit | PbrMaterialUseGlobalMaps;
    if (pbrMaterialEmissive.w() > 0.0f) {
      config.pbrFlags |= PbrMaterialHasConstantEmissive;
    }
    if (pbrMaterialRmaos) {
      config.pbrFlags |= PbrMaterialHasRmaos;
    }
    if (pbrMaterialRoughness) {
      config.pbrFlags |= PbrMaterialHasRoughness;
    }
    if (pbrMaterialMetallic) {
      config.pbrFlags |= PbrMaterialHasMetallic;
    }
    if (pbrMaterialAo) {
      config.pbrFlags |= PbrMaterialHasAo;
    }
    if (pbrMaterialSpecular) {
      config.pbrFlags |= PbrMaterialHasSpecular;
    }
    if (pbrMaterialNormal) {
      config.pbrFlags |= PbrMaterialHasNormal;
    }
    if (pbrMaterialEmissiveMap) {
      config.pbrFlags |= PbrMaterialHasEmissive;
    }
    config.pbrTexMapId = 0;
    config.pbrTexCoordId = underlying(choosePbrTexCoord());
    config.pbrChannelId = underlying(choosePbrChannel());
    return;
  }

  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (stage.texMapId == GX_TEXMAP_NULL || stage.texCoordId == GX_TEXCOORD_NULL) {
      continue;
    }

    const u32 texMap = underlying(stage.texMapId);
    if (texMap >= MaxTextures || !info.sampledTextures.test(texMap)) {
      continue;
    }

    const auto& tex = g_gxState.textures[texMap];
    if (!tex) {
      continue;
    }

    const auto& ref = *tex.ref;
    if (!ref.pbrRmaos && !ref.pbrRoughness && !ref.pbrMetallic && !ref.pbrAo && !ref.pbrSpecular && !ref.pbrNormal &&
        !ref.pbrEmissive) {
      continue;
    }

    config.pbrFlags = PbrMaterialEnabled | PbrMaterialUsePrevAlbedo | PbrMaterialPrevAlbedoIsLit;
    if (ref.pbrRmaos) {
      config.pbrFlags |= PbrMaterialHasRmaos;
    }
    if (ref.pbrRoughness) {
      config.pbrFlags |= PbrMaterialHasRoughness;
    }
    if (ref.pbrMetallic) {
      config.pbrFlags |= PbrMaterialHasMetallic;
    }
    if (ref.pbrAo) {
      config.pbrFlags |= PbrMaterialHasAo;
    }
    if (ref.pbrSpecular) {
      config.pbrFlags |= PbrMaterialHasSpecular;
    }
    if (tex.ref->pbrNormal) {
      config.pbrFlags |= PbrMaterialHasNormal;
    }
    if (tex.ref->pbrEmissive) {
      config.pbrFlags |= PbrMaterialHasEmissive;
    }
    config.pbrTexMapId = texMap;
    config.pbrTexCoordId = underlying(stage.texCoordId);
    config.pbrChannelId = underlying(stage.channelId);
    return;
  }
}
} // namespace aurora::gx

void aurora_enable_pbr(bool enabled) {
  aurora::gx::enablePbrMaterialOverride = enabled;
  aurora::gx::g_gxState.stateDirty = true;
}

bool aurora_pbr_enabled() { return aurora::gx::enablePbrMaterialOverride; }

void aurora_set_pbr_light_params(float ambient, float ambient_specular, float fill_intensity) {
  aurora::gx::pbrParams = {ambient, ambient_specular, fill_intensity, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_material_params(float diffuse_scale, float specular_scale) {
  aurora::gx::pbrScales = {diffuse_scale, specular_scale, aurora::gx::pbrScales.z(), 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_debug_mode(AuroraPbrDebugMode mode) {
  const int clampedMode =
      std::clamp(static_cast<int>(mode), static_cast<int>(AURORA_PBR_DEBUG_OFF),
                 static_cast<int>(AURORA_PBR_DEBUG_IBL_SPECULAR));
  aurora::gx::pbrScales = {aurora::gx::pbrScales.x(), aurora::gx::pbrScales.y(), static_cast<float>(clampedMode),
                           0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_normal_params(float strength, bool flip_y, bool invert_handedness) {
  const float normalStrength = strength < 0.0f ? 0.0f : strength;
  aurora::gx::pbrNormalParams = {normalStrength, flip_y ? -1.0f : 1.0f, invert_handedness ? -1.0f : 1.0f, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ambient_gradient_params(float sky, float ground, float horizon, float environment_tint) {
  aurora::gx::pbrAmbientGradient = {sky, ground, horizon, environment_tint};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ibl_params(bool enabled, float diffuse_strength, float specular_strength) {
  aurora::gx::pbrIblParams = {enabled ? 1.0f : 0.0f, std::max(diffuse_strength, 0.0f),
                              std::max(specular_strength, 0.0f), aurora::gx::pbrIblParams.w()};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ibl_source(AuroraPbrIblSource source) {
  switch (source) {
  case AURORA_PBR_IBL_SOURCE_PROBE:
  case AURORA_PBR_IBL_SOURCE_AUTHORED:
  case AURORA_PBR_IBL_SOURCE_FALLBACK:
    aurora::gx::pbr_internal::sRequestedPbrIblSource = source;
    break;
  default:
    aurora::gx::pbr_internal::sRequestedPbrIblSource = AURORA_PBR_IBL_SOURCE_PROBE;
    break;
  }

  aurora::gx::pbr_internal::select_active_pbr_ibl_set();
}

bool aurora_pbr_probe_ibl_available() { return aurora::gx::pbr_internal::sProbePbrIbl.available; }

const AuroraPbrIblStatus* aurora_get_pbr_ibl_status() {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  static AuroraPbrIblStatus status{};
  status = {};
  status.pbrEnabled = enablePbrMaterialOverride;
  status.iblEnabled = pbrIblParams.x() > 0.0f;
  status.requestedSource = sRequestedPbrIblSource;
  status.activeSource = active_pbr_ibl_source();
  status.fallbackAvailable = sFallbackPbrIbl.available;
  status.authoredAvailable = sAuthoredPbrIbl.available;
  status.probeAvailable = sProbePbrIbl.available;
  status.probeCameraValid = sProbeCamera.valid;
  status.probeCaptureResourcesReady = sProbeCaptureResourcesReady;
  status.probeCaptureInProgress = sProbeCaptureCamera.valid || sProbeCaptureFace != 0;
  status.probeRefreshPending = sProbeRefreshPending;
  status.probeFilterPending = sProbeFilterPending;
  status.probeAutoRefresh = sProbeAutoRefresh;
  status.probeCaptureFace = sProbeCaptureFace;
  status.probeCaptureDelayFrames = sProbeCaptureDelayFrames;
  status.probeFramesSinceRefresh = sProbeFramesSinceRefresh;
  status.probeCubeSize = PbrRuntimeProbeCubeSize;
  status.probeIrradianceSize = PbrIrradianceCubeSize;
  status.probePrefilterMipCount = PbrRuntimeProbePrefilterMipCount;
  status.activePrefilterMipCount = active_pbr_ibl_set().prefilterMipCount;
  status.authoredGlobalLoaded = sAuthoredPbrIblGlobalLoaded;
  status.authoredStageLoaded = sAuthoredPbrIblStageLoaded;
  status.authoredRoomLoaded = sAuthoredPbrIblRoomLoaded;
  status.authoredRoom = sAuthoredPbrIblRoom;
  copy_status_string(status.authoredRoot, sAuthoredPbrIblRootPath);
  copy_status_string(status.authoredStage, sAuthoredPbrIblStage);
  copy_status_string(status.authoredSceneKey, sActivePbrIblSceneKey);
  return &status;
}

void aurora_set_pbr_probe_auto_refresh(bool enabled) {
  aurora::gx::pbr_internal::sProbeAutoRefresh = enabled;
  if (enabled && !aurora::gx::pbr_internal::sProbePbrIbl.available) {
    aurora::gx::pbr_internal::sProbeRefreshPending = true;
  }
}

void aurora_request_pbr_probe_refresh() {
  aurora::gx::pbr_internal::sProbeCaptureFace = 0;
  aurora::gx::pbr_internal::sProbeCaptureDelayFrames = 0;
  aurora::gx::pbr_internal::sProbeRefreshPending = true;
}

void aurora_set_pbr_probe_camera_matrices(const float* source_view_3x4, const float* source_inv_view_3x4,
                                          const float* projection_4x4) {
  auto& camera = aurora::gx::pbr_internal::sProbeCamera;
  camera.sourceView = aurora::gx::pbr_internal::pbr_load_mtx3x4(source_view_3x4);
  camera.sourceInvView = aurora::gx::pbr_internal::pbr_load_mtx3x4(source_inv_view_3x4);
  camera.projection = aurora::gx::pbr_internal::pbr_projection_from_gx_mtx(projection_4x4);
  camera.valid = source_view_3x4 != nullptr && source_inv_view_3x4 != nullptr && projection_4x4 != nullptr;
  if (camera.valid) {
    aurora::gx::pbr_internal::pbr_update_probe_face_matrices();
    if (!aurora::gx::pbr_internal::sProbePbrIbl.available) {
      aurora::gx::pbr_internal::sProbeRefreshPending = true;
    } else if (aurora::gx::pbr_internal::sProbeAutoRefresh) {
      if (aurora::gx::pbr_internal::sProbeCaptureFace != 0 &&
          aurora::gx::pbr_internal::sProbeCaptureCamera.valid) {
        aurora::gx::pbr_internal::sProbeRefreshPending =
            aurora::gx::pbr_internal::sProbeRefreshPending ||
            aurora::gx::pbr_internal::pbr_probe_camera_changed(aurora::gx::pbr_internal::sProbeCaptureCamera, camera);
      } else if (aurora::gx::pbr_internal::pbr_probe_camera_changed(
                     aurora::gx::pbr_internal::sProbeLastCompletedCamera, camera)) {
        aurora::gx::pbr_internal::sProbeRefreshPending = true;
      }
    }
  }
}

void aurora_set_pbr_ibl_scene(const char* stage_name, int room_no) {
  const auto root = aurora::gx::pbr_internal::pbr_ibl_root_path();
  std::string stage = stage_name != nullptr ? stage_name : "";
  if (!stage.empty() && !aurora::gx::pbr_internal::pbr_is_safe_scene_component(stage)) {
    stage.clear();
  }

  const std::string key = fmt::format("{}:{}", stage, room_no);
  if (key == aurora::gx::pbr_internal::sActivePbrIblSceneKey) {
    return;
  }
  aurora::gx::pbr_internal::sActivePbrIblSceneKey = key;
  aurora::gx::pbr_internal::sAuthoredPbrIblRootPath = root.empty() ? std::string{} : root.string();
  aurora::gx::pbr_internal::sAuthoredPbrIblStage = stage;
  aurora::gx::pbr_internal::sAuthoredPbrIblRoom = room_no;
  aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded = false;
  aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded = false;
  aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded = false;

  aurora::gx::pbr_internal::load_fallback_pbr_ibl_textures(aurora::gx::pbr_internal::sAuthoredPbrIbl);
  aurora::gx::pbr_internal::sAuthoredPbrIbl.available = false;
  bool loaded = false;
  if (!root.empty()) {
    aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded =
        aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / "global",
                                                                       aurora::gx::pbr_internal::sAuthoredPbrIbl);
    loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded;
    if (!stage.empty()) {
      aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded =
          aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / stage,
                                                                         aurora::gx::pbr_internal::sAuthoredPbrIbl);
      loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded;
      if (room_no >= 0 &&
          (aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded =
               aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(
                   root / stage / fmt::format("room_{}", room_no), aurora::gx::pbr_internal::sAuthoredPbrIbl))) {
        loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded;
      }
    }
  }

  aurora::gx::pbr_internal::sAuthoredPbrIbl.available = loaded;
  aurora::gx::pbr_internal::sProbePbrIbl.available = false;
  aurora::gx::pbr_internal::sProbeCaptureFace = 0;
  aurora::gx::pbr_internal::sProbeCaptureDelayFrames = 0;
  aurora::gx::pbr_internal::sProbeLastCompletedCamera.valid = false;
  aurora::gx::pbr_internal::sProbeRefreshPending = true;
  if (loaded) {
    PbrLog.info("Loaded authored PBR IBL set for stage '{}' room {}", stage, room_no);
  }
  aurora::gx::pbr_internal::select_active_pbr_ibl_set();
}

void aurora_set_pbr_fill_dir(float x, float y, float z) {
  const float len = std::sqrt(x * x + y * y + z * z);
  const float inv = len > 1e-6f ? 1.0f / len : 1.0f;
  aurora::gx::pbrFillDir = {x * inv, y * inv, z * inv, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override(float roughness, float metallic, float ao, float specular,
                                               float emissive_r, float emissive_g, float emissive_b,
                                               float emissive_strength) {
  aurora::gx::pbrMaterialOverrideActive = true;
  aurora::gx::pbrMaterialFactors = {std::clamp(roughness, 0.04f, 1.0f), std::clamp(metallic, 0.0f, 1.0f),
                                    std::clamp(ao, 0.0f, 1.0f), std::clamp(specular, 0.0f, 1.0f)};
  aurora::gx::pbrMaterialEmissive = {std::max(emissive_r, 0.0f), std::max(emissive_g, 0.0f),
                                     std::max(emissive_b, 0.0f), std::max(emissive_strength, 0.0f)};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override_normal_map(const char* normal_map_name) {
  if (normal_map_name == nullptr || normal_map_name[0] == '\0') {
    aurora::gx::pbrMaterialNormal.reset();
    aurora::gx::g_gxState.stateDirty = true;
    return;
  }

  const auto normal = aurora::gfx::texture_replacement::find_named_pbr_texture(normal_map_name);
  aurora::gx::pbrMaterialNormal = normal.value_or(nullptr);
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override_maps(const char* material_name, bool use_rmaos, bool use_loose_maps,
                                                    bool use_normal, bool use_emissive) {
  aurora::gx::pbr_internal::clear_pbr_material_maps();

  if (material_name == nullptr || material_name[0] == '\0') {
    aurora::gx::g_gxState.stateDirty = true;
    return;
  }

  const std::string_view materialName{material_name};
  if (use_rmaos) {
    aurora::gx::pbrMaterialRmaos = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_rmaos"});
  }
  if (use_loose_maps) {
    aurora::gx::pbrMaterialRoughness =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_roughness", "_rough"});
    aurora::gx::pbrMaterialMetallic =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_metallic", "_metal"});
    aurora::gx::pbrMaterialAo = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_ao"});
    aurora::gx::pbrMaterialSpecular =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_specular", "_spec"});
  }
  if (use_normal) {
    aurora::gx::pbrMaterialNormal = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_normal", "_n"});
  }
  if (use_emissive) {
    aurora::gx::pbrMaterialEmissiveMap =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_emissive", "_e"});
  }

  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_clear_pbr_constant_material_override() {
  if (!aurora::gx::pbrMaterialOverrideActive) {
    return;
  }
  aurora::gx::pbrMaterialOverrideActive = false;
  aurora::gx::pbrMaterialFactors = {0.5f, 0.0f, 1.0f, 0.5f};
  aurora::gx::pbrMaterialEmissive = {0.0f, 0.0f, 0.0f, 0.0f};
  aurora::gx::pbr_internal::clear_pbr_material_maps();
  aurora::gx::g_gxState.stateDirty = true;
}
