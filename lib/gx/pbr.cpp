#include "pbr.hpp"

#include "../gfx/dds_io.hpp"
#include "../gfx/texture_replacement.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

static aurora::Module PbrLog("aurora::gx::pbr");

namespace aurora::gx {
using webgpu::g_device;
using webgpu::g_queue;

bool enablePbrMaterialOverride = false;
bool pbrMaterialOverrideActive = false;
Vec4<float> pbrParams{0.30f, 0.04f, 0.20f, 0.0f};
Vec4<float> pbrScales{1.0f, 1.0f, 0.0f, 0.0f};
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
constexpr u32 PbrBrdfLutSize = 128;
constexpr float PbrPi = 3.14159265358979323846f;
constexpr std::array<std::string_view, 6> PbrCubeFaceNames{"px", "nx", "py", "ny", "pz", "nz"};

wgpu::Sampler sPbrMaterialSampler;
wgpu::Sampler sPbrIblSampler;
wgpu::Sampler sPbrBrdfLutSampler;
wgpu::Texture sPbrIrradianceCubeTexture;
wgpu::TextureView sPbrIrradianceCubeView;
wgpu::Texture sPbrPrefilterCubeTexture;
wgpu::TextureView sPbrPrefilterCubeView;
wgpu::Texture sPbrBrdfLutTexture;
wgpu::TextureView sPbrBrdfLutView;
std::string sActivePbrIblSceneKey;

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

bool pbr_load_brdf_lut_from_directory(const std::filesystem::path& dir) noexcept {
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
  sPbrBrdfLutTexture = g_device.CreateTexture(&desc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = "PBR authored BRDF LUT view",
      .format = lut->format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  sPbrBrdfLutView = sPbrBrdfLutTexture.CreateView(&viewDesc);
  pbr_write_texture_layer_mip(sPbrBrdfLutTexture, 0, 0, *lut);
  return true;
}

void pbr_update_ibl_max_mip(u32 mipCount) noexcept {
  const float maxMip = static_cast<float>(std::max(mipCount, 1u) - 1u);
  pbrIblParams = {pbrIblParams.x(), pbrIblParams.y(), pbrIblParams.z(), maxMip};
}

void load_fallback_pbr_ibl_textures() noexcept {
  const wgpu::TextureDescriptor irradianceDesc{
      .label = "PBR fallback irradiance cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrIrradianceCubeSize, .height = PbrIrradianceCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  sPbrIrradianceCubeTexture = g_device.CreateTexture(&irradianceDesc);
  const wgpu::TextureViewDescriptor irradianceViewDesc{
      .label = "PBR fallback irradiance cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  sPbrIrradianceCubeView = sPbrIrradianceCubeTexture.CreateView(&irradianceViewDesc);
  pbr_fill_cube_texture(sPbrIrradianceCubeTexture, PbrIrradianceCubeSize, 1, false);

  const wgpu::TextureDescriptor prefilterDesc{
      .label = "PBR fallback prefiltered specular cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrPrefilterCubeSize, .height = PbrPrefilterCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = PbrPrefilterMipCount,
      .sampleCount = 1,
  };
  sPbrPrefilterCubeTexture = g_device.CreateTexture(&prefilterDesc);
  const wgpu::TextureViewDescriptor prefilterViewDesc{
      .label = "PBR fallback prefiltered specular cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = PbrPrefilterMipCount,
      .arrayLayerCount = 6,
  };
  sPbrPrefilterCubeView = sPbrPrefilterCubeTexture.CreateView(&prefilterViewDesc);
  pbr_fill_cube_texture(sPbrPrefilterCubeTexture, PbrPrefilterCubeSize, PbrPrefilterMipCount, true);
  pbr_update_ibl_max_mip(PbrPrefilterMipCount);

  const wgpu::TextureDescriptor brdfDesc{
      .label = "PBR BRDF LUT",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrBrdfLutSize, .height = PbrBrdfLutSize, .depthOrArrayLayers = 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  sPbrBrdfLutTexture = g_device.CreateTexture(&brdfDesc);
  const wgpu::TextureViewDescriptor brdfViewDesc{
      .label = "PBR BRDF LUT view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  sPbrBrdfLutView = sPbrBrdfLutTexture.CreateView(&brdfViewDesc);

  const auto brdfLut = pbr_generate_brdf_lut();
  pbr_write_texture_layer_mip(sPbrBrdfLutTexture, 0, 0, PbrBrdfLutSize, PbrBrdfLutSize, brdfLut);
}

bool load_authored_pbr_ibl_from_directory(const std::filesystem::path& dir) noexcept {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return false;
  }

  bool loadedAny = false;
  u32 prefilterMipCount = PbrPrefilterMipCount;
  loadedAny |= pbr_load_cube_from_directory(dir, "irradiance", false, sPbrIrradianceCubeTexture,
                                            sPbrIrradianceCubeView, prefilterMipCount);
  if (pbr_load_cube_from_directory(dir, "prefilter", true, sPbrPrefilterCubeTexture, sPbrPrefilterCubeView,
                                   prefilterMipCount)) {
    pbr_update_ibl_max_mip(prefilterMipCount);
    loadedAny = true;
  }
  loadedAny |= pbr_load_brdf_lut_from_directory(dir);
  return loadedAny;
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

  load_fallback_pbr_ibl_textures();
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
  entries[pbr_ibl_irradiance_texture_binding(0)] = {
      .binding = pbr_ibl_irradiance_texture_binding(0),
      .textureView = sPbrIrradianceCubeView,
  };
  entries[pbr_ibl_irradiance_sampler_binding(0)] = {
      .binding = pbr_ibl_irradiance_sampler_binding(0),
      .sampler = sPbrIblSampler,
  };
  entries[pbr_ibl_prefilter_texture_binding(0)] = {
      .binding = pbr_ibl_prefilter_texture_binding(0),
      .textureView = sPbrPrefilterCubeView,
  };
  entries[pbr_ibl_prefilter_sampler_binding(0)] = {
      .binding = pbr_ibl_prefilter_sampler_binding(0),
      .sampler = sPbrIblSampler,
  };
  entries[pbr_ibl_brdf_lut_texture_binding(0)] = {
      .binding = pbr_ibl_brdf_lut_texture_binding(0),
      .textureView = sPbrBrdfLutView,
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
  const bool usePrevAlbedo = (info.pbrFlags & PbrMaterialUsePrevAlbedo) != 0;
  if (usePrevAlbedo) {
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
  bindPbrView(pbr_ibl_irradiance_texture_binding(0), pbr_ibl_irradiance_sampler_binding(0), sPbrIrradianceCubeView,
              sPbrIblSampler);
  bindPbrView(pbr_ibl_prefilter_texture_binding(0), pbr_ibl_prefilter_sampler_binding(0), sPbrPrefilterCubeView,
              sPbrIblSampler);
  bindPbrView(pbr_ibl_brdf_lut_texture_binding(0), pbr_ibl_brdf_lut_sampler_binding(0), sPbrBrdfLutView,
              sPbrBrdfLutSampler);
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
    config.pbrFlags = PbrMaterialEnabled | PbrMaterialUsePrevAlbedo | PbrMaterialPrevAlbedoIsLit;
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

    config.pbrFlags = PbrMaterialEnabled;
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
  aurora::gx::pbrScales = {diffuse_scale, specular_scale, 0.0f, 0.0f};
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

  aurora::gx::pbr_internal::load_fallback_pbr_ibl_textures();
  bool loaded = false;
  if (!root.empty()) {
    if (aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / "global")) {
      loaded = true;
    }
    if (!stage.empty()) {
      if (aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / stage)) {
        loaded = true;
      }
      if (room_no >= 0 &&
          aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / stage / fmt::format("room_{}", room_no))) {
        loaded = true;
      }
    }
  }

  if (loaded) {
    PbrLog.info("Loaded authored PBR IBL set for stage '{}' room {}", stage, room_no);
  }
  aurora::gx::g_gxState.stateDirty = true;
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
