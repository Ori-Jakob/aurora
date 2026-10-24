#include "self_test.hpp"

#include "__gx.h"
#include "gx/fifo.hpp"
#include "gx/gpu_channel.hpp"
#include "gx/guest_memory.hpp"
#include "trace/capture_session.hpp"

#include <aurora/gpu_systems.hpp>
#include <dolphin/gx.h>
#include <fmt/format.h>

#include <bit>
#include <cmath>
#include <numbers>

namespace aurora::gxreplay {
namespace {
constexpr std::array<uint8_t, 16> HintPattern{0x61, 0x45, 0x00, 0x00, 0x02, 0x50, 0x00, 0x01,
                                              0x80, 0x00, 0x04, 0x10, 0x00, 0x05, 0x10, 0x1A};

struct Quad {
  float left;
  float top;
  float right;
  float bottom;
  uint32_t probeX;
  uint32_t probeY;
};
constexpr std::array<Quad, 3> Quads{{
    {64.f, 64.f, 320.f, 288.f, 192, 176},
    {352.f, 64.f, 576.f, 288.f, 464, 176},
    {64.f, 304.f, 320.f, 464.f, 192, 384},
}};
static_assert(Quads.size() == SelfTestObjects.size());

struct Rgb {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  bool operator==(const Rgb&) const = default;
};

Rgb clear_color(uint32_t frame) noexcept { return {static_cast<uint8_t>(32 + 64 * frame), 48, 96}; }

// The EFB is cleared at frame begin with the copy clear the previous frame set; frame 0 sees the GXState default
Rgb background(uint32_t frame) noexcept { return frame == 0 ? Rgb{0, 0, 0} : clear_color(frame - 1); }

const void* guest_pointer(uint64_t address) noexcept {
  return reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
}

void append_be(std::vector<uint8_t>& out, float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<uint8_t>(bits >> shift));
  }
}

void append_le(std::vector<uint8_t>& out, float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  for (int shift = 0; shift <= 24; shift += 8) {
    out.push_back(static_cast<uint8_t>(bits >> shift));
  }
}

std::vector<uint8_t> quad_positions(bool littleEndian = false) {
  std::vector<uint8_t> out;
  for (const auto& quad : Quads) {
    const std::array<std::array<float, 2>, 4> corners{{
        {quad.left, quad.top},
        {quad.right, quad.top},
        {quad.right, quad.bottom},
        {quad.left, quad.bottom},
    }};
    for (const auto& corner : corners) {
      for (const float value : {corner[0], corner[1], -0.5f}) {
        if (littleEndian) {
          append_le(out, value);
        } else {
          append_be(out, value);
        }
      }
    }
  }
  return out;
}

std::vector<uint8_t> quad_colors(uint32_t frame) {
  static constexpr std::array<std::array<uint8_t, 4>, 3> Steady{{
      {255, 255, 255, 255},
      {255, 255, 255, 255},
      {255, 255, 255, 255},
  }};
  static constexpr std::array<std::array<uint8_t, 4>, 3> Changed{{
      {255, 96, 64, 255},
      {64, 255, 96, 255},
      {96, 64, 255, 255},
  }};
  const auto& colors = frame == 2 ? Changed : Steady;
  std::vector<uint8_t> out;
  for (const auto& color : colors) {
    for (int vertex = 0; vertex < 4; ++vertex) {
      out.insert(out.end(), color.begin(), color.end());
    }
  }
  return out;
}

std::vector<uint8_t> checker_texels() {
  std::vector<uint8_t> out(256);
  for (uint32_t block = 0; block < 4; ++block) {
    const uint32_t blockX = block % 2;
    const uint32_t blockY = block / 2;
    for (uint32_t texel = 0; texel < 16; ++texel) {
      const uint32_t x = blockX * 4 + texel % 4;
      const uint32_t y = blockY * 4 + texel / 4;
      const uint8_t value = ((x ^ y) & 1) != 0 ? 255 : 128;
      uint8_t* base = out.data() + block * 64;
      base[texel * 2] = 255;
      base[texel * 2 + 1] = value;
      base[32 + texel * 2] = value;
      base[32 + texel * 2 + 1] = value;
    }
  }
  return out;
}

GXTexObj make_texture(uint64_t address, u16 width, u16 height) {
  GXTexObj obj;
  GXInitTexObj(&obj, guest_pointer(address), width, height, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
  GXInitTexObjLOD(&obj, GX_NEAR, GX_NEAR, 0.f, 0.f, 0.f, GX_FALSE, GX_FALSE, GX_ANISO_1);
  return obj;
}

// Built once so both variants carry identical texture object ids
GXTexObj checker_texture() {
  static const GXTexObj obj = make_texture(SelfTestTextureAddress, 8, 8);
  return obj;
}

GXTexObj copy_texture() {
  static const GXTexObj obj = make_texture(SelfTestCopyAddress, 256, 224);
  return obj;
}

void write_hints(bool enabled, uint32_t size) {
  if (!enabled) {
    return;
  }
  static const std::vector<uint8_t> payload = [] {
    std::vector<uint8_t> bytes(SelfTestPayloadSizes.back());
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = self_test_hint_byte(i);
    }
    return bytes;
  }();
  for (const uint16_t id : SelfTestHintIds) {
    GX_WRITE_AURORA_HINT(id, payload.data(), size);
  }
}

void write_frame_hints(bool enabled) {
  if (!enabled) {
    return;
  }
  const SelfTestCamera camera = self_test_camera();
  GXAuroraSetCamera(AURORA_CAMERA_MAIN, camera.view, camera.proj, GX_PERSPECTIVE, 0);
  GXAuroraSetSceneEpoch(1);
  const AuroraLight light = self_test_light();
  GXAuroraSetLights(&light, 1);
  const AuroraEnvironment environment = self_test_environment();
  GXAuroraSetEnvironment(&environment);
  const AuroraInfluencer influencer = self_test_influencer();
  GXAuroraSetInfluencers(&influencer, 1);
  const AuroraWindSource windSource = self_test_wind_source();
  GXAuroraSetWindSources(&windSource, 1);
  GXAuroraSetArrayImmutable(guest_pointer(SelfTestPositionAddress), 144, GX_FALSE, 1);
  GXAuroraReleaseArrayImmutable(guest_pointer(SelfTestPositionAddress), 1);
  const AuroraSky sky = self_test_sky();
  GXAuroraSetSky(&sky);
  const AuroraFogVolume fogVolume = self_test_fog_volume();
  GXAuroraSetFogVolumes(&fogVolume, 1);
  const AuroraGrassLightSet lightSet = self_test_grass_light_set();
  GXAuroraSetGrassPatches(0, 0, 0.5f, &lightSet, 1);
  GXAuroraStage(AURORA_STAGE_SCENE_BEGIN);
}

void begin_quad_object(bool enabled, size_t quad) {
  if (!enabled) {
    return;
  }
  if (quad == 2) {
    GXAuroraPushScope(AURORA_CAT_EMISSIVE, 0, AURORA_CAMERA_INHERIT);
  }
  const SelfTestObject& object = SelfTestObjects[quad];
  GXAuroraBeginObject(object.id, 0, 1, object.categories, 0);
  if (quad == 0) {
    GXAuroraSetMaterialClass(1);
    const AuroraTevPolicy policy = self_test_tev_policy();
    GXAuroraSetTevPolicy(&policy);
    const f32 joint[1][3][4]{{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}}};
    GXAuroraSetSkeleton(object.id, joint, 1);
  }
}

void end_quad_object(bool enabled, size_t quad) {
  if (!enabled) {
    return;
  }
  GXAuroraEndObject();
  if (quad == 2) {
    GXAuroraPopScope();
  }
}

// After the last quad, so every quad draws in the Opaque window
void write_markers(bool enabled) {
  if (!enabled) {
    return;
  }
  for (const AuroraRenderStage stage : {AURORA_STAGE_OPAQUE_DONE, AURORA_STAGE_PRE_POST, AURORA_STAGE_SCENE_END,
                                        AURORA_STAGE_BEFORE_HUD, AURORA_STAGE_FRAME_END}) {
    GXAuroraStage(stage);
  }
}

void draw_quad(uint8_t firstVertex) {
  static constexpr std::array<std::array<float, 2>, 4> TexCoords{{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}}};
  GXBegin(GX_QUADS, GX_VTXFMT0, 4);
  for (uint8_t i = 0; i < 4; ++i) {
    GXPosition1x8(firstVertex + i);
    GXColor1x8(firstVertex + i);
    GXTexCoord2f32(TexCoords[i][0], TexCoords[i][1]);
  }
  GXEnd();
}

std::vector<uint8_t> cut_fifo() {
  const uint8_t* data = gx::fifo::get_buffer_data();
  std::vector<uint8_t> bytes(data, data + gx::fifo::get_buffer_size());
  gx::fifo::clear_buffer();
  return bytes;
}

void emit_state(uint32_t frame, SelfTestMode mode) {
  const bool withHints = mode == SelfTestMode::Hints;
  if (frame == 0) {
    GXInit(nullptr, 0);
  }
  write_hints(withHints, SelfTestPayloadSizes[0]);
  write_hints(withHints, SelfTestPayloadSizes[1]);
  write_frame_hints(withHints);

  GXSetViewport(0.f, 0.f, static_cast<f32>(SelfTestWidth), static_cast<f32>(SelfTestHeight), 0.f, 1.f);
  GXSetScissor(0, 0, SelfTestWidth, SelfTestHeight);
  // C_MTXOrtho(top 0, bottom 480, left 0, right 640, near 0, far 1) without a mtx dependency
  f32 ortho[4][4]{};
  ortho[0][0] = 2.f / static_cast<f32>(SelfTestWidth);
  ortho[0][3] = -1.f;
  ortho[1][1] = -2.f / static_cast<f32>(SelfTestHeight);
  ortho[1][3] = 1.f;
  ortho[2][2] = -1.f;
  ortho[2][3] = -1.f;
  ortho[3][3] = 1.f;
  GXSetProjection(ortho, GX_ORTHOGRAPHIC);
  f32 identity[3][4]{};
  identity[0][0] = identity[1][1] = identity[2][2] = 1.f;
  GXLoadPosMtxImm(identity, GX_PNMTX0);
  GXSetCurrentMtx(GX_PNMTX0);

  const Rgb clear = clear_color(frame);
  GXSetCopyClear({clear.r, clear.g, clear.b, 255}, GX_MAX_Z24);
  GXSetZMode(GX_FALSE, GX_ALWAYS, GX_FALSE);
  GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
  GXSetCullMode(GX_CULL_NONE);
  GXSetNumChans(1);
  GXSetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_REG, GX_SRC_VTX, GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
  GXSetNumTexGens(1);
  GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
  GXSetNumTevStages(1);
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
  GXSetTevOp(GX_TEVSTAGE0, GX_MODULATE);

  GXTexObj checker = checker_texture();
  GXLoadTexObj(&checker, GX_TEXMAP0);
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX8);
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
  if (mode == SelfTestMode::GpuArrays) {
    const auto positions = quad_positions(true);
    const auto colors = quad_colors(frame);
    const auto positionHandle = render::gpu::upload(positions.data(), static_cast<uint32_t>(positions.size()));
    const auto colorHandle = render::gpu::upload(colors.data(), static_cast<uint32_t>(colors.size()));
    GXSetArrayGPU(GX_VA_POS, positionHandle, 0, 12);
    GXSetArrayGPU(GX_VA_CLR0, colorHandle, 0, 4);
  } else {
    GXSetArray(GX_VA_POS, guest_pointer(SelfTestPositionAddress), 144, 12, false);
    GXSetArray(GX_VA_CLR0, guest_pointer(SelfTestColorAddress), 48, 4, false);
  }
}

void emit_draws(bool withHints) {
  begin_quad_object(withHints, 0);
  draw_quad(0);
  end_quad_object(withHints, 0);

  alignas(32) std::array<uint8_t, 512> displayList{};
  GXBeginDisplayList(displayList.data(), static_cast<u32>(displayList.size()));
  write_hints(withHints, SelfTestPayloadSizes[2]);
  begin_quad_object(withHints, 1);
  draw_quad(4);
  end_quad_object(withHints, 1);
  const u32 displayListSize = GXEndDisplayList();
  GXCallDisplayList(displayList.data(), displayListSize);

  write_hints(withHints, SelfTestPayloadSizes[3]);
  GXSetTexCopySrc(64, 64, 256, 224);
  GXSetTexCopyDst(256, 224, GX_TF_RGBA8, GX_FALSE);
  GXCopyTex(const_cast<void*>(guest_pointer(SelfTestCopyAddress)), GX_FALSE);
  GXTexObj copy = copy_texture();
  GXLoadTexObj(&copy, GX_TEXMAP0);
  begin_quad_object(withHints, 2);
  draw_quad(8);
  end_quad_object(withHints, 2);
  write_markers(withHints);
}

Rgb pixel(const std::vector<uint8_t>& image, uint32_t x, uint32_t y) {
  const size_t offset = (static_cast<size_t>(y) * SelfTestWidth + x) * 4;
  return {image[offset], image[offset + 1], image[offset + 2]};
}

std::string describe(const Rgb& color) { return fmt::format("({}, {}, {})", color.r, color.g, color.b); }
} // namespace

trace::TraceInfo self_test_info() {
  return {
      .msaaSamples = 1,
      .maxAnisotropy = 16,
      .requestedFrames = SelfTestFrameCount,
      .label = "self-test",
  };
}

uint8_t self_test_hint_byte(size_t index) noexcept { return HintPattern[index % HintPattern.size()]; }

SelfTestCamera self_test_camera() noexcept {
  // C_MTXPerspective(60, 4:3, 1, 1000) without a mtx dependency; the quads keep their ortho projection
  constexpr f32 Near = 1.f;
  constexpr f32 Far = 1000.f;
  const f32 cot = 1.f / std::tan(30.f * std::numbers::pi_v<f32> / 180.f);
  SelfTestCamera camera{};
  camera.view[0][0] = camera.view[1][1] = camera.view[2][2] = 1.f;
  camera.proj[0][0] = cot / (4.f / 3.f);
  camera.proj[1][1] = cot;
  camera.proj[2][2] = -Near / (Far - Near);
  camera.proj[2][3] = -(Far * Near) / (Far - Near);
  camera.proj[3][2] = -1.f;
  return camera;
}

AuroraLight self_test_light() noexcept {
  return {
      .id = 1,
      .type = AURORA_LIGHT_POINT,
      .position = {1.f, 2.f, 3.f},
      .color = {1.f, 0.5f, 0.25f},
      .radius = 220.f,
      .spotInnerCos = 1.f,
      .spotOuterCos = -1.f,
      .falloffExponent = 1.5f,
      .volumetricScale = 1.f,
  };
}

AuroraEnvironment self_test_environment() noexcept {
  AuroraEnvironment environment{
      .skyVisibility = 1.f,
      .timeOfDay = 12.f,
      .flags = AURORA_ENV_RAIN,
      .unitsPerMeter = 100.f,
      .fogType = GX_FOG_PERSP_LIN,
      .fogStart = 40.f,
      .fogEnd = 160.f,
      .viewNear = 0.5f,
      .viewFar = 400.f,
      .grassDensityScale = 1.f,
      .hazeAtFogEnd = -1.f,
      .hazeDensity = -1.f,
      .mediaAlbedo = {-1.f, -1.f, -1.f},
      .phaseG = -2.f,
      .phaseG2 = -2.f,
      .phaseBlend = -1.f,
      .noiseScale = -1.f,
      .noiseAmount = -1.f,
      .relightStrength = -1.f,
      .relightAmbient = -1.f,
      .sunGlow = -1.f,
      .volumeFar = -1.f,
  };
  for (size_t i = 0; i < std::size(environment.fogRangeTable); ++i) {
    environment.fogRangeTable[i] = static_cast<u16>(i + 1);
  }
  return environment;
}

AuroraTevPolicy self_test_tev_policy() noexcept {
  return {.mask = AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K1, AURORA_TEV_CH_R)};
}

AuroraInfluencer self_test_influencer() noexcept {
  return {.pos = {1.f, 2.f, 3.f}, .radius = 1.5f, .kind = AURORA_INFLUENCER_PUSH};
}

AuroraWindSource self_test_wind_source() noexcept { return {.radius = 50.f, .dir = {1.f, 0.f, 0.f}, .power = 0.3f}; }

AuroraGrassLightSet self_test_grass_light_set() noexcept { return {.grassLightRatio = 1.f}; }

AuroraSky self_test_sky() noexcept {
  AuroraSky sky{.flags = AURORA_SKY_VISIBLE, .horizonY = 0.5f, .layerCount = 2};
  sky.layers[0].base = 1.f;
  sky.layers[1].flags = AURORA_CLOUD_LAYER_SEA;
  return sky;
}

AuroraFogVolume self_test_fog_volume() noexcept {
  return {
      .center = {-18.f, 1.f, 14.f},
      .radius = {6.f, 1.5f, 6.f},
      .shape = AURORA_FOG_SHAPE_ELLIPSOID,
      .mode = AURORA_FOG_MODE_ADD,
      .density = 0.05f,
  };
}

bool write_self_test_trace(const std::filesystem::path& path, SelfTestMode mode, std::string& error) {
  const bool withHints = mode == SelfTestMode::Hints;
  const bool gpuArrays = mode == SelfTestMode::GpuArrays;
  trace::CaptureSession session;
  if (!session.open(path, 0, trace::FileSynthetic, error) ||
      !session.write_header_chunks(self_test_info(), {}, error)) {
    session.discard();
    return false;
  }

  gx::fifo::clear_buffer();
  const auto positions = quad_positions();
  const auto checker = checker_texels();
  if (gpuArrays) {
    gx::gpu_channel::end_frame();
  }
  for (uint32_t frame = 0; frame < SelfTestFrameCount; ++frame) {
    session.begin_frame(frame, {
                                   .efbWidth = SelfTestWidth,
                                   .efbHeight = SelfTestHeight,
                                   .viFbWidth = SelfTestWidth,
                                   .viEfbHeight = SelfTestHeight,
                                   .viewportPolicy = AURORA_VIEWPORT_NATIVE,
                               });
    emit_state(frame, mode);
    session.processed(cut_fifo());
    if (!gpuArrays) {
      session.observe(static_cast<uint8_t>(gx::guest_memory::Kind::VertexArray), SelfTestPositionAddress, positions);
      session.observe(static_cast<uint8_t>(gx::guest_memory::Kind::VertexArray), SelfTestColorAddress,
                      quad_colors(frame));
    }
    session.observe(static_cast<uint8_t>(gx::guest_memory::Kind::Texture), SelfTestTextureAddress, checker);
    emit_draws(withHints);
    session.processed(cut_fifo());
    if (gpuArrays) {
      gx::gpu_channel::end_frame();
    }
    if (!session.end_frame(error)) {
      session.discard();
      return false;
    }
  }
  if (!session.finalize(0, {}, error)) {
    session.discard();
    return false;
  }
  return true;
}

bool check_self_test_frames(std::span<const std::vector<uint8_t>> withHints,
                            std::span<const std::vector<uint8_t>> plain, bool pixelProbes, std::string& error) {
  if (withHints.size() != SelfTestFrameCount || plain.size() != SelfTestFrameCount) {
    error = fmt::format("expected {} frames from each self-test trace, got {} and {}", SelfTestFrameCount,
                        withHints.size(), plain.size());
    return false;
  }
  for (uint32_t frame = 0; frame < SelfTestFrameCount; ++frame) {
    if (withHints[frame] != plain[frame]) {
      error = fmt::format("frame {} differs between the hint and plain traces", frame);
      return false;
    }
  }
  if (!pixelProbes) {
    return true;
  }

  constexpr size_t ImageSize = static_cast<size_t>(SelfTestWidth) * SelfTestHeight * 4;
  for (uint32_t frame = 0; frame < SelfTestFrameCount; ++frame) {
    const auto& image = withHints[frame];
    if (image.size() != ImageSize) {
      error = fmt::format("frame {} is {} bytes, expected a {}x{} RGBA8 image", frame, image.size(), SelfTestWidth,
                          SelfTestHeight);
      return false;
    }
    const Rgb expected = background(frame);
    if (const Rgb corner = pixel(image, 8, 8); corner != expected) {
      error = fmt::format("frame {}: pixel (8, 8) is {} but the clear color is {}", frame, describe(corner),
                          describe(expected));
      return false;
    }
    for (size_t quad = 0; quad < Quads.size(); ++quad) {
      const auto& probe = Quads[quad];
      if (pixel(image, probe.probeX, probe.probeY) == expected) {
        error = fmt::format("frame {}: quad {} is missing at ({}, {})", frame, static_cast<char>('A' + quad),
                            probe.probeX, probe.probeY);
        return false;
      }
    }
  }
  const auto& quadA = Quads[0];
  if (pixel(withHints[2], quadA.probeX, quadA.probeY) == pixel(withHints[1], quadA.probeX, quadA.probeY)) {
    error = "quad A did not change color in frame 2";
    return false;
  }
  return true;
}

bool check_self_test_view_frames(std::span<const std::vector<uint8_t>> images,
                                 std::span<const SelfTestQuadColors> quadColors, std::string& error) {
  if (images.size() != SelfTestFrameCount || quadColors.size() != SelfTestFrameCount) {
    error = fmt::format("expected {} frames and colour sets, got {} and {}", SelfTestFrameCount, images.size(),
                        quadColors.size());
    return false;
  }
  constexpr size_t ImageSize = static_cast<size_t>(SelfTestWidth) * SelfTestHeight * 4;
  for (uint32_t frame = 0; frame < SelfTestFrameCount; ++frame) {
    const auto& image = images[frame];
    if (image.size() != ImageSize) {
      error = fmt::format("frame {} is {} bytes, expected a {}x{} RGBA8 image", frame, image.size(), SelfTestWidth,
                          SelfTestHeight);
      return false;
    }
    const Rgb expected = background(frame);
    if (const Rgb corner = pixel(image, 8, 8); corner != expected) {
      error = fmt::format("frame {}: pixel (8, 8) is {} but the clear color is {}", frame, describe(corner),
                          describe(expected));
      return false;
    }
    for (size_t quad = 0; quad < Quads.size(); ++quad) {
      const auto& probe = Quads[quad];
      const SelfTestColor& color = quadColors[frame][quad];
      const Rgb want{color.r, color.g, color.b};
      if (const Rgb actual = pixel(image, probe.probeX, probe.probeY); actual != want) {
        error =
            fmt::format("frame {}: quad {} at ({}, {}) is {} but its view colour is {}", frame,
                        static_cast<char>('A' + quad), probe.probeX, probe.probeY, describe(actual), describe(want));
        return false;
      }
    }
  }
  return true;
}
} // namespace aurora::gxreplay
