// Render hint wire format: emitters, golden bytes, core decode and validation

#include "gx_test_common.hpp"
#include "__gx.h"
#include "../tools/gxreplay/self_test.hpp"
#include "gx/draw_route.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace hints = aurora::gx::hints;
namespace wire = aurora::gx::hints::wire;
namespace route = aurora::gx::draw_route;
using aurora::gx::g_gxState;
using aurora::render::Stage;

// No other translation unit includes the router header, so its declared shape is pinned here
static_assert(std::is_same_v<decltype(route::Router::push),
                             bool (*)(void*, aurora::gx::DrawData&, const aurora::gx::PipelineConfig&,
                                      const aurora::gx::ShaderInfo&) noexcept>);
static_assert(std::is_same_v<decltype(route::Router::route), route::Decision (*)(void*, const route::Query&) noexcept>);
static_assert(std::is_standard_layout_v<route::Router>);
static_assert(offsetof(route::Router, route) < offsetof(route::Router, push) &&
              offsetof(route::Router, push) < offsetof(route::Router, userData));
static_assert(route::Router{}.route == nullptr && route::Router{}.push == nullptr &&
              route::Router{}.userData == nullptr);
static_assert(static_cast<uint8_t>(route::Route::Push) == 0 && static_cast<uint8_t>(route::Route::Suppress) == 1);
static_assert(route::Decision{}.route == route::Route::Push && !route::Decision{}.breakMerge);
static_assert(route::Query{}.prim == GX_TRIANGLES && route::Query{}.fmt == GX_VTXFMT0 && route::Query{}.vtxCount == 0 &&
              !route::Query{}.indexed);

namespace {
constexpr std::array<u8, 5> kRealBpWrite{GX_LOAD_BP_REG, 0x41, 0x12, 0x34, 0x56};
const void* const kArray = reinterpret_cast<const void*>(uintptr_t{0x80200000});

std::vector<u8> hex(std::string_view text) {
  std::vector<u8> out;
  for (size_t i = 0; i < text.size();) {
    if (text[i] == ' ') {
      ++i;
      continue;
    }
    out.push_back(static_cast<u8>(std::stoul(std::string{text.substr(i, 2)}, nullptr, 16)));
    i += 2;
  }
  return out;
}

std::vector<u8> operator+(std::vector<u8> lhs, const std::vector<u8>& rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

std::vector<u8> zeros(size_t count) { return std::vector<u8>(count, 0); }

std::vector<u8> raw_hint(u16 id, const std::vector<u8>& payload) {
  std::vector<u8> out(GX_AURORA_HINT_HEADER_SIZE + payload.size());
  wire::Writer writer{out};
  writer.header(id, static_cast<u32>(payload.size()));
  writer.bytes(payload);
  return out;
}

// Payload bytes of a single hint, header stripped
std::vector<u8> payload_of(const std::vector<u8>& hint) {
  return {hint.begin() + GX_AURORA_HINT_HEADER_SIZE, hint.end()};
}

u32 header_size(const std::vector<u8>& hint) {
  return static_cast<u32>(hint[3]) << 24 | static_cast<u32>(hint[4]) << 16 | static_cast<u32>(hint[5]) << 8 | hint[6];
}

std::vector<u8> set_array_payload(u16 version, u64 address, u32 size, u8 le) {
  std::vector<u8> out(wire::SetArrayImmutableV1);
  wire::Writer writer{out};
  writer.u16(version);
  writer.u64(address);
  writer.u32(size);
  writer.u8(le);
  writer.u32(1);
  return out;
}

std::vector<AuroraLight> make_lights(size_t count) {
  std::vector<AuroraLight> lights(count);
  for (size_t i = 0; i < count; ++i) {
    auto& light = lights[i];
    light = aurora::gxreplay::self_test_light();
    light.id = static_cast<u32>(i + 1);
    light.type = static_cast<u8>(i % 3);
    light.shadowPriority = static_cast<u8>(i);
    light.flags = static_cast<u16>(AURORA_LIGHT_FLICKER | i);
    light.categoryMask = static_cast<u32>(i * 7);
    light.position[2] = static_cast<f32>(i);
    light.direction[0] = 0.25f * static_cast<f32>(i);
    light.sourceRadius = 0.5f;
    light.volumetricScale = 2.f;
  }
  return lights;
}

AuroraTevPolicy busy_policy() {
  AuroraTevPolicy policy{.mask = AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_C0, AURORA_TEV_CH_G) |
                                 AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K3, AURORA_TEV_CH_A)};
  for (size_t reg = 0; reg < 7; ++reg) {
    for (size_t channel = 0; channel < 4; ++channel) {
      policy.neutral[reg][channel] = static_cast<u8>(reg * 4 + channel + 1);
    }
  }
  return policy;
}

const f32 kWorld[3][4]{{1.f, 0.f, 0.f, 10.f}, {0.f, 1.f, 0.f, -20.f}, {0.f, 0.f, 1.f, 30.f}};
const f32 kJoint[1][3][4]{{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}}};

struct EmitterCase {
  u16 id;
  u32 payloadSize;
  std::function<void()> emit;
  // A second emitter for an ID that already has one
  bool extra = false;
};

std::vector<EmitterCase> emitter_cases() {
  using namespace aurora::gxreplay;
  return {
      {GX_AURORA_SET_ARRAY_IMMUTABLE, wire::SetArrayImmutableV1,
       [] { GXAuroraSetArrayImmutable(kArray, 0x90, GX_TRUE, 2); }},
      {GX_AURORA_RELEASE_ARRAY_IMMUTABLE, wire::ReleaseArrayImmutableV1,
       [] { GXAuroraReleaseArrayImmutable(kArray, 2); }},
      {GX_AURORA_RENDER_CAMERA, wire::CameraV1,
       [] {
         const auto camera = self_test_camera();
         GXAuroraSetCamera(AURORA_CAMERA_MAIN, camera.view, camera.proj, GX_PERSPECTIVE, 3);
       }},
      {GX_AURORA_RENDER_OBJECT_BEGIN, wire::ObjectBeginV1,
       [] { GXAuroraBeginObject(0x10, 0, 1, AURORA_CAT_TERRAIN, 0); }},
      {GX_AURORA_RENDER_OBJECT_BEGIN, wire::ObjectBeginV2,
       [] { GXAuroraBeginObjectWorld(0x10, 2, 3, AURORA_CAT_TERRAIN, AURORA_OBJ_IMMOBILE, kWorld); }, true},
      {GX_AURORA_RENDER_OBJECT_END, wire::EmptyV1, [] { GXAuroraEndObject(); }},
      {GX_AURORA_RENDER_SCOPE_PUSH, wire::ScopePushV1,
       [] { GXAuroraPushScope(AURORA_CAT_EMISSIVE, 0, AURORA_CAMERA_INHERIT); }},
      {GX_AURORA_RENDER_SCOPE_POP, wire::EmptyV1, [] { GXAuroraPopScope(); }},
      {GX_AURORA_RENDER_STAGE, wire::StageV1, [] { GXAuroraStage(AURORA_STAGE_OPAQUE_DONE); }},
      {GX_AURORA_RENDER_LIGHTS, wire::CountedHeader + 2 * wire::LightV2,
       [] {
         const auto lights = make_lights(2);
         GXAuroraSetLights(lights.data(), 2);
       }},
      {GX_AURORA_RENDER_ENVIRONMENT, wire::EnvironmentV2,
       [] {
         const auto environment = self_test_environment();
         GXAuroraSetEnvironment(&environment);
       }},
      {GX_AURORA_RENDER_MATERIAL, wire::MaterialV1, [] { GXAuroraSetMaterialClass(1); }},
      {GX_AURORA_RENDER_SKELETON, wire::SkeletonHeader + wire::Joint, [] { GXAuroraSetSkeleton(0x10, kJoint, 1); }},
      {GX_AURORA_RENDER_SCENE_EPOCH, wire::SceneEpochV1, [] { GXAuroraSetSceneEpoch(1); }},
      {GX_AURORA_RENDER_TEV_POLICY, wire::TevPolicyV1,
       [] {
         const auto policy = self_test_tev_policy();
         GXAuroraSetTevPolicy(&policy);
       }},
      {GX_AURORA_RENDER_INFLUENCERS, wire::CountedHeader + wire::Influencer,
       [] {
         const auto influencer = self_test_influencer();
         GXAuroraSetInfluencers(&influencer, 1);
       }},
      {GX_AURORA_RENDER_WIND_SOURCES, wire::CountedHeader + wire::WindSource,
       [] {
         const auto source = self_test_wind_source();
         GXAuroraSetWindSources(&source, 1);
       }},
      {GX_AURORA_RENDER_GRASS_PATCHES, wire::GrassPatchesHeader + wire::GrassLightSet,
       [] {
         const auto set = self_test_grass_light_set();
         GXAuroraSetGrassPatches(0, 0, 0.5f, &set, 1);
       }},
      {GX_AURORA_RENDER_SKY, wire::SkyV1,
       [] {
         const auto sky = self_test_sky();
         GXAuroraSetSky(&sky);
       }},
      {GX_AURORA_RENDER_FOG_VOLUMES, wire::CountedHeader + wire::FogVolume,
       [] {
         const auto volume = self_test_fog_volume();
         GXAuroraSetFogVolumes(&volume, 1);
       }},
  };
}

constexpr std::array<u16, 26> kDeclaredIds{
    GX_AURORA_SET_ARRAY_IMMUTABLE,  GX_AURORA_RELEASE_ARRAY_IMMUTABLE,
    GX_AURORA_RENDER_CAMERA,        GX_AURORA_RENDER_OBJECT_BEGIN,
    GX_AURORA_RENDER_OBJECT_END,    GX_AURORA_RENDER_SCOPE_PUSH,
    GX_AURORA_RENDER_SCOPE_POP,     GX_AURORA_RENDER_STAGE,
    GX_AURORA_RENDER_LIGHTS,        GX_AURORA_RENDER_ENVIRONMENT,
    GX_AURORA_RENDER_MATERIAL,      GX_AURORA_RENDER_SKELETON,
    GX_AURORA_RENDER_SCENE_EPOCH,   GX_AURORA_RENDER_TEV_POLICY,
    GX_AURORA_RENDER_INFLUENCERS,   GX_AURORA_RENDER_WIND_SOURCES,
    GX_AURORA_RENDER_GRASS_PATCHES, GX_AURORA_RENDER_SKY,
    GX_AURORA_RENDER_FOG_VOLUMES,   GX_AURORA_GPU_UPLOAD,
    GX_AURORA_GPU_RESERVE,          GX_AURORA_GPU_DISPATCH,
    GX_AURORA_LOAD_ARRAY_GPU,       GX_AURORA_GPU_UPDATE_PERSISTENT,
    GX_AURORA_GPU_CREATE_SYSTEM,    GX_AURORA_GPU_DESTROY_SYSTEM,
};
} // namespace

class GXHintTest : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    hints::testing::reset_counters();
    aurora::gfx::g_testDrawCount = 0;
  }

  void TearDown() override {
    hints::set_sink(nullptr);
    hints::testing::reset_counters();
    GXFifoTest::TearDown();
  }

  void install() {
    sink = recorder.sink();
    hints::set_sink(&sink);
  }

  template <typename Fn>
  std::vector<u8> emit(Fn&& fn) {
    capture_fifo();
    fn();
    return capture_fifo();
  }

  // Decodes the bytes followed by a BP write, which must still apply after whatever the hint did
  void decode_then_bp(std::vector<u8> bytes) {
    bytes.insert(bytes.end(), kRealBpWrite.begin(), kRealBpWrite.end());
    decode_fifo(bytes);
    EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
  }

  void expect_malformed(u16 id, const std::vector<u8>& payload, uint64_t times = 1) {
    install();
    std::vector<u8> bytes;
    for (uint64_t i = 0; i < times; ++i) {
      bytes = bytes + raw_hint(id, payload);
    }
    decode_then_bp(bytes);
    EXPECT_TRUE(recorder.events.empty());
    EXPECT_EQ(hints::counters(id).malformed, times);
    EXPECT_EQ(hints::counters(id).decoded, 0u);
  }

  void expect_core_malformed(u16 id, const std::vector<u8>& payload) {
    const uint32_t warnings = hints::testing::warnings();
    expect_malformed(id, payload);
    EXPECT_EQ(hints::testing::warnings(), warnings + 1);
    EXPECT_EQ(aurora::gx::fifo::immutable_array_stats().entries, 0u);
  }

  HintRecorder recorder;
  hints::Sink sink;
};

TEST_F(GXHintTest, Camera_RoundTrip) {
  const auto camera = aurora::gxreplay::self_test_camera();
  f32 view[3][4];
  std::memcpy(view, camera.view, sizeof(view));
  view[0][3] = 10.f;
  view[1][3] = -20.f;
  view[2][3] = 30.f;
  const auto bytes = emit([&] { GXAuroraSetCamera(AURORA_CAMERA_MAIN, view, camera.proj, GX_PERSPECTIVE, 7); });
  ASSERT_EQ(bytes.size(), 127u);
  const auto prefix = hex("50 01 00 00 00 00 78 00 01 00 3F 80 00 00");
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), bytes.begin()));

  install();
  decode_fifo(bytes);
  ASSERT_EQ(recorder.cameras.size(), 1u);
  const auto& decoded = recorder.cameras[0];
  EXPECT_EQ(decoded.role, AURORA_CAMERA_MAIN);
  EXPECT_EQ(std::memcmp(&decoded.view, view, sizeof(view)), 0);
  EXPECT_EQ(std::memcmp(&decoded.proj, camera.proj, sizeof(camera.proj)), 0);
  EXPECT_EQ(decoded.type, GX_PERSPECTIVE);
  EXPECT_EQ(decoded.cutEpoch, 7u);

  GXSetProjection(camera.proj, GX_PERSPECTIVE);
  decode_fifo(capture_fifo());
  EXPECT_EQ(std::memcmp(&decoded.gxProj, &g_gxState.proj, sizeof(g_gxState.proj)), 0);
}

TEST_F(GXHintTest, Camera_OrthoGxProjection) {
  f32 ortho[4][4]{};
  ortho[0][0] = 2.f / 640.f;
  ortho[0][3] = -1.f;
  ortho[1][1] = -2.f / 480.f;
  ortho[1][3] = 1.f;
  ortho[2][2] = -1.f;
  ortho[2][3] = -1.f;
  ortho[3][3] = 1.f;
  const auto camera = aurora::gxreplay::self_test_camera();
  install();
  decode_fifo(emit([&] { GXAuroraSetCamera(AURORA_CAMERA_UI, camera.view, ortho, GX_ORTHOGRAPHIC, 0); }));
  ASSERT_EQ(recorder.cameras.size(), 1u);
  EXPECT_EQ(recorder.cameras[0].role, AURORA_CAMERA_UI);
  EXPECT_EQ(recorder.cameras[0].type, GX_ORTHOGRAPHIC);

  GXSetProjection(ortho, GX_ORTHOGRAPHIC);
  decode_fifo(capture_fifo());
  EXPECT_EQ(std::memcmp(&recorder.cameras[0].gxProj, &g_gxState.proj, sizeof(g_gxState.proj)), 0);
}

TEST_F(GXHintTest, Camera_UsableRule) {
  const auto camera = aurora::gxreplay::self_test_camera();
  aurora::Mat4x4<float> perspective;
  std::memcpy(&perspective, camera.proj, sizeof(perspective));
  const auto usable = hints::gx_projection(perspective, GX_PERSPECTIVE);
  EXPECT_TRUE(hints::main_camera_usable(usable));
  EXPECT_FALSE(hints::main_camera_usable(hints::gx_projection(perspective, GX_ORTHOGRAPHIC)));
  auto nan = usable;
  nan.m1[2] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(hints::main_camera_usable(nan));
  auto flat = usable;
  flat.m0[0] = 0.f;
  EXPECT_FALSE(hints::main_camera_usable(flat));
  auto noDepth = usable;
  noDepth.m2[3] = 0.f;
  EXPECT_FALSE(hints::main_camera_usable(noDepth));
}

TEST_F(GXHintTest, ObjectBegin_V1RoundTrip) {
  const auto bytes = emit([] { GXAuroraBeginObject(0x10, 0, 1, AURORA_CAT_TERRAIN, 0); });
  EXPECT_EQ(bytes, hex("50 01 01 00 00 00 1A 00 01 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 01 00 00 00 08 00 00 "
                       "00 00"));
  install();
  decode_fifo(emit([] { GXAuroraBeginObject(0x123456789ABCDEF0ull, 4, 5, AURORA_CAT_DECAL, AURORA_OBJ_TELEPORT); }));
  ASSERT_EQ(recorder.objects.size(), 1u);
  const auto& object = recorder.objects[0];
  EXPECT_EQ(object.id, 0x123456789ABCDEF0ull);
  EXPECT_EQ(object.subId, 4u);
  EXPECT_EQ(object.generation, 5u);
  EXPECT_EQ(object.categories, AURORA_CAT_DECAL);
  EXPECT_EQ(object.flags, AURORA_OBJ_TELEPORT);
  EXPECT_FALSE(object.hasWorld);
}

TEST_F(GXHintTest, ObjectBegin_V2CarriesWorld) {
  const auto bytes = emit([] { GXAuroraBeginObjectWorld(0x10, 2, 3, AURORA_CAT_TERRAIN, 0, kWorld); });
  EXPECT_EQ(header_size(bytes), wire::ObjectBeginV2);
  EXPECT_EQ(bytes[7], 0x00);
  EXPECT_EQ(bytes[8], 0x02);
  install();
  decode_fifo(bytes);
  ASSERT_EQ(recorder.objects.size(), 1u);
  EXPECT_TRUE(recorder.objects[0].hasWorld);
  EXPECT_EQ(recorder.objects[0].subId, 2u);
  EXPECT_EQ(std::memcmp(&recorder.objects[0].world, kWorld, sizeof(kWorld)), 0);
}

TEST_F(GXHintTest, ObjectEnd_RoundTrip) {
  const auto bytes = emit([] { GXAuroraEndObject(); });
  EXPECT_EQ(bytes, hex("50 01 02 00 00 00 02 00 01"));
  install();
  decode_fifo(bytes);
  EXPECT_EQ(recorder.count(GX_AURORA_RENDER_OBJECT_END), 1u);
}

TEST_F(GXHintTest, Scope_RoundTripWithInherit) {
  const auto bytes = emit([] { GXAuroraPushScope(AURORA_CAT_EMISSIVE, 0, AURORA_CAMERA_INHERIT); });
  EXPECT_EQ(bytes, hex("50 01 03 00 00 00 0B 00 01 00 00 08 00 00 00 00 00 FF"));
  EXPECT_EQ(emit([] { GXAuroraPopScope(); }), hex("50 01 04 00 00 00 02 00 01"));
  install();
  decode_fifo(bytes);
  decode_fifo(emit([] { GXAuroraPushScope(AURORA_CAT_SKY, AURORA_OBJ_CAPTURE_ONLY, AURORA_CAMERA_SKY); }));
  decode_fifo(emit([] { GXAuroraPopScope(); }));
  ASSERT_EQ(recorder.scopes.size(), 2u);
  EXPECT_EQ(recorder.scopes[0].categories, AURORA_CAT_EMISSIVE);
  EXPECT_EQ(recorder.scopes[0].flags, 0u);
  EXPECT_EQ(recorder.scopes[0].role, AURORA_CAMERA_INHERIT);
  EXPECT_EQ(recorder.scopes[1].flags, AURORA_OBJ_CAPTURE_ONLY);
  EXPECT_EQ(recorder.scopes[1].role, AURORA_CAMERA_SKY);
  EXPECT_EQ(recorder.count(GX_AURORA_RENDER_SCOPE_POP), 1u);
}

TEST_F(GXHintTest, Stage_AllMarkers) {
  EXPECT_EQ(emit([] { GXAuroraStage(AURORA_STAGE_OPAQUE_DONE); }), hex("50 01 05 00 00 00 03 00 01 02"));
  install();
  for (int stage = AURORA_STAGE_SCENE_BEGIN; stage <= AURORA_STAGE_FRAME_END; ++stage) {
    GXAuroraStage(static_cast<AuroraRenderStage>(stage));
  }
  decode_fifo(capture_fifo());
  EXPECT_EQ(recorder.stages, (std::vector{Stage::SceneBegin, Stage::OpaqueDone, Stage::PrePost, Stage::SceneEnd,
                                          Stage::BeforeHud, Stage::FrameEnd}));
}

TEST_F(GXHintTest, Lights_RoundTrip) {
  const auto light = aurora::gxreplay::self_test_light();
  EXPECT_EQ(emit([&] { GXAuroraSetLights(&light, 1); }),
            hex("50 01 06 00 00 00 4C 00 02 00 01 00 00 00 01 01 00 00 00 00 00 00 00 3F 80 00 00 40 00 00 00 40 40 "
                "00 00 00 00 00 00 00 00 00 00 00 00 00 00 3F 80 00 00 3F 00 00 00 3E 80 00 00 43 5C 00 00 00 00 00 "
                "00 3F 80 00 00 BF 80 00 00 3F C0 00 00 3F 80 00 00"));
  const auto lights = make_lights(3);
  install();
  decode_fifo(emit([&] { GXAuroraSetLights(lights.data(), 3); }));
  ASSERT_EQ(recorder.lights.size(), 1u);
  ASSERT_EQ(recorder.lights[0].size(), 3u);
  EXPECT_EQ(std::memcmp(recorder.lights[0].data(), lights.data(), 3 * sizeof(AuroraLight)), 0);
}

TEST_F(GXHintTest, Lights_MaxCount) {
  const auto lights = make_lights(AURORA_MAX_LIGHTS);
  const auto bytes = emit([&] { GXAuroraSetLights(lights.data(), AURORA_MAX_LIGHTS); });
  EXPECT_EQ(header_size(bytes), wire::CountedHeader + AURORA_MAX_LIGHTS * wire::LightV2);
  install();
  decode_fifo(bytes);
  ASSERT_EQ(recorder.lights.size(), 1u);
  ASSERT_EQ(recorder.lights[0].size(), size_t{AURORA_MAX_LIGHTS});
  EXPECT_EQ(std::memcmp(recorder.lights[0].data(), lights.data(), lights.size() * sizeof(AuroraLight)), 0);
}

TEST_F(GXHintTest, Lights_EmitterClampsAboveMax) {
  const auto lights = make_lights(300);
  const uint32_t warnings = hints::testing::warnings();
  const auto bytes = emit([&] { GXAuroraSetLights(lights.data(), 300); });
  emit([&] { GXAuroraSetLights(lights.data(), 300); });
  EXPECT_EQ(header_size(bytes), wire::CountedHeader + AURORA_MAX_LIGHTS * wire::LightV2);
  EXPECT_EQ(bytes[9], 0x01);
  EXPECT_EQ(bytes[10], 0x00);
  EXPECT_EQ(hints::testing::warnings(), warnings + 1);
}

TEST_F(GXHintTest, Lights_V1DecodesDefaultVolumetricScale) {
  auto light = make_lights(2)[1];
  std::vector<u8> payload(wire::CountedHeader + wire::LightV1);
  wire::Writer writer{payload};
  writer.u16(1);
  writer.u16(1);
  wire::write(writer, light, 1);
  ASSERT_EQ(writer.size(), payload.size());
  install();
  decode_fifo(raw_hint(GX_AURORA_RENDER_LIGHTS, payload));
  ASSERT_EQ(recorder.lights.size(), 1u);
  light.volumetricScale = 1.f;
  EXPECT_TRUE(same_bits(recorder.lights[0][0], light));
}

TEST_F(GXHintTest, Environment_V2RoundTrip) {
  const auto environment = aurora::gxreplay::self_test_environment();
  const auto bytes = emit([&] { GXAuroraSetEnvironment(&environment); });
  ASSERT_EQ(bytes.size(), 245u);
  const auto prefix = hex("50 01 07 00 00 00 EE 00 02 3F 80 00 00");
  const auto suffix = hex("BF 80 00 00 00 01 00 02 00 03 00 04 00 05 00 06 00 07 00 08 00 09 00 0A 00 00 00 00");
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), bytes.begin()));
  EXPECT_TRUE(std::equal(suffix.rbegin(), suffix.rend(), bytes.rbegin()));
  install();
  decode_fifo(bytes);
  ASSERT_EQ(recorder.environments.size(), 1u);
  EXPECT_EQ(recorder.environments[0].version, 2u);
  EXPECT_TRUE(same_bits(recorder.environments[0].value, environment));
}

TEST_F(GXHintTest, Environment_V1DecodesDefaults) {
  AuroraEnvironment sent{};
  sent.skyVisibility = 0.75f;
  sent.ambientColor[1] = 0.5f;
  sent.fogColor[2] = 0.25f;
  sent.windDir[0] = 1.f;
  sent.windStrength = 2.f;
  sent.exposureBias = -1.f;
  sent.timeOfDay = 18.5f;
  sent.flags = AURORA_ENV_INTERIOR | AURORA_ENV_SNOW;
  sent.unitsPerMeter = 55.f;
  sent.volumeFar = 9.f;
  std::vector<u8> payload(wire::EnvironmentV1);
  wire::Writer writer{payload};
  writer.u16(1);
  wire::write(writer, sent, 1);
  ASSERT_EQ(writer.size(), payload.size());
  install();
  decode_fifo(raw_hint(GX_AURORA_RENDER_ENVIRONMENT, payload));
  ASSERT_EQ(recorder.environments.size(), 1u);
  EXPECT_EQ(recorder.environments[0].version, 1u);

  AuroraEnvironment expected{
      .skyVisibility = 0.75f,
      .ambientColor = {0.f, 0.5f, 0.f},
      .fogColor = {0.f, 0.f, 0.25f},
      .windDir = {1.f, 0.f, 0.f},
      .windStrength = 2.f,
      .exposureBias = -1.f,
      .timeOfDay = 18.5f,
      .flags = AURORA_ENV_INTERIOR | AURORA_ENV_SNOW,
      .unitsPerMeter = 100.f,
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
  EXPECT_TRUE(same_bits(recorder.environments[0].value, expected));
  EXPECT_EQ(recorder.environments[0].value.fogType, static_cast<u32>(GX_FOG_NONE));
}

TEST_F(GXHintTest, Material_RoundTrip) {
  const auto bytes = emit([] { GXAuroraSetMaterialClass(1); });
  EXPECT_EQ(bytes, hex("50 01 08 00 00 00 06 00 01 00 00 00 01"));
  install();
  decode_fifo(bytes + emit([] { GXAuroraSetMaterialClass(0xDEADBEEF); }));
  EXPECT_EQ(recorder.materials, (std::vector<uint32_t>{1, 0xDEADBEEF}));
}

TEST_F(GXHintTest, SceneEpoch_RoundTrip) {
  const auto bytes = emit([] { GXAuroraSetSceneEpoch(1); });
  EXPECT_EQ(bytes, hex("50 01 0A 00 00 00 06 00 01 00 00 00 01"));
  install();
  decode_fifo(bytes + emit([] { GXAuroraSetSceneEpoch(0xFFFFFFFE); }));
  EXPECT_EQ(recorder.epochs, (std::vector<uint32_t>{1, 0xFFFFFFFE}));
}

TEST_F(GXHintTest, Skeleton_RoundTrip) {
  const auto bytes = emit([] { GXAuroraSetSkeleton(0x10, kJoint, 1); });
  ASSERT_EQ(bytes.size(), 67u);
  const auto prefix = hex("50 01 09 00 00 00 3C 00 01 00 00 00 00 00 00 00 10 00 01 3F 80 00 00");
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), bytes.begin()));
  const f32 joints[2][3][4]{{{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}},
                            {{-1.f, 0.f, 0.f, 0.5f}, {0.f, -1.f, 0.f, 0.25f}, {0.f, 0.f, 1.f, 0.125f}}};
  install();
  decode_fifo(emit([&] { GXAuroraSetSkeleton(0xABCDEF, joints, 2); }));
  ASSERT_EQ(recorder.skeletons.size(), 1u);
  EXPECT_EQ(recorder.skeletonIds[0], 0xABCDEFu);
  ASSERT_EQ(recorder.skeletons[0].size(), 2u);
  EXPECT_EQ(std::memcmp(recorder.skeletons[0].data(), joints, sizeof(joints)), 0);
}

TEST_F(GXHintTest, TevPolicy_RoundTrip) {
  const auto policy = aurora::gxreplay::self_test_tev_policy();
  const auto bytes = emit([&] { GXAuroraSetTevPolicy(&policy); });
  EXPECT_EQ(bytes, hex("50 01 0B 00 00 00 22 00 01 00 01 00 00") + zeros(28));
  const auto busy = busy_policy();
  install();
  decode_fifo(bytes + emit([&] { GXAuroraSetTevPolicy(&busy); }));
  ASSERT_EQ(recorder.policies.size(), 2u);
  EXPECT_TRUE(same_bits(recorder.policies[0], policy));
  EXPECT_TRUE(same_bits(recorder.policies[1], busy));
}

TEST_F(GXHintTest, Influencers_RoundTrip) {
  const auto influencer = aurora::gxreplay::self_test_influencer();
  EXPECT_EQ(emit([&] { GXAuroraSetInfluencers(&influencer, 1); }),
            hex("50 01 0C 00 00 00 24 00 01 00 01 3F 80 00 00 40 00 00 00 40 40 00 00 3F C0 00 00") + zeros(16));
  std::vector<AuroraInfluencer> influencers(AURORA_MAX_INFLUENCERS);
  for (size_t i = 0; i < influencers.size(); ++i) {
    influencers[i] = {.pos = {static_cast<f32>(i), 1.f, 2.f},
                      .radius = 3.f,
                      .vel = {0.f, -static_cast<f32>(i), 0.5f},
                      .kind = static_cast<u32>(AURORA_INFLUENCER_CUT_SWEEP | i << 8)};
  }
  install();
  decode_fifo(emit([&] { GXAuroraSetInfluencers(influencers.data(), AURORA_MAX_INFLUENCERS); }));
  ASSERT_EQ(recorder.influencers.size(), 1u);
  ASSERT_EQ(recorder.influencers[0].size(), influencers.size());
  EXPECT_EQ(std::memcmp(recorder.influencers[0].data(), influencers.data(), influencers.size() * 32), 0);
}

TEST_F(GXHintTest, WindSources_RoundTrip) {
  const auto source = aurora::gxreplay::self_test_wind_source();
  EXPECT_EQ(emit([&] { GXAuroraSetWindSources(&source, 1); }),
            hex("50 01 0D 00 00 00 24 00 01 00 01") + zeros(12) +
                hex("42 48 00 00 3F 80 00 00 00 00 00 00 00 00 00 00 3E 99 99 9A"));
  std::vector<AuroraWindSource> sources(AURORA_MAX_WIND_SOURCES);
  for (size_t i = 0; i < sources.size(); ++i) {
    sources[i] = {.pos = {1.f, static_cast<f32>(i), 3.f}, .radius = 40.f, .dir = {0.f, 1.f, 0.f}, .power = 0.1f};
  }
  install();
  decode_fifo(emit([&] { GXAuroraSetWindSources(sources.data(), AURORA_MAX_WIND_SOURCES); }));
  ASSERT_EQ(recorder.windSources.size(), 1u);
  ASSERT_EQ(recorder.windSources[0].size(), sources.size());
  EXPECT_EQ(std::memcmp(recorder.windSources[0].data(), sources.data(), sources.size() * 32), 0);
}

TEST_F(GXHintTest, ArrayImmutable_RoundTrip) {
  const auto set = emit([] { GXAuroraSetArrayImmutable(kArray, 0x90, GX_TRUE, 2); });
  EXPECT_EQ(set, hex("50 00 50 00 00 00 13 00 01 00 00 00 00 80 20 00 00 00 00 00 90 01 00 00 00 02"));
  const auto release = emit([] { GXAuroraReleaseArrayImmutable(kArray, 2); });
  EXPECT_EQ(release, hex("50 00 51 00 00 00 0E 00 01 00 00 00 00 80 20 00 00 00 00 00 02"));

  const auto setPayload = payload_of(set);
  aurora::ByteReader in{setPayload};
  EXPECT_EQ(in.read<u16>(), 1u);
  EXPECT_EQ(in.read<u64>(), 0x80200000u);
  EXPECT_EQ(in.read<u32>(), 0x90u);
  EXPECT_EQ(in.read<u8>(), 1u);
  EXPECT_EQ(in.read<u32>(), 2u);
  EXPECT_TRUE(in.empty());
  const auto releasePayload = payload_of(release);
  aurora::ByteReader releaseIn{releasePayload};
  EXPECT_EQ(releaseIn.read<u16>(), 1u);
  EXPECT_EQ(releaseIn.read<u64>(), 0x80200000u);
  EXPECT_EQ(releaseIn.read<u32>(), 2u);
  EXPECT_TRUE(releaseIn.empty());

  install();
  decode_fifo(set);
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_SET_ARRAY_IMMUTABLE).decoded, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_SET_ARRAY_IMMUTABLE).skipped, 0u);
  EXPECT_EQ(hints::counters(GX_AURORA_SET_ARRAY_IMMUTABLE).bytes, 26u);
  EXPECT_EQ(aurora::gx::fifo::immutable_array_stats().entries, 1u);
  decode_fifo(release);
  EXPECT_EQ(hints::counters(GX_AURORA_RELEASE_ARRAY_IMMUTABLE).decoded, 1u);
  EXPECT_EQ(aurora::gx::fifo::immutable_array_stats().entries, 0u);
  EXPECT_EQ(aurora::gx::fifo::immutable_array_stats().staleReleases, 0u);
}

TEST_F(GXHintTest, GrassPatches_RoundTrip) {
  const auto set = aurora::gxreplay::self_test_grass_light_set();
  const auto bytes = emit([&] { GXAuroraSetGrassPatches(0, 0, 0.5f, &set, 1); });
  EXPECT_EQ(bytes, hex("50 01 0E 00 00 00 3F 00 01 00 00 00 00 00 00 00 00 3F 00 00 00 01") + zeros(16) +
                       hex("3F 80 00 00") + zeros(28));
  const AuroraGrassLightSet busy{.reg1Rgb = {1, 2, 3},
                                 .shine = 4,
                                 .matRgba = {5, 6, 7, 8},
                                 .lightIdx = {9, 10, 11, 12, 13, 14, 0xFF},
                                 .flags = AURORA_GRASS_LIGHTS_1_4_ONLY,
                                 .grassLightRatio = 0.5f,
                                 .fog = {1.f, 2.f, 3.f, 4.f, 5.f},
                                 .fogRgb = {20, 30, 40},
                                 .fogType = 2,
                                 .reserved = 0};
  const auto busyBytes = emit([&] { GXAuroraSetGrassPatches(7, 1000, 0.25f, &busy, 1); });
  EXPECT_EQ(header_size(busyBytes), wire::GrassPatchesHeader + wire::GrassLightSet);
  const auto payload = payload_of(busyBytes);
  aurora::ByteReader in{payload};
  EXPECT_EQ(in.read<u16>(), 1u);
  EXPECT_EQ(in.read<u32>(), 7u);
  EXPECT_EQ(in.read<u32>(), 1000u);
  EXPECT_EQ(in.read<f32>(), 0.25f);
  EXPECT_EQ(in.read<u8>(), 1u);
  ASSERT_EQ(in.offset(), size_t{wire::GrassPatchesHeader});
  AuroraGrassLightSet decoded{};
  EXPECT_TRUE(wire::read(in, decoded));
  EXPECT_TRUE(same_bits(decoded, busy));

  install();
  decode_fifo(bytes);
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_GRASS_PATCHES).skipped, 1u);
}

TEST_F(GXHintTest, Sky_RoundTrip) {
  const auto sky = aurora::gxreplay::self_test_sky();
  const auto bytes = emit([&] { GXAuroraSetSky(&sky); });
  ASSERT_EQ(bytes.size(), 173u);
  const auto prefix = hex("50 01 0F 00 00 00 A6 00 01 00 00 00 01 3F 00");
  const auto suffix = hex("00 00 00 02 3F 80 00 00") + zeros(50) + hex("00 02");
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), bytes.begin()));
  EXPECT_TRUE(std::equal(suffix.rbegin(), suffix.rend(), bytes.rbegin()));
  const auto payload = payload_of(bytes);
  aurora::ByteReader in{payload};
  in.skip(sizeof(u16));
  AuroraSky decoded{};
  EXPECT_TRUE(wire::read(in, decoded));
  EXPECT_TRUE(same_bits(decoded, sky));

  install();
  decode_fifo(bytes);
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SKY).skipped, 1u);
}

TEST_F(GXHintTest, Sky_LayerCountAboveTwoRejected) {
  auto sky = aurora::gxreplay::self_test_sky();
  sky.layerCount = 3;
  std::vector<u8> payload(wire::SkyV1 - sizeof(u16));
  wire::Writer writer{payload};
  wire::write(writer, sky);
  aurora::ByteReader in{payload};
  AuroraSky decoded{};
  EXPECT_FALSE(wire::read(in, decoded));
}

TEST_F(GXHintTest, FogVolumes_RoundTrip) {
  const auto volume = aurora::gxreplay::self_test_fog_volume();
  const auto bytes = emit([&] { GXAuroraSetFogVolumes(&volume, 1); });
  EXPECT_EQ(bytes,
            hex("50 01 10 00 00 00 44 00 01 00 01 C1 90 00 00 3F 80 00 00 41 60 00 00 40 C0 00 00 3F C0 00 00 40 "
                "C0 00 00 00 00 00 00 00 00 00 00 3D 4C CC CD") +
                zeros(28));
  const auto payload = payload_of(bytes);
  aurora::ByteReader in{payload};
  in.skip(wire::CountedHeader);
  AuroraFogVolume decoded{};
  EXPECT_TRUE(wire::read(in, decoded));
  EXPECT_TRUE(same_bits(decoded, volume));

  install();
  decode_fifo(bytes);
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_FOG_VOLUMES).skipped, 1u);
}

TEST_F(GXHintTest, SkippedReservedAndDispatch) {
  install();
  std::vector<u8> bytes;
  const std::vector<u16> ids{0x0052, 0x00FF, 0x0111, GX_AURORA_GPU_DISPATCH, 0x0127};
  for (const u16 id : ids) {
    bytes = bytes + raw_hint(id, hex("00 01 AA BB CC"));
  }
  bytes = bytes + raw_hint(0x01FF, {});
  decode_then_bp(bytes);
  EXPECT_TRUE(recorder.events.empty());
  for (const u16 id : ids) {
    EXPECT_EQ(hints::counters(id).skipped, 1u) << id;
    EXPECT_EQ(hints::counters(id).bytes, 12u) << id;
  }
  EXPECT_EQ(hints::counters(0x01FF).skipped, 1u);
  EXPECT_EQ(hints::counters(0x01FF).bytes, 7u);
}

TEST_F(GXHintTest, GpuCommandsRejectMalformedSizes) {
  install();
  std::vector<u8> bytes;
  std::vector<u16> ids;
  for (u16 id = GX_AURORA_GPU_UPLOAD; id <= GX_AURORA_GPU_DESTROY_SYSTEM; ++id) {
    if (id != GX_AURORA_GPU_DISPATCH) {
      ids.push_back(id);
      bytes = bytes + raw_hint(id, hex("00 01 AA BB CC"));
    }
  }
  const uint64_t hintBytes = hints::total_bytes();
  decode_then_bp(bytes);
  EXPECT_TRUE(recorder.events.empty());
  for (const u16 id : ids) {
    EXPECT_EQ(hints::counters(id).malformed, 1u) << id;
    EXPECT_EQ(hints::counters(id).skipped, 0u) << id;
    EXPECT_EQ(hints::counters(id).decoded, 0u) << id;
    EXPECT_EQ(hints::counters(id).bytes, 12u) << id;
  }
  EXPECT_EQ(hints::total_bytes(), hintBytes);
  EXPECT_EQ(hints::gpu_command_bytes(), 12u * ids.size());
}

TEST_F(GXHintTest, EveryDeclaredIdHasNameAndVersion) {
  for (const u16 id : kDeclaredIds) {
    EXPECT_NE(hints::name(id), "reserved") << id;
    EXPECT_EQ(hints::latest_version(id) == 0, id == GX_AURORA_GPU_DISPATCH) << id;
  }
  EXPECT_EQ(hints::latest_version(GX_AURORA_GPU_UPLOAD), 1u);
  EXPECT_EQ(hints::latest_version(GX_AURORA_GPU_DESTROY_SYSTEM), 1u);
  EXPECT_EQ(hints::name(GX_AURORA_RENDER_TEV_POLICY), "TEV_POLICY");
  EXPECT_EQ(hints::latest_version(GX_AURORA_RENDER_LIGHTS), 2u);
  EXPECT_EQ(hints::latest_version(GX_AURORA_RENDER_ENVIRONMENT), 2u);
  EXPECT_EQ(hints::latest_version(GX_AURORA_RENDER_OBJECT_BEGIN), 2u);
  for (u16 id = hints::FirstId; id <= hints::LastId; ++id) {
    if (std::ranges::find(kDeclaredIds, id) == kDeclaredIds.end()) {
      EXPECT_EQ(hints::name(id), "reserved") << id;
      EXPECT_EQ(hints::latest_version(id), 0u) << id;
    }
  }
}

TEST_F(GXHintTest, NoSinkSkipsEverything) {
  g_gxState.dirty = 0;
  for (const auto& emitter : emitter_cases()) {
    if (!emitter.extra) {
      emitter.emit();
    }
  }
  decode_fifo(capture_fifo());
  EXPECT_EQ(g_gxState.dirty, 0u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 0u);
  for (const auto& emitter : emitter_cases()) {
    const bool core = emitter.id <= GX_AURORA_HINT_CORE_LAST;
    EXPECT_EQ(hints::counters(emitter.id).skipped, core ? 0u : 1u) << emitter.id;
    EXPECT_EQ(hints::counters(emitter.id).decoded, core ? 1u : 0u) << emitter.id;
  }
}

TEST_F(GXHintTest, NullCallbackSkipsOnlyThatId) {
  sink = {.userData = &recorder, .stage = recorder.sink().stage};
  hints::set_sink(&sink);
  GXAuroraStage(AURORA_STAGE_SCENE_BEGIN);
  GXAuroraSetSceneEpoch(4);
  GXAuroraEndObject();
  decode_fifo(capture_fifo());
  EXPECT_EQ(recorder.stages, std::vector{Stage::SceneBegin});
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_STAGE).decoded, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SCENE_EPOCH).skipped, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_OBJECT_END).skipped, 1u);
}

TEST_F(GXHintTest, DecodeLeavesGxStateUntouched) {
  decode_fifo(flush_and_capture());
  g_gxState.dirty = 0;
  const aurora::gx::GXState before = g_gxState;
  install();
  for (const auto& emitter : emitter_cases()) {
    emitter.emit();
  }
  decode_fifo(capture_fifo());
  EXPECT_EQ(recorder.events.size(), 15u);
  expect_state_eq(before, g_gxState);
  EXPECT_EQ(g_gxState.dirty, 0u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 0u);
}

TEST_F(GXHintTest, BytesCountEveryOutcome) {
  install();
  const auto sky = aurora::gxreplay::self_test_sky();
  GXAuroraSetSceneEpoch(1);
  GXAuroraSetSky(&sky);
  auto bytes = capture_fifo() + raw_hint(GX_AURORA_RENDER_CAMERA, hex("00 01 00 00 00"));
  decode_fifo(bytes);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SCENE_EPOCH).decoded, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SKY).skipped, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_CAMERA).malformed, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SCENE_EPOCH).bytes, 13u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_SKY).bytes, 173u);
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_CAMERA).bytes, 12u);
  uint64_t sum = 0;
  for (u16 id = hints::FirstId; id <= hints::LastId; ++id) {
    sum += hints::counters(id).bytes;
  }
  EXPECT_EQ(sum, 13u + 173u + 12u);
  EXPECT_EQ(hints::total_bytes(), sum);
}

TEST_F(GXHintTest, Malformed_Size) { expect_malformed(GX_AURORA_RENDER_SCENE_EPOCH, hex("00 01 00 00 00 01 00")); }

TEST_F(GXHintTest, Malformed_VersionZero) { expect_malformed(GX_AURORA_RENDER_STAGE, hex("00 00 01")); }

TEST_F(GXHintTest, Malformed_UnknownVersion) { expect_malformed(GX_AURORA_RENDER_STAGE, hex("00 02 01")); }

// A newer producer's hint is rejected even when its size happens to match the newest layout this build knows
TEST_F(GXHintTest, Malformed_NewerLightsVersion) {
  const auto light = aurora::gxreplay::self_test_light();
  auto payload = payload_of(emit([&] { GXAuroraSetLights(&light, 1); }));
  payload[1] = 3;
  expect_malformed(GX_AURORA_RENDER_LIGHTS, payload);
}

TEST_F(GXHintTest, Malformed_NewerEnvironmentVersion) {
  const auto environment = aurora::gxreplay::self_test_environment();
  auto payload = payload_of(emit([&] { GXAuroraSetEnvironment(&environment); }));
  payload[1] = 3;
  expect_malformed(GX_AURORA_RENDER_ENVIRONMENT, payload);
}

TEST_F(GXHintTest, Malformed_NewerObjectBeginVersion) {
  auto payload = payload_of(emit([] { GXAuroraBeginObjectWorld(0x10, 0, 1, AURORA_CAT_TERRAIN, 0, kWorld); }));
  payload[1] = 3;
  expect_malformed(GX_AURORA_RENDER_OBJECT_BEGIN, payload);
}

TEST_F(GXHintTest, Malformed_CountAboveMax) {
  std::vector<u8> payload(wire::CountedHeader + (AURORA_MAX_LIGHTS + 1) * wire::LightV2);
  wire::Writer writer{payload};
  writer.u16(2);
  writer.u16(AURORA_MAX_LIGHTS + 1);
  expect_malformed(GX_AURORA_RENDER_LIGHTS, payload);
}

TEST_F(GXHintTest, Malformed_CountSizeMismatch) {
  const auto bytes = emit([] {
    const auto light = aurora::gxreplay::self_test_light();
    GXAuroraSetLights(&light, 1);
  });
  auto payload = payload_of(bytes);
  payload[3] = 2;
  expect_malformed(GX_AURORA_RENDER_LIGHTS, payload);
}

TEST_F(GXHintTest, Malformed_CameraInheritRole) {
  const auto camera = aurora::gxreplay::self_test_camera();
  const auto bytes =
      emit([&] { GXAuroraSetCamera(AURORA_CAMERA_INHERIT, camera.view, camera.proj, GX_PERSPECTIVE, 0); });
  expect_malformed(GX_AURORA_RENDER_CAMERA, payload_of(bytes));
}

TEST_F(GXHintTest, Malformed_StageOutOfRange) {
  install();
  decode_then_bp(raw_hint(GX_AURORA_RENDER_STAGE, hex("00 01 00")) + raw_hint(GX_AURORA_RENDER_STAGE, hex("00 01 07")));
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_STAGE).malformed, 2u);
}

TEST_F(GXHintTest, Malformed_LightType) {
  auto light = aurora::gxreplay::self_test_light();
  light.type = 3;
  expect_malformed(GX_AURORA_RENDER_LIGHTS, payload_of(emit([&] { GXAuroraSetLights(&light, 1); })));
}

TEST_F(GXHintTest, Malformed_TevReservedBits) {
  AuroraTevPolicy policy{.mask = 1u << 28};
  expect_malformed(GX_AURORA_RENDER_TEV_POLICY, payload_of(emit([&] { GXAuroraSetTevPolicy(&policy); })));
}

TEST_F(GXHintTest, Malformed_WarnsOncePerId) {
  const uint32_t warnings = hints::testing::warnings();
  expect_malformed(GX_AURORA_RENDER_CAMERA, hex("00 01 05"), 2);
  EXPECT_EQ(hints::testing::warnings(), warnings + 1);
}

TEST_F(GXHintTest, Malformed_SetArrayShort) {
  auto payload = set_array_payload(1, 0x80200000, 0x90, 1);
  payload.pop_back();
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, payload);
}

TEST_F(GXHintTest, Malformed_SetArrayVersion) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(2, 0x80200000, 0x90, 1));
}

TEST_F(GXHintTest, Malformed_SetArrayEndian) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(1, 0x80200000, 0x90, 2));
}

TEST_F(GXHintTest, Malformed_SetArrayNullAddress) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(1, 0, 0x90, 1));
}

TEST_F(GXHintTest, Malformed_SetArrayEmpty) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(1, 0x80200000, 0, 1));
}

TEST_F(GXHintTest, Malformed_SetArrayWraps) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(1, 0xFFFFFFFFFFFFFF00ull, 0x200, 1));
}

TEST_F(GXHintTest, Malformed_SetArrayEndsPastTheTop) {
  expect_core_malformed(GX_AURORA_SET_ARRAY_IMMUTABLE, set_array_payload(1, 0xFFFFFFFFFFFFFF00ull, 0x100, 1));
}

TEST_F(GXHintTest, Malformed_ReleaseArrayShort) {
  std::vector<u8> payload(wire::ReleaseArrayImmutableV1);
  wire::Writer writer{payload};
  writer.u16(1);
  writer.u64(0x80200000);
  writer.u32(1);
  payload.pop_back();
  expect_core_malformed(GX_AURORA_RELEASE_ARRAY_IMMUTABLE, payload);
}

TEST_F(GXHintTest, SelfTestGarbageIsRejected) {
  install();
  std::vector<u8> bytes;
  for (const u32 size : aurora::gxreplay::SelfTestPayloadSizes) {
    std::vector<u8> payload(size);
    for (u32 i = 0; i < size; ++i) {
      payload[i] = aurora::gxreplay::self_test_hint_byte(i);
    }
    bytes = bytes + raw_hint(GX_AURORA_RENDER_CAMERA, payload);
  }
  decode_then_bp(bytes);
  EXPECT_TRUE(recorder.events.empty());
  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_CAMERA).malformed, 4u);
}

TEST_F(GXHintTest, HintsAndDrawsKeepStreamOrder) {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_U8, 0);
  const auto draw = [] {
    GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
    GXPosition3u8(0, 0, 0);
    GXPosition3u8(1, 0, 0);
    GXPosition3u8(0, 1, 0);
    GXEnd();
  };
  const auto camera = aurora::gxreplay::self_test_camera();
  draw();
  GXAuroraSetCamera(AURORA_CAMERA_MAIN, camera.view, camera.proj, GX_PERSPECTIVE, 0);
  draw();
  GXAuroraStage(AURORA_STAGE_OPAQUE_DONE);
  draw();
  install();
  aurora::gfx::g_testDrawCount = 0;
  decode_fifo(capture_fifo());
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 3u);
  ASSERT_EQ(recorder.events.size(), 2u);
  EXPECT_EQ(recorder.events[0].id, GX_AURORA_RENDER_CAMERA);
  EXPECT_EQ(recorder.events[0].draws, 1u);
  EXPECT_EQ(recorder.events[1].id, GX_AURORA_RENDER_STAGE);
  EXPECT_EQ(recorder.events[1].draws, 2u);
}

TEST_F(GXHintTest, HintsInsideDisplayListDecode) {
  alignas(32) std::array<u8, 128> storage{};
  install();

  aurora::gx::fifo::init();
  aurora::gx::fifo::begin_frame();
  GXBeginDisplayList(storage.data(), static_cast<u32>(storage.size()));
  GXAuroraPushScope(AURORA_CAT_UI, 0, AURORA_CAMERA_UI);
  GXAuroraStage(AURORA_STAGE_BEFORE_HUD);
  GXAuroraPopScope();
  GX_WRITE_RAS_REG(0x41123456);
  const u32 size = GXEndDisplayList();
  GXCallDisplayList(storage.data(), size);
  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::shutdown();

  const std::vector<u16> expected{GX_AURORA_RENDER_SCOPE_PUSH, GX_AURORA_RENDER_STAGE, GX_AURORA_RENDER_SCOPE_POP};
  const auto ids = [&] {
    std::vector<u16> out;
    for (const auto& event : recorder.events) {
      out.push_back(event.id);
    }
    return out;
  };
  EXPECT_EQ(ids(), expected);
  EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);

  recorder.events.clear();
  reset_gx_state();
  aurora::gx::fifo::process(storage.data(), size);
  EXPECT_EQ(ids(), expected);
  EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
}

TEST_F(GXHintTest, LargeLightsThroughThreadedFifo) {
  const auto lights = make_lights(AURORA_MAX_LIGHTS);
  install();
  aurora::gx::fifo::init();
  aurora::gx::fifo::begin_frame();
  GXAuroraSetLights(lights.data(), AURORA_MAX_LIGHTS);
  aurora::gx::fifo::write_data(kRealBpWrite.data(), static_cast<u32>(kRealBpWrite.size()));
  aurora::gx::fifo::publish();
  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::shutdown();

  EXPECT_EQ(hints::counters(GX_AURORA_RENDER_LIGHTS).decoded, 1u);
  ASSERT_EQ(recorder.lights.size(), 1u);
  EXPECT_EQ(std::memcmp(recorder.lights[0].data(), lights.data(), lights.size() * sizeof(AuroraLight)), 0);
  EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
}

TEST_F(GXHintTest, EmitterWritesDeclaredSize) {
  for (const auto& emitter : emitter_cases()) {
    const auto bytes = emit(emitter.emit);
    ASSERT_GE(bytes.size(), size_t{GX_AURORA_HINT_HEADER_SIZE}) << emitter.id;
    EXPECT_EQ(bytes[0], GX_AURORA) << emitter.id;
    EXPECT_EQ(static_cast<u16>(bytes[1] << 8 | bytes[2]), emitter.id);
    EXPECT_EQ(header_size(bytes), bytes.size() - GX_AURORA_HINT_HEADER_SIZE) << emitter.id;
    EXPECT_EQ(header_size(bytes), emitter.payloadSize) << emitter.id;
  }
}
