#include "hints.hpp"

#include "command_processor.hpp"

#include <dolphin/gx/GXCommandList.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace aurora::gx::hints {
namespace {
constexpr Module Log{"aurora::gx::hints"};

static_assert(sizeof(AuroraLight) == 72 && offsetof(AuroraLight, volumetricScale) == 68);
static_assert(sizeof(AuroraEnvironment) == 236 && offsetof(AuroraEnvironment, unitsPerMeter) == 56 &&
              offsetof(AuroraEnvironment, fogRangeTable) == 212);
static_assert(sizeof(AuroraTevPolicy) == 32);
static_assert(sizeof(AuroraInfluencer) == 32 && sizeof(AuroraWindSource) == 32 && sizeof(AuroraGrassPatch) == 32);
static_assert(sizeof(AuroraGrassLightSet) == 48);
static_assert(sizeof(AuroraCloudLayer) == 28 && sizeof(AuroraSky) == 164 && sizeof(AuroraFogVolume) == 64);

constexpr uint32_t TevPolicyReservedBits = 0xF0000000u;
constexpr uint8_t MaxCameraRole = AURORA_CAMERA_UI;

enum Counter : size_t { Decoded, Skipped, Malformed, Late, Bytes, CounterCount };

std::array<std::array<std::atomic_uint64_t, CounterCount>, IdCount> g_counters{};
std::atomic<const Sink*> g_sink{nullptr};
std::atomic_uint32_t g_warnings{0};
// FIFO thread only
std::array<bool, IdCount> g_warnedMalformed{};
std::array<bool, IdCount> g_warnedLate{};
// Game thread only
std::array<bool, IdCount> g_warnedEmitter{};

// Decoder scratch on the FIFO thread; the spans handed to callbacks point here
std::vector<AuroraLight> g_lights;
std::vector<Mat3x4<float>> g_joints;
std::vector<AuroraInfluencer> g_influencers;
std::vector<AuroraWindSource> g_windSources;

bool in_range(uint16_t id) noexcept { return id >= FirstId && id <= LastId; }

void add(uint16_t id, Counter counter, uint64_t value = 1) noexcept {
  g_counters[id - FirstId][counter].fetch_add(value, std::memory_order_relaxed);
}

struct IdInfo {
  std::string_view name;
  uint16_t version;
};

IdInfo id_info(uint16_t id) noexcept {
  switch (id) {
  case GX_AURORA_SET_ARRAY_IMMUTABLE:
    return {"SET_ARRAY_IMMUTABLE", AURORA_HINT_SET_ARRAY_IMMUTABLE_VERSION};
  case GX_AURORA_RELEASE_ARRAY_IMMUTABLE:
    return {"RELEASE_ARRAY_IMMUTABLE", AURORA_HINT_RELEASE_ARRAY_IMMUTABLE_VERSION};
  case GX_AURORA_RENDER_CAMERA:
    return {"CAMERA", AURORA_HINT_CAMERA_VERSION};
  case GX_AURORA_RENDER_OBJECT_BEGIN:
    return {"OBJECT_BEGIN", AURORA_HINT_OBJECT_BEGIN_VERSION};
  case GX_AURORA_RENDER_OBJECT_END:
    return {"OBJECT_END", AURORA_HINT_OBJECT_END_VERSION};
  case GX_AURORA_RENDER_SCOPE_PUSH:
    return {"SCOPE_PUSH", AURORA_HINT_SCOPE_PUSH_VERSION};
  case GX_AURORA_RENDER_SCOPE_POP:
    return {"SCOPE_POP", AURORA_HINT_SCOPE_POP_VERSION};
  case GX_AURORA_RENDER_STAGE:
    return {"STAGE", AURORA_HINT_STAGE_VERSION};
  case GX_AURORA_RENDER_LIGHTS:
    return {"LIGHTS", AURORA_HINT_LIGHTS_VERSION};
  case GX_AURORA_RENDER_ENVIRONMENT:
    return {"ENVIRONMENT", AURORA_HINT_ENVIRONMENT_VERSION};
  case GX_AURORA_RENDER_MATERIAL:
    return {"MATERIAL", AURORA_HINT_MATERIAL_VERSION};
  case GX_AURORA_RENDER_SKELETON:
    return {"SKELETON", AURORA_HINT_SKELETON_VERSION};
  case GX_AURORA_RENDER_SCENE_EPOCH:
    return {"SCENE_EPOCH", AURORA_HINT_SCENE_EPOCH_VERSION};
  case GX_AURORA_RENDER_TEV_POLICY:
    return {"TEV_POLICY", AURORA_HINT_TEV_POLICY_VERSION};
  case GX_AURORA_RENDER_INFLUENCERS:
    return {"INFLUENCERS", AURORA_HINT_INFLUENCERS_VERSION};
  case GX_AURORA_RENDER_WIND_SOURCES:
    return {"WIND_SOURCES", AURORA_HINT_WIND_SOURCES_VERSION};
  case GX_AURORA_RENDER_GRASS_PATCHES:
    return {"GRASS_PATCHES", AURORA_HINT_GRASS_PATCHES_VERSION};
  case GX_AURORA_RENDER_SKY:
    return {"SKY", AURORA_HINT_SKY_VERSION};
  case GX_AURORA_RENDER_FOG_VOLUMES:
    return {"FOG_VOLUMES", AURORA_HINT_FOG_VOLUMES_VERSION};
  // Payloads arrive together with their decoder
  case GX_AURORA_GPU_UPLOAD:
    return {"GPU_UPLOAD", 0};
  case GX_AURORA_GPU_RESERVE:
    return {"GPU_RESERVE", 0};
  case GX_AURORA_GPU_DISPATCH:
    return {"GPU_DISPATCH", 0};
  case GX_AURORA_LOAD_ARRAY_GPU:
    return {"LOAD_ARRAY_GPU", 0};
  case GX_AURORA_GPU_UPDATE_PERSISTENT:
    return {"GPU_UPDATE_PERSISTENT", 0};
  case GX_AURORA_GPU_CREATE_SYSTEM:
    return {"GPU_CREATE_SYSTEM", 0};
  case GX_AURORA_GPU_DESTROY_SYSTEM:
    return {"GPU_DESTROY_SYSTEM", 0};
  default:
    return {"reserved", 0};
  }
}

bool sink_handles(const Sink& sink, uint16_t id) noexcept {
  switch (id) {
  case GX_AURORA_RENDER_CAMERA:
    return sink.camera != nullptr;
  case GX_AURORA_RENDER_OBJECT_BEGIN:
    return sink.objectBegin != nullptr;
  case GX_AURORA_RENDER_OBJECT_END:
    return sink.objectEnd != nullptr;
  case GX_AURORA_RENDER_SCOPE_PUSH:
    return sink.scopePush != nullptr;
  case GX_AURORA_RENDER_SCOPE_POP:
    return sink.scopePop != nullptr;
  case GX_AURORA_RENDER_STAGE:
    return sink.stage != nullptr;
  case GX_AURORA_RENDER_LIGHTS:
    return sink.lights != nullptr;
  case GX_AURORA_RENDER_ENVIRONMENT:
    return sink.environment != nullptr;
  case GX_AURORA_RENDER_MATERIAL:
    return sink.material != nullptr;
  case GX_AURORA_RENDER_SKELETON:
    return sink.skeleton != nullptr;
  case GX_AURORA_RENDER_SCENE_EPOCH:
    return sink.sceneEpoch != nullptr;
  case GX_AURORA_RENDER_TEV_POLICY:
    return sink.tevPolicy != nullptr;
  case GX_AURORA_RENDER_INFLUENCERS:
    return sink.influencers != nullptr;
  case GX_AURORA_RENDER_WIND_SOURCES:
    return sink.windSources != nullptr;
  default:
    return false;
  }
}

struct Fault {
  enum Kind : uint8_t { None, NoVersion, Version, Size, Count, Value } kind = None;
  std::string_view field;
  uint64_t value = 0;
  uint64_t limit = 0;
};

void warn_rejected(uint16_t id, const Fault& fault) noexcept {
  if (std::exchange(g_warnedMalformed[id - FirstId], true)) {
    return;
  }
  g_warnings.fetch_add(1, std::memory_order_relaxed);
  const std::string_view hintName = id_info(id).name;
  switch (fault.kind) {
  case Fault::NoVersion:
    Log.warn("Render hint {} (0x{:04X}) rejected: {}-byte payload has no version; skipped", hintName, id, fault.value);
    break;
  case Fault::Version:
    Log.warn("Render hint {} (0x{:04X}) rejected: unsupported version {}; skipped", hintName, id, fault.value);
    break;
  case Fault::Size:
    Log.warn("Render hint {} (0x{:04X}) rejected: {}-byte payload, expected {}; skipped", hintName, id, fault.value,
             fault.limit);
    break;
  case Fault::Count:
    Log.warn("Render hint {} (0x{:04X}) rejected: count {} above the maximum {}; skipped", hintName, id, fault.value,
             fault.limit);
    break;
  case Fault::Value:
  case Fault::None:
    Log.warn("Render hint {} (0x{:04X}) rejected: {} 0x{:X} out of range; skipped", hintName, id, fault.field,
             fault.value);
    break;
  }
}

// Every check below runs before the bytes it guards are read, so ByteReader never asserts
class Decoder {
public:
  explicit Decoder(std::span<const uint8_t> payload) noexcept : m_in(payload) {}

  ByteReader& in() noexcept { return m_in; }
  [[nodiscard]] const Fault& fault() const noexcept { return m_fault; }

  bool version(uint16_t latest, uint16_t& out) noexcept {
    if (m_in.remaining() < sizeof(uint16_t)) {
      return fail({Fault::NoVersion, {}, m_in.size()});
    }
    out = m_in.read<uint16_t>();
    if (out == 0 || out > latest) {
      return fail({Fault::Version, {}, out});
    }
    return true;
  }

  bool size(size_t expected) noexcept {
    return m_in.size() == expected || fail({Fault::Size, {}, m_in.size(), expected});
  }

  // Guards the per-ID header of an array hint, which the caller reads next
  bool header(size_t header) noexcept { return m_in.size() >= header || fail({Fault::Size, {}, m_in.size(), header}); }

  bool count(uint32_t count, uint32_t max, size_t header, size_t element) noexcept {
    return (count <= max || fail({Fault::Count, {}, count, max})) && size(header + size_t{count} * element);
  }

  bool value(bool valid, std::string_view field, uint64_t value) noexcept {
    return valid || fail({Fault::Value, field, value});
  }

private:
  bool fail(const Fault& fault) noexcept {
    m_fault = fault;
    return false;
  }

  ByteReader m_in;
  Fault m_fault;
};

template <size_t N>
void read_floats(ByteReader& in, float (&out)[N]) noexcept {
  for (float& value : out) {
    value = in.read<float>();
  }
}

template <size_t N>
void write_floats(wire::Writer& out, const float (&values)[N]) noexcept {
  for (const float value : values) {
    out.f32(value);
  }
}

template <size_t N>
void read_bytes(ByteReader& in, uint8_t (&out)[N]) noexcept {
  std::memcpy(out, in.take(N).data(), N);
}

void read_mat(ByteReader& in, Mat3x4<float>& out) noexcept {
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      (&out.m0)[row][col] = in.read<float>();
    }
  }
}

void read_mat(ByteReader& in, Mat4x4<float>& out) noexcept {
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      out[row][col] = in.read<float>();
    }
  }
}

void set_environment_defaults(AuroraEnvironment& environment) noexcept {
  environment = {};
  environment.unitsPerMeter = 100.f;
  environment.grassDensityScale = 1.f;
  environment.hazeAtFogEnd = -1.f;
  environment.hazeDensity = -1.f;
  for (float& albedo : environment.mediaAlbedo) {
    albedo = -1.f;
  }
  // phaseG ranges over [-1, 1], so its engine-default sentinel is -2
  environment.phaseG = -2.f;
  environment.phaseG2 = -2.f;
  environment.phaseBlend = -1.f;
  environment.noiseScale = -1.f;
  environment.noiseAmount = -1.f;
  environment.relightStrength = -1.f;
  environment.relightAmbient = -1.f;
  environment.sunGlow = -1.f;
  environment.volumeFar = -1.f;
}

bool decode(const Sink& sink, uint16_t id, Decoder& decoder) noexcept {
  ByteReader& in = decoder.in();
  uint16_t version = 0;
  if (!decoder.version(id_info(id).version, version)) {
    return false;
  }
  switch (id) {
  case GX_AURORA_RENDER_CAMERA: {
    if (!decoder.size(wire::CameraV1)) {
      return false;
    }
    Camera camera;
    const uint8_t role = in.read<uint8_t>();
    read_mat(in, camera.view);
    read_mat(in, camera.proj);
    const uint8_t type = in.read<uint8_t>();
    camera.cutEpoch = in.read<uint32_t>();
    if (!decoder.value(role <= MaxCameraRole, "role", role) ||
        !decoder.value(type <= GX_ORTHOGRAPHIC, "projection type", type)) {
      return false;
    }
    camera.role = static_cast<AuroraCameraRole>(role);
    camera.type = static_cast<GXProjectionType>(type);
    camera.gxProj = gx_projection(camera.proj, camera.type);
    add(id, Decoded);
    sink.camera(sink.userData, camera);
    return true;
  }
  case GX_AURORA_RENDER_OBJECT_BEGIN: {
    if (!decoder.size(version == 1 ? wire::ObjectBeginV1 : wire::ObjectBeginV2)) {
      return false;
    }
    ObjectBegin object;
    object.id = in.read<uint64_t>();
    object.subId = in.read<uint32_t>();
    object.generation = in.read<uint32_t>();
    object.categories = in.read<uint32_t>();
    object.flags = in.read<uint32_t>();
    if (version >= 2) {
      object.hasWorld = true;
      read_mat(in, object.world);
    }
    add(id, Decoded);
    sink.objectBegin(sink.userData, object);
    return true;
  }
  case GX_AURORA_RENDER_OBJECT_END:
    if (!decoder.size(wire::EmptyV1)) {
      return false;
    }
    add(id, Decoded);
    sink.objectEnd(sink.userData);
    return true;
  case GX_AURORA_RENDER_SCOPE_PUSH: {
    if (!decoder.size(wire::ScopePushV1)) {
      return false;
    }
    ScopePush scope;
    scope.categories = in.read<uint32_t>();
    scope.flags = in.read<uint32_t>();
    const uint8_t role = in.read<uint8_t>();
    if (!decoder.value(role <= MaxCameraRole || role == AURORA_CAMERA_INHERIT, "role", role)) {
      return false;
    }
    scope.role = static_cast<AuroraCameraRole>(role);
    add(id, Decoded);
    sink.scopePush(sink.userData, scope);
    return true;
  }
  case GX_AURORA_RENDER_SCOPE_POP:
    if (!decoder.size(wire::EmptyV1)) {
      return false;
    }
    add(id, Decoded);
    sink.scopePop(sink.userData);
    return true;
  case GX_AURORA_RENDER_STAGE: {
    if (!decoder.size(wire::StageV1)) {
      return false;
    }
    const uint8_t stage = in.read<uint8_t>();
    // AfterFrame is native-only, so it is never a marker on the wire
    if (!decoder.value(stage >= AURORA_STAGE_SCENE_BEGIN && stage <= AURORA_STAGE_FRAME_END, "stage", stage)) {
      return false;
    }
    add(id, Decoded);
    sink.stage(sink.userData, static_cast<render::Stage>(stage - 1));
    return true;
  }
  case GX_AURORA_RENDER_LIGHTS: {
    if (!decoder.header(wire::CountedHeader)) {
      return false;
    }
    const uint16_t count = in.read<uint16_t>();
    if (!decoder.count(count, AURORA_MAX_LIGHTS, wire::CountedHeader, version == 1 ? wire::LightV1 : wire::LightV2)) {
      return false;
    }
    g_lights.resize(count);
    for (auto& light : g_lights) {
      if (!wire::read(in, light, version)) {
        return decoder.value(false, "light type", light.type);
      }
    }
    add(id, Decoded);
    sink.lights(sink.userData, g_lights);
    return true;
  }
  case GX_AURORA_RENDER_ENVIRONMENT: {
    if (!decoder.size(version == 1 ? wire::EnvironmentV1 : wire::EnvironmentV2)) {
      return false;
    }
    Environment environment{.version = version};
    wire::read(in, environment.value, version);
    add(id, Decoded);
    sink.environment(sink.userData, environment);
    return true;
  }
  case GX_AURORA_RENDER_MATERIAL: {
    if (!decoder.size(wire::MaterialV1)) {
      return false;
    }
    const uint32_t materialClass = in.read<uint32_t>();
    add(id, Decoded);
    sink.material(sink.userData, materialClass);
    return true;
  }
  case GX_AURORA_RENDER_SKELETON: {
    if (!decoder.header(wire::SkeletonHeader)) {
      return false;
    }
    Skeleton skeleton;
    skeleton.objectId = in.read<uint64_t>();
    const uint16_t count = in.read<uint16_t>();
    if (!decoder.count(count, AURORA_MAX_SKELETON_JOINTS, wire::SkeletonHeader, wire::Joint)) {
      return false;
    }
    g_joints.resize(count);
    for (auto& joint : g_joints) {
      read_mat(in, joint);
    }
    skeleton.joints = g_joints;
    add(id, Decoded);
    sink.skeleton(sink.userData, skeleton);
    return true;
  }
  case GX_AURORA_RENDER_SCENE_EPOCH: {
    if (!decoder.size(wire::SceneEpochV1)) {
      return false;
    }
    const uint32_t epoch = in.read<uint32_t>();
    add(id, Decoded);
    sink.sceneEpoch(sink.userData, epoch);
    return true;
  }
  case GX_AURORA_RENDER_TEV_POLICY: {
    if (!decoder.size(wire::TevPolicyV1)) {
      return false;
    }
    AuroraTevPolicy policy;
    if (!wire::read(in, policy)) {
      return decoder.value(false, "reserved TEV policy bits", policy.mask & TevPolicyReservedBits);
    }
    add(id, Decoded);
    sink.tevPolicy(sink.userData, policy);
    return true;
  }
  case GX_AURORA_RENDER_INFLUENCERS: {
    if (!decoder.header(wire::CountedHeader)) {
      return false;
    }
    const uint16_t count = in.read<uint16_t>();
    if (!decoder.count(count, AURORA_MAX_INFLUENCERS, wire::CountedHeader, wire::Influencer)) {
      return false;
    }
    g_influencers.resize(count);
    for (auto& influencer : g_influencers) {
      wire::read(in, influencer);
    }
    add(id, Decoded);
    sink.influencers(sink.userData, g_influencers);
    return true;
  }
  case GX_AURORA_RENDER_WIND_SOURCES: {
    if (!decoder.header(wire::CountedHeader)) {
      return false;
    }
    const uint16_t count = in.read<uint16_t>();
    if (!decoder.count(count, AURORA_MAX_WIND_SOURCES, wire::CountedHeader, wire::WindSource)) {
      return false;
    }
    g_windSources.resize(count);
    for (auto& source : g_windSources) {
      wire::read(in, source);
    }
    add(id, Decoded);
    sink.windSources(sink.userData, g_windSources);
    return true;
  }
  default:
    return false;
  }
}

bool decode_core(uint16_t id, Decoder& decoder) noexcept {
  ByteReader& in = decoder.in();
  uint16_t version = 0;
  if (!decoder.version(id_info(id).version, version)) {
    return false;
  }
  if (id == GX_AURORA_SET_ARRAY_IMMUTABLE) {
    if (!decoder.size(wire::SetArrayImmutableV1)) {
      return false;
    }
    const uint64_t address = in.read<uint64_t>();
    const uint32_t size = in.read<uint32_t>();
    const uint8_t le = in.read<uint8_t>();
    const uint32_t generation = in.read<uint32_t>();
    if (!decoder.value(le <= 1, "endianness", le) ||
        !decoder.value(address != 0 && address <= UINTPTR_MAX, "address", address) ||
        !decoder.value(size != 0, "size", size)) {
      return false;
    }
    add(id, Decoded);
    fifo::set_array_immutable(address, size, le == 1, generation);
    return true;
  }
  if (!decoder.size(wire::ReleaseArrayImmutableV1)) {
    return false;
  }
  const uint64_t address = in.read<uint64_t>();
  const uint32_t generation = in.read<uint32_t>();
  add(id, Decoded);
  fifo::release_array_immutable(address, generation);
  return true;
}
} // namespace

void set_sink(const Sink* sink) noexcept { g_sink.store(sink, std::memory_order_release); }

const Sink* sink() noexcept { return g_sink.load(std::memory_order_acquire); }

void reset_state() noexcept {
  fifo::reset_immutable_arrays();
  if (const Sink* current = sink(); current != nullptr && current->reset != nullptr) {
    current->reset(current->userData);
  }
}

void handle(uint16_t id, ByteReader& reader) noexcept {
  AURORA_ASSERT(in_range(id), "GX_AURORA subcommand {:04X} is not a render hint", id);
  const u32 payloadSize = reader.read<u32>();
  AURORA_ASSERT(payloadSize <= reader.remaining(),
                "GX_AURORA hint {:04X}: {}-byte payload overruns the command buffer ({} bytes left)", id, payloadSize,
                reader.remaining());
  const auto payload = reader.take(payloadSize);
  add(id, Bytes, GX_AURORA_HINT_HEADER_SIZE + payloadSize);

  // Core hints act without the render module, which may be off or compiled out
  if (id == GX_AURORA_SET_ARRAY_IMMUTABLE || id == GX_AURORA_RELEASE_ARRAY_IMMUTABLE) {
    Decoder decoder{payload};
    if (!decode_core(id, decoder)) {
      add(id, Malformed);
      warn_rejected(id, decoder.fault());
    }
    return;
  }

  const Sink* current = g_sink.load(std::memory_order_acquire);
  if (current == nullptr || !sink_handles(*current, id)) {
    add(id, Skipped);
    return;
  }
  Decoder decoder{payload};
  if (!decode(*current, id, decoder)) {
    add(id, Malformed);
    warn_rejected(id, decoder.fault());
  }
}

Counters counters(uint16_t id) noexcept {
  if (!in_range(id)) {
    return {};
  }
  const auto& values = g_counters[id - FirstId];
  return {
      .decoded = values[Decoded].load(std::memory_order_relaxed),
      .skipped = values[Skipped].load(std::memory_order_relaxed),
      .malformed = values[Malformed].load(std::memory_order_relaxed),
      .late = values[Late].load(std::memory_order_relaxed),
      .bytes = values[Bytes].load(std::memory_order_relaxed),
  };
}

uint64_t total_bytes() noexcept {
  uint64_t total = 0;
  for (const auto& values : g_counters) {
    total += values[Bytes].load(std::memory_order_relaxed);
  }
  return total;
}

void count_late(uint16_t id) noexcept {
  if (!in_range(id)) {
    return;
  }
  add(id, Late);
  if (!std::exchange(g_warnedLate[id - FirstId], true)) {
    g_warnings.fetch_add(1, std::memory_order_relaxed);
    Log.warn("Render hint {} (0x{:04X}) arrived after the frame's scene hints froze; ignored for this frame",
             id_info(id).name, id);
  }
}

bool emitter_warned(uint16_t id) noexcept { return !in_range(id) || g_warnedEmitter[id - FirstId]; }

void warn_emitter(uint16_t id, std::string_view problem) noexcept {
  if (!in_range(id) || std::exchange(g_warnedEmitter[id - FirstId], true)) {
    return;
  }
  g_warnings.fetch_add(1, std::memory_order_relaxed);
  Log.warn("Render hint {} (0x{:04X}): {}", id_info(id).name, id, problem);
}

std::string_view name(uint16_t id) noexcept { return id_info(id).name; }

uint16_t latest_version(uint16_t id) noexcept { return id_info(id).version; }

Mat4x4<float> gx_projection(const Mat4x4<float>& proj, GXProjectionType type) noexcept {
  // GXSetProjection keeps six parameters and xf_load_projection rebuilds the rest as +0.0
  const bool ortho = type == GX_ORTHOGRAPHIC;
  Mat4x4<float> out{};
  out.m0[0] = proj[0][0];
  out.m1[1] = proj[1][1];
  out.m2[2] = proj[2][2];
  out.m2[3] = proj[2][3];
  if (ortho) {
    out.m0[3] = proj[0][3];
    out.m1[3] = proj[1][3];
    out.m3[3] = 1.f;
  } else {
    out.m0[2] = proj[0][2];
    out.m1[2] = proj[1][2];
    out.m3[2] = -1.f;
  }
  return out;
}

bool main_camera_usable(const Mat4x4<float>& gxProj) noexcept {
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      if (!std::isfinite(gxProj[row][col])) {
        return false;
      }
    }
  }
  return gxProj[3][2] == -1.f && gxProj[3][3] == 0.f && gxProj[0][0] != 0.f && gxProj[1][1] != 0.f &&
         gxProj[2][3] != 0.f;
}

namespace wire {
void Writer::u8(uint8_t v) noexcept { bytes({&v, 1}); }

void Writer::u16(uint16_t v) noexcept {
  const uint16_t be = bswap(v);
  bytes({reinterpret_cast<const uint8_t*>(&be), sizeof(be)});
}

void Writer::u32(uint32_t v) noexcept {
  const uint32_t be = bswap(v);
  bytes({reinterpret_cast<const uint8_t*>(&be), sizeof(be)});
}

void Writer::u64(uint64_t v) noexcept {
  const uint64_t be = bswap(v);
  bytes({reinterpret_cast<const uint8_t*>(&be), sizeof(be)});
}

void Writer::f32(float v) noexcept {
  const float be = bswap(v);
  bytes({reinterpret_cast<const uint8_t*>(&be), sizeof(be)});
}

void Writer::bytes(std::span<const uint8_t> v) noexcept {
  AURORA_ASSERT(v.size() <= m_out.size() - m_pos, "Render hint writer overrun: {} bytes at {} of {}", v.size(), m_pos,
                m_out.size());
  std::memcpy(m_out.data() + m_pos, v.data(), v.size());
  m_pos += v.size();
}

void Writer::header(uint16_t id, uint32_t payloadSize) noexcept {
  u8(GX_AURORA);
  u16(id);
  u32(payloadSize);
}

void write(Writer& out, const AuroraLight& light, uint16_t version) noexcept {
  out.u32(light.id);
  out.u8(light.type);
  out.u8(light.shadowPriority);
  out.u16(light.flags);
  out.u32(light.categoryMask);
  write_floats(out, light.position);
  write_floats(out, light.direction);
  write_floats(out, light.color);
  out.f32(light.radius);
  out.f32(light.sourceRadius);
  out.f32(light.spotInnerCos);
  out.f32(light.spotOuterCos);
  out.f32(light.falloffExponent);
  if (version >= 2) {
    out.f32(light.volumetricScale);
  }
}

void write(Writer& out, const AuroraEnvironment& environment, uint16_t version) noexcept {
  out.f32(environment.skyVisibility);
  write_floats(out, environment.ambientColor);
  write_floats(out, environment.fogColor);
  write_floats(out, environment.windDir);
  out.f32(environment.windStrength);
  out.f32(environment.exposureBias);
  out.f32(environment.timeOfDay);
  out.u32(environment.flags);
  if (version < 2) {
    return;
  }
  out.f32(environment.unitsPerMeter);
  write_floats(out, environment.windPhase);
  write_floats(out, environment.windPhasePrev);
  write_floats(out, environment.flutterPhase);
  out.u32(environment.fogType);
  out.f32(environment.fogStart);
  out.f32(environment.fogEnd);
  out.f32(environment.viewNear);
  out.f32(environment.viewFar);
  out.u32(environment.fogRangeAdj);
  out.f32(environment.shadowSun);
  out.f32(environment.shadowLocal);
  out.f32(environment.shadowThunderAdd);
  out.f32(environment.grassDensityScale);
  out.f32(environment.sunShaftGate);
  out.f32(environment.hazeAtFogEnd);
  out.f32(environment.hazeDensity);
  out.f32(environment.heightFogDensity);
  out.f32(environment.heightFogBase);
  out.f32(environment.heightFogFalloff);
  write_floats(out, environment.mediaAlbedo);
  write_floats(out, environment.mediaEmissive);
  out.f32(environment.phaseG);
  out.f32(environment.phaseG2);
  out.f32(environment.phaseBlend);
  out.f32(environment.noiseScale);
  out.f32(environment.noiseAmount);
  out.f32(environment.relightStrength);
  out.f32(environment.relightAmbient);
  out.f32(environment.sunGlow);
  out.f32(environment.volumeFar);
  for (const uint16_t entry : environment.fogRangeTable) {
    out.u16(entry);
  }
  for (const uint16_t entry : environment.fogRangeReserved) {
    out.u16(entry);
  }
}

void write(Writer& out, const AuroraTevPolicy& policy) noexcept {
  out.u32(policy.mask);
  for (const auto& channels : policy.neutral) {
    out.bytes(channels);
  }
}

void write(Writer& out, const AuroraInfluencer& influencer) noexcept {
  write_floats(out, influencer.pos);
  out.f32(influencer.radius);
  write_floats(out, influencer.vel);
  out.u32(influencer.kind);
}

void write(Writer& out, const AuroraWindSource& source) noexcept {
  write_floats(out, source.pos);
  out.f32(source.radius);
  write_floats(out, source.dir);
  out.f32(source.power);
}

void write(Writer& out, const AuroraGrassLightSet& set) noexcept {
  out.bytes(set.reg1Rgb);
  out.u8(set.shine);
  out.bytes(set.matRgba);
  out.bytes(set.lightIdx);
  out.u8(set.flags);
  out.f32(set.grassLightRatio);
  write_floats(out, set.fog);
  out.bytes(set.fogRgb);
  out.u8(set.fogType);
  out.u32(set.reserved);
}

void write(Writer& out, const AuroraSky& sky) noexcept {
  out.u32(sky.flags);
  out.f32(sky.horizonY);
  out.u32(sky.cloudTicks);
  out.f32(sky.cloudTickFrac);
  out.f32(sky.tickSeconds);
  out.u32(sky.seed);
  out.f32(sky.cloudSpeedScale);
  out.f32(sky.cloudShadowDensity);
  out.f32(sky.rain);
  out.f32(sky.thunderFlash);
  out.f32(sky.weatherAmount);
  write_floats(out, sky.skyColor);
  write_floats(out, sky.cloudTop);
  write_floats(out, sky.cloudBottom);
  write_floats(out, sky.cloudShadow);
  write_floats(out, sky.hazeOuter);
  out.u32(sky.layerCount);
  for (const auto& layer : sky.layers) {
    out.f32(layer.base);
    out.f32(layer.thickness);
    out.f32(layer.coverage);
    out.f32(layer.density);
    out.u32(layer.type);
    out.f32(layer.windScale);
    out.u32(layer.flags);
  }
}

void write(Writer& out, const AuroraFogVolume& volume) noexcept {
  write_floats(out, volume.center);
  write_floats(out, volume.radius);
  out.f32(volume.rotationY);
  out.u8(volume.shape);
  out.u8(volume.mode);
  out.u16(volume.flags);
  out.f32(volume.density);
  out.f32(volume.edgeFalloff);
  write_floats(out, volume.albedo);
  write_floats(out, volume.emissive);
}

bool read(ByteReader& in, AuroraLight& out, uint16_t version) noexcept {
  if ((version != 1 && version != 2) || in.remaining() < (version == 1 ? LightV1 : LightV2)) {
    return false;
  }
  out.id = in.read<uint32_t>();
  out.type = in.read<uint8_t>();
  out.shadowPriority = in.read<uint8_t>();
  out.flags = in.read<uint16_t>();
  out.categoryMask = in.read<uint32_t>();
  read_floats(in, out.position);
  read_floats(in, out.direction);
  read_floats(in, out.color);
  out.radius = in.read<float>();
  out.sourceRadius = in.read<float>();
  out.spotInnerCos = in.read<float>();
  out.spotOuterCos = in.read<float>();
  out.falloffExponent = in.read<float>();
  out.volumetricScale = version >= 2 ? in.read<float>() : 1.f;
  return out.type <= AURORA_LIGHT_SPOT;
}

bool read(ByteReader& in, AuroraEnvironment& out, uint16_t version) noexcept {
  const size_t size = (version == 1 ? EnvironmentV1 : EnvironmentV2) - sizeof(uint16_t);
  if ((version != 1 && version != 2) || in.remaining() < size) {
    return false;
  }
  set_environment_defaults(out);
  out.skyVisibility = in.read<float>();
  read_floats(in, out.ambientColor);
  read_floats(in, out.fogColor);
  read_floats(in, out.windDir);
  out.windStrength = in.read<float>();
  out.exposureBias = in.read<float>();
  out.timeOfDay = in.read<float>();
  out.flags = in.read<uint32_t>();
  if (version < 2) {
    return true;
  }
  out.unitsPerMeter = in.read<float>();
  read_floats(in, out.windPhase);
  read_floats(in, out.windPhasePrev);
  read_floats(in, out.flutterPhase);
  out.fogType = in.read<uint32_t>();
  out.fogStart = in.read<float>();
  out.fogEnd = in.read<float>();
  out.viewNear = in.read<float>();
  out.viewFar = in.read<float>();
  out.fogRangeAdj = in.read<uint32_t>();
  out.shadowSun = in.read<float>();
  out.shadowLocal = in.read<float>();
  out.shadowThunderAdd = in.read<float>();
  out.grassDensityScale = in.read<float>();
  out.sunShaftGate = in.read<float>();
  out.hazeAtFogEnd = in.read<float>();
  out.hazeDensity = in.read<float>();
  out.heightFogDensity = in.read<float>();
  out.heightFogBase = in.read<float>();
  out.heightFogFalloff = in.read<float>();
  read_floats(in, out.mediaAlbedo);
  read_floats(in, out.mediaEmissive);
  out.phaseG = in.read<float>();
  out.phaseG2 = in.read<float>();
  out.phaseBlend = in.read<float>();
  out.noiseScale = in.read<float>();
  out.noiseAmount = in.read<float>();
  out.relightStrength = in.read<float>();
  out.relightAmbient = in.read<float>();
  out.sunGlow = in.read<float>();
  out.volumeFar = in.read<float>();
  for (uint16_t& entry : out.fogRangeTable) {
    entry = in.read<uint16_t>();
  }
  for (uint16_t& entry : out.fogRangeReserved) {
    entry = in.read<uint16_t>();
  }
  return true;
}

bool read(ByteReader& in, AuroraTevPolicy& out) noexcept {
  if (in.remaining() < sizeof(AuroraTevPolicy)) {
    return false;
  }
  out.mask = in.read<uint32_t>();
  for (auto& channels : out.neutral) {
    read_bytes(in, channels);
  }
  return (out.mask & TevPolicyReservedBits) == 0;
}

bool read(ByteReader& in, AuroraInfluencer& out) noexcept {
  if (in.remaining() < Influencer) {
    return false;
  }
  read_floats(in, out.pos);
  out.radius = in.read<float>();
  read_floats(in, out.vel);
  out.kind = in.read<uint32_t>();
  return true;
}

bool read(ByteReader& in, AuroraWindSource& out) noexcept {
  if (in.remaining() < WindSource) {
    return false;
  }
  read_floats(in, out.pos);
  out.radius = in.read<float>();
  read_floats(in, out.dir);
  out.power = in.read<float>();
  return true;
}

bool read(ByteReader& in, AuroraGrassLightSet& out) noexcept {
  if (in.remaining() < GrassLightSet) {
    return false;
  }
  read_bytes(in, out.reg1Rgb);
  out.shine = in.read<uint8_t>();
  read_bytes(in, out.matRgba);
  read_bytes(in, out.lightIdx);
  out.flags = in.read<uint8_t>();
  out.grassLightRatio = in.read<float>();
  read_floats(in, out.fog);
  read_bytes(in, out.fogRgb);
  out.fogType = in.read<uint8_t>();
  out.reserved = in.read<uint32_t>();
  return true;
}

bool read(ByteReader& in, AuroraSky& out) noexcept {
  if (in.remaining() < SkyV1 - sizeof(uint16_t)) {
    return false;
  }
  out.flags = in.read<uint32_t>();
  out.horizonY = in.read<float>();
  out.cloudTicks = in.read<uint32_t>();
  out.cloudTickFrac = in.read<float>();
  out.tickSeconds = in.read<float>();
  out.seed = in.read<uint32_t>();
  out.cloudSpeedScale = in.read<float>();
  out.cloudShadowDensity = in.read<float>();
  out.rain = in.read<float>();
  out.thunderFlash = in.read<float>();
  out.weatherAmount = in.read<float>();
  read_floats(in, out.skyColor);
  read_floats(in, out.cloudTop);
  read_floats(in, out.cloudBottom);
  read_floats(in, out.cloudShadow);
  read_floats(in, out.hazeOuter);
  out.layerCount = in.read<uint32_t>();
  for (auto& layer : out.layers) {
    layer.base = in.read<float>();
    layer.thickness = in.read<float>();
    layer.coverage = in.read<float>();
    layer.density = in.read<float>();
    layer.type = in.read<uint32_t>();
    layer.windScale = in.read<float>();
    layer.flags = in.read<uint32_t>();
  }
  return out.layerCount <= 2;
}

bool read(ByteReader& in, AuroraFogVolume& out) noexcept {
  if (in.remaining() < FogVolume) {
    return false;
  }
  read_floats(in, out.center);
  read_floats(in, out.radius);
  out.rotationY = in.read<float>();
  out.shape = in.read<uint8_t>();
  out.mode = in.read<uint8_t>();
  out.flags = in.read<uint16_t>();
  out.density = in.read<float>();
  out.edgeFalloff = in.read<float>();
  read_floats(in, out.albedo);
  read_floats(in, out.emissive);
  return true;
}
} // namespace wire

namespace testing {
void reset_counters() noexcept {
  for (auto& values : g_counters) {
    for (auto& value : values) {
      value.store(0, std::memory_order_relaxed);
    }
  }
  g_warnedMalformed = {};
  g_warnedLate = {};
  g_warnedEmitter = {};
  g_warnings.store(0, std::memory_order_relaxed);
}

uint32_t warnings() noexcept { return g_warnings.load(std::memory_order_relaxed); }
} // namespace testing
} // namespace aurora::gx::hints
