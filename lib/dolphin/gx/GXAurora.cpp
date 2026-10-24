#include "dolphin/gx/GXAurora.h"

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "__gx.h"
#include "dolphin/gd/GDGeometry.h"
#include "gx.hpp"
#include "../../window.hpp"

#include "../../gx/fifo.hpp"
#include "../../gx/gpu_channel.hpp"
#include "../../gx/hints.hpp"

namespace {
namespace hints = aurora::gx::hints;
namespace wire = aurora::gx::hints::wire;

// Header and payload go out in one write, so a display list that is full drops the whole hint
template <u32 PayloadSize, typename Fill>
void write_fixed_hint(u16 id, u16 version, Fill&& fill) {
  std::array<u8, GX_AURORA_HINT_HEADER_SIZE + PayloadSize> buffer{};
  wire::Writer out{buffer};
  out.header(id, PayloadSize);
  out.u16(version);
  fill(out);
  CHECK(out.size() == buffer.size(), "render hint {:04X} wrote {} of {} bytes", id, out.size(), buffer.size());
  GX_WRITE_DATA(buffer.data(), static_cast<u32>(buffer.size()));
}

template <typename Fill>
void write_array_hint(u16 id, u16 version, u32 payloadSize, Fill&& fill) {
  thread_local std::vector<u8> buffer;
  buffer.resize(GX_AURORA_HINT_HEADER_SIZE + payloadSize);
  wire::Writer out{buffer};
  out.header(id, payloadSize);
  out.u16(version);
  fill(out);
  CHECK(out.size() == buffer.size(), "render hint {:04X} wrote {} of {} bytes", id, out.size(), buffer.size());
  GX_WRITE_DATA(buffer.data(), static_cast<u32>(buffer.size()));
}

template <typename T>
u32 checked_count(u16 id, const T* items, u32 count, u32 max) {
  if (items == nullptr && count != 0) {
    hints::warn_emitter(id, "null array with a non-zero count; sending none");
    return 0;
  }
  if (count > max) {
    if (!hints::emitter_warned(id)) {
      hints::warn_emitter(id, fmt::format("count {} above the maximum {}; clamped", count, max));
    }
    return max;
  }
  return count;
}

void write_rows(wire::Writer& out, const f32 (*rows)[4], size_t count) {
  for (size_t row = 0; row < count; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      out.f32(rows[row][col]);
    }
  }
}

void write_object(wire::Writer& out, u64 objectId, u32 subId, u32 generation, u32 categories, u32 flags) {
  out.u64(objectId);
  out.u32(subId);
  out.u32(generation);
  out.u32(categories);
  out.u32(flags);
}
} // namespace

static void GXWriteString(const char* label) {
  auto length = strlen(label);

  if (length > std::numeric_limits<u16>::max()) {
    Log.warn("Debug marker size over u16 max, truncating");
    length = std::numeric_limits<u16>::max();
  }

  GX_WRITE_U16(length);
  GX_WRITE_DATA(label, length);
}

void GXPushDebugGroup(const char* label) {
  GX_WRITE_AURORA(GX_AURORA_DEBUG_GROUP_PUSH);
  GXWriteString(label);
}

void GXPopDebugGroup() { GX_WRITE_AURORA(GX_AURORA_DEBUG_GROUP_POP); }

void GXInsertDebugMarker(const char* label) {
  GX_WRITE_AURORA(GX_AURORA_DEBUG_MARKER_INSERT);
  GXWriteString(label);
}

void AuroraSetViewportPolicy(AuroraViewportPolicy policy) {
  aurora::gx::set_viewport_policy(policy);
}

void AuroraGetRenderSize(u32* width, u32* height) {
  const auto windowSize = aurora::window::get_window_size();
  if (width != nullptr) {
    *width = windowSize.fb_width;
  }
  if (height != nullptr) {
    *height = windowSize.fb_height;
  }
}

void AuroraGXSync() {
  GXFlush();
  aurora::gx::fifo::drain();
}

void GXSetViewportRender(f32 left, f32 top, f32 wd, f32 ht, f32 nearz, f32 farz) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_VIEWPORT_RENDER);
  GX_WRITE_F32(left);
  GX_WRITE_F32(top);
  GX_WRITE_F32(wd);
  GX_WRITE_F32(ht);
  GX_WRITE_F32(nearz);
  GX_WRITE_F32(farz);
}

void GXSetScissorRender(u32 left, u32 top, u32 wd, u32 ht) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_SCISSOR_RENDER);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetProjectionFull(const void* mtx) {
  const f32* values = reinterpret_cast<const f32*>(mtx);
  GX_WRITE_AURORA(GX_AURORA_LOAD_PROJECTION_FULL);
  for (int i = 0; i < 16; ++i) {
    GX_WRITE_F32(values[i]);
  }
}

void GX2SetPolygonOffset(f32 mFrontOffset, f32 mFrontScale, f32 mBackOffset, f32 mBackScale, f32 mClamp) {
  GX_WRITE_AURORA(GX2_SET_POLYGON_OFFSET);
  GX_WRITE_F32(mFrontOffset);
  GX_WRITE_F32(mFrontScale);
  GX_WRITE_F32(mBackOffset);
  GX_WRITE_F32(mBackScale);
  GX_WRITE_F32(mClamp);
}

void GXCreateFrameBuffer(u32 width, u32 height) {
  GX_WRITE_AURORA(GX_AURORA_BEGIN_OFFSCREEN);
  GX_WRITE_U32(width);
  GX_WRITE_U32(height);
  aurora::gx::fifo::publish();
}

void GXRestoreFrameBuffer() {
  GX_WRITE_AURORA(GX_AURORA_END_OFFSCREEN);
  aurora::gx::fifo::publish();
}

void GXAuroraSetArrayImmutable(const void* data, u32 size, GXBool littleEndian, u32 generation) {
  write_fixed_hint<wire::SetArrayImmutableV1>(GX_AURORA_SET_ARRAY_IMMUTABLE, AURORA_HINT_SET_ARRAY_IMMUTABLE_VERSION,
                                              [&](wire::Writer& out) {
                                                out.u64(reinterpret_cast<uintptr_t>(data));
                                                out.u32(size);
                                                out.u8(littleEndian ? 1 : 0);
                                                out.u32(generation);
                                              });
}

void GXAuroraReleaseArrayImmutable(const void* data, u32 generation) {
  write_fixed_hint<wire::ReleaseArrayImmutableV1>(GX_AURORA_RELEASE_ARRAY_IMMUTABLE,
                                                  AURORA_HINT_RELEASE_ARRAY_IMMUTABLE_VERSION, [&](wire::Writer& out) {
                                                    out.u64(reinterpret_cast<uintptr_t>(data));
                                                    out.u32(generation);
                                                  });
}

void GXAuroraSetCamera(AuroraCameraRole role, const f32 view[3][4], const f32 proj[4][4], GXProjectionType type,
                       u32 cutEpoch) {
  write_fixed_hint<wire::CameraV1>(GX_AURORA_RENDER_CAMERA, AURORA_HINT_CAMERA_VERSION, [&](wire::Writer& out) {
    out.u8(static_cast<u8>(role));
    write_rows(out, view, 3);
    write_rows(out, proj, 4);
    out.u8(static_cast<u8>(type));
    out.u32(cutEpoch);
  });
}

void GXAuroraBeginObject(u64 objectId, u32 subId, u32 generation, u32 categories, u32 flags) {
  // Version 1 carries no world matrix, so producers that have none send the smaller hint
  write_fixed_hint<wire::ObjectBeginV1>(GX_AURORA_RENDER_OBJECT_BEGIN, 1, [&](wire::Writer& out) {
    write_object(out, objectId, subId, generation, categories, flags);
  });
}

void GXAuroraBeginObjectWorld(u64 objectId, u32 subId, u32 generation, u32 categories, u32 flags,
                              const f32 world[3][4]) {
  write_fixed_hint<wire::ObjectBeginV2>(GX_AURORA_RENDER_OBJECT_BEGIN, 2, [&](wire::Writer& out) {
    write_object(out, objectId, subId, generation, categories, flags);
    write_rows(out, world, 3);
  });
}

void GXAuroraEndObject() {
  write_fixed_hint<wire::EmptyV1>(GX_AURORA_RENDER_OBJECT_END, AURORA_HINT_OBJECT_END_VERSION, [](wire::Writer&) {});
}

void GXAuroraPushScope(u32 categories, u32 flags, AuroraCameraRole role) {
  write_fixed_hint<wire::ScopePushV1>(GX_AURORA_RENDER_SCOPE_PUSH, AURORA_HINT_SCOPE_PUSH_VERSION,
                                      [&](wire::Writer& out) {
                                        out.u32(categories);
                                        out.u32(flags);
                                        out.u8(static_cast<u8>(role));
                                      });
}

void GXAuroraPopScope() {
  write_fixed_hint<wire::EmptyV1>(GX_AURORA_RENDER_SCOPE_POP, AURORA_HINT_SCOPE_POP_VERSION, [](wire::Writer&) {});
}

void GXAuroraStage(AuroraRenderStage stage) {
  write_fixed_hint<wire::StageV1>(GX_AURORA_RENDER_STAGE, AURORA_HINT_STAGE_VERSION,
                                  [&](wire::Writer& out) { out.u8(static_cast<u8>(stage)); });
}

void GXAuroraSetLights(const AuroraLight* lights, u16 count) {
  const u32 sent = checked_count(GX_AURORA_RENDER_LIGHTS, lights, count, AURORA_MAX_LIGHTS);
  write_array_hint(GX_AURORA_RENDER_LIGHTS, AURORA_HINT_LIGHTS_VERSION, wire::CountedHeader + sent * wire::LightV2,
                   [&](wire::Writer& out) {
                     out.u16(static_cast<u16>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       wire::write(out, lights[i], AURORA_HINT_LIGHTS_VERSION);
                     }
                   });
}

void GXAuroraSetEnvironment(const AuroraEnvironment* environment) {
  if (environment == nullptr) {
    hints::warn_emitter(GX_AURORA_RENDER_ENVIRONMENT, "null environment; nothing sent");
    return;
  }
  write_fixed_hint<wire::EnvironmentV2>(
      GX_AURORA_RENDER_ENVIRONMENT, AURORA_HINT_ENVIRONMENT_VERSION,
      [&](wire::Writer& out) { wire::write(out, *environment, AURORA_HINT_ENVIRONMENT_VERSION); });
}

void GXAuroraSetMaterialClass(u32 materialClass) {
  write_fixed_hint<wire::MaterialV1>(GX_AURORA_RENDER_MATERIAL, AURORA_HINT_MATERIAL_VERSION,
                                     [&](wire::Writer& out) { out.u32(materialClass); });
}

void GXAuroraSetSkeleton(u64 objectId, const f32 (*joints)[3][4], u16 count) {
  const u32 sent = checked_count(GX_AURORA_RENDER_SKELETON, joints, count, AURORA_MAX_SKELETON_JOINTS);
  write_array_hint(GX_AURORA_RENDER_SKELETON, AURORA_HINT_SKELETON_VERSION, wire::SkeletonHeader + sent * wire::Joint,
                   [&](wire::Writer& out) {
                     out.u64(objectId);
                     out.u16(static_cast<u16>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       write_rows(out, joints[i], 3);
                     }
                   });
}

void GXAuroraSetSceneEpoch(u32 epoch) {
  write_fixed_hint<wire::SceneEpochV1>(GX_AURORA_RENDER_SCENE_EPOCH, AURORA_HINT_SCENE_EPOCH_VERSION,
                                       [&](wire::Writer& out) { out.u32(epoch); });
}

void GXAuroraSetTevPolicy(const AuroraTevPolicy* policy) {
  if (policy == nullptr) {
    hints::warn_emitter(GX_AURORA_RENDER_TEV_POLICY, "null policy; nothing sent");
    return;
  }
  write_fixed_hint<wire::TevPolicyV1>(GX_AURORA_RENDER_TEV_POLICY, AURORA_HINT_TEV_POLICY_VERSION,
                                      [&](wire::Writer& out) { wire::write(out, *policy); });
}

void GXAuroraSetInfluencers(const AuroraInfluencer* influencers, u16 count) {
  const u32 sent = checked_count(GX_AURORA_RENDER_INFLUENCERS, influencers, count, AURORA_MAX_INFLUENCERS);
  write_array_hint(GX_AURORA_RENDER_INFLUENCERS, AURORA_HINT_INFLUENCERS_VERSION,
                   wire::CountedHeader + sent * wire::Influencer, [&](wire::Writer& out) {
                     out.u16(static_cast<u16>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       wire::write(out, influencers[i]);
                     }
                   });
}

void GXAuroraSetWindSources(const AuroraWindSource* sources, u16 count) {
  const u32 sent = checked_count(GX_AURORA_RENDER_WIND_SOURCES, sources, count, AURORA_MAX_WIND_SOURCES);
  write_array_hint(GX_AURORA_RENDER_WIND_SOURCES, AURORA_HINT_WIND_SOURCES_VERSION,
                   wire::CountedHeader + sent * wire::WindSource, [&](wire::Writer& out) {
                     out.u16(static_cast<u16>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       wire::write(out, sources[i]);
                     }
                   });
}

void GXAuroraSetGrassPatches(u32 gpuHandle, u32 count, f32 interpStep, const AuroraGrassLightSet* sets, u8 setCount) {
  const u32 sent = checked_count(GX_AURORA_RENDER_GRASS_PATCHES, sets, setCount, AURORA_MAX_GRASS_LIGHT_SETS);
  write_array_hint(GX_AURORA_RENDER_GRASS_PATCHES, AURORA_HINT_GRASS_PATCHES_VERSION,
                   wire::GrassPatchesHeader + sent * wire::GrassLightSet, [&](wire::Writer& out) {
                     out.u32(gpuHandle);
                     out.u32(count);
                     out.f32(interpStep);
                     out.u8(static_cast<u8>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       wire::write(out, sets[i]);
                     }
                   });
}

void GXSetArrayGPU(GXAttr attr, u64 gpuHandle, u32 byteOffset, u8 stride) {
  const GXAttr cpAttr = attr == GX_VA_NBT ? GX_VA_NRM : attr;
  if (cpAttr < GX_VA_POS || cpAttr > GX_VA_TEX7) {
    aurora::gx::gpu_channel::warn_bad_attribute(static_cast<u32>(attr));
    return;
  }
  const u64 handle = aurora::gx::gpu_channel::bind_handle(gpuHandle);
  write_fixed_hint<wire::LoadArrayGpuV1>(GX_AURORA_LOAD_ARRAY_GPU, AURORA_HINT_LOAD_ARRAY_GPU_VERSION,
                                         [&](wire::Writer& out) {
                                           out.u8(static_cast<u8>(cpAttr));
                                           out.u64(handle);
                                           out.u32(byteOffset);
                                         });
  GX_WRITE_CP_REG(CP_REG_ARRAYSTRIDE_ID | (cpAttr - GX_VA_POS), stride);
}

void GXAuroraSetSky(const AuroraSky* sky) {
  if (sky == nullptr) {
    hints::warn_emitter(GX_AURORA_RENDER_SKY, "null sky; nothing sent");
    return;
  }
  write_fixed_hint<wire::SkyV1>(GX_AURORA_RENDER_SKY, AURORA_HINT_SKY_VERSION,
                                [&](wire::Writer& out) { wire::write(out, *sky); });
}

void GXAuroraSetFogVolumes(const AuroraFogVolume* volumes, u16 count) {
  const u32 sent = checked_count(GX_AURORA_RENDER_FOG_VOLUMES, volumes, count, AURORA_MAX_FOG_VOLUMES);
  write_array_hint(GX_AURORA_RENDER_FOG_VOLUMES, AURORA_HINT_FOG_VOLUMES_VERSION,
                   wire::CountedHeader + sent * wire::FogVolume, [&](wire::Writer& out) {
                     out.u16(static_cast<u16>(sent));
                     for (u32 i = 0; i < sent; ++i) {
                       wire::write(out, volumes[i]);
                     }
                   });
}
