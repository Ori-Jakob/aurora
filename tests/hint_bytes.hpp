#pragma once

#include "gx/command_processor.hpp"
#include "gx/hints.hpp"

#include <aurora/render.hpp>
#include <dolphin/gx/GXAurora.h>

#include <cstdint>
#include <vector>

// Raw hint bytes for tests that feed the command processor directly, without the GX emitters
namespace aurora::test {
template <typename Fill>
std::vector<uint8_t> make_hint(uint16_t id, uint16_t version, uint32_t payloadSize, Fill&& fill) {
  std::vector<uint8_t> out(GX_AURORA_HINT_HEADER_SIZE + payloadSize);
  gx::hints::wire::Writer writer{out};
  writer.header(id, payloadSize);
  writer.u16(version);
  fill(writer);
  return out;
}

inline void process_hints(const std::vector<uint8_t>& bytes) {
  gx::fifo::process(bytes.data(), static_cast<uint32_t>(bytes.size()));
}

inline std::vector<uint8_t> stage_hint(render::Stage stage) {
  return make_hint(GX_AURORA_RENDER_STAGE, 1, gx::hints::wire::StageV1,
                   [&](gx::hints::wire::Writer& out) { out.u8(static_cast<uint8_t>(stage) + 1); });
}

inline std::vector<uint8_t> lights_hint(uint16_t count) {
  return make_hint(GX_AURORA_RENDER_LIGHTS, 2, gx::hints::wire::CountedHeader + count * gx::hints::wire::LightV2,
                   [&](gx::hints::wire::Writer& out) {
                     out.u16(count);
                     for (uint16_t i = 0; i < count; ++i) {
                       const AuroraLight light{
                           .id = i + 1u,
                           .type = AURORA_LIGHT_POINT,
                           .radius = 100.f,
                           .spotInnerCos = 1.f,
                           .spotOuterCos = -1.f,
                           .volumetricScale = 1.f,
                       };
                       gx::hints::wire::write(out, light, 2);
                     }
                   });
}

inline std::vector<uint8_t> environment_hint() {
  const AuroraEnvironment environment{.skyVisibility = 1.f, .timeOfDay = 12.f, .unitsPerMeter = 100.f};
  return make_hint(GX_AURORA_RENDER_ENVIRONMENT, 2, gx::hints::wire::EnvironmentV2,
                   [&](gx::hints::wire::Writer& out) { gx::hints::wire::write(out, environment, 2); });
}

inline std::vector<uint8_t> epoch_hint(uint32_t epoch) {
  return make_hint(GX_AURORA_RENDER_SCENE_EPOCH, 1, gx::hints::wire::SceneEpochV1,
                   [&](gx::hints::wire::Writer& out) { out.u32(epoch); });
}

// Perspective is C_MTXPerspective(60, 4:3, 1, 1000); orthographic is the 640x480 screen projection
inline std::vector<uint8_t> camera_hint(AuroraCameraRole role, bool perspective, uint32_t cutEpoch = 0) {
  float proj[4][4]{};
  if (perspective) {
    proj[0][0] = 1.2990381f;
    proj[1][1] = 1.7320508f;
    proj[2][2] = -1.f / 999.f;
    proj[2][3] = -1000.f / 999.f;
    proj[3][2] = -1.f;
  } else {
    proj[0][0] = 2.f / 640.f;
    proj[0][3] = -1.f;
    proj[1][1] = -2.f / 480.f;
    proj[1][3] = 1.f;
    proj[2][2] = -1.f;
    proj[2][3] = -1.f;
    proj[3][3] = 1.f;
  }
  return make_hint(GX_AURORA_RENDER_CAMERA, 1, gx::hints::wire::CameraV1, [&](gx::hints::wire::Writer& out) {
    out.u8(static_cast<uint8_t>(role));
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 4; ++col) {
        out.f32(row == col ? 1.f : 0.f);
      }
    }
    for (const auto& row : proj) {
      for (const float value : row) {
        out.f32(value);
      }
    }
    out.u8(static_cast<uint8_t>(perspective ? GX_PERSPECTIVE : GX_ORTHOGRAPHIC));
    out.u32(cutEpoch);
  });
}
} // namespace aurora::test
