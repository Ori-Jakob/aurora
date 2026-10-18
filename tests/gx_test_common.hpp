#pragma once

#include <gtest/gtest.h>

#include <dolphin/gx.h>

#include "gx/gx.hpp"
#include "gx/fifo.hpp"
#include "gx/command_processor.hpp"
#include "gx/hints.hpp"
#include "gfx/depth_peek.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// Forward declare dirty state flush
extern "C" void __GXSetDirtyState();

namespace aurora::gfx {
extern uint32_t g_testDrawCount;
} // namespace aurora::gfx

// Records every decoded hint with the draw count at the time, so tests can check stream order
struct HintRecorder {
  struct Event {
    uint16_t id;
    uint32_t draws;
  };

  std::vector<Event> events;
  std::vector<aurora::gx::hints::Camera> cameras;
  std::vector<aurora::gx::hints::ObjectBegin> objects;
  std::vector<aurora::gx::hints::ScopePush> scopes;
  std::vector<aurora::render::Stage> stages;
  std::vector<std::vector<AuroraLight>> lights;
  std::vector<aurora::gx::hints::Environment> environments;
  std::vector<uint32_t> materials;
  std::vector<uint64_t> skeletonIds;
  std::vector<std::vector<aurora::Mat3x4<float>>> skeletons;
  std::vector<uint32_t> epochs;
  std::vector<AuroraTevPolicy> policies;
  std::vector<std::vector<AuroraInfluencer>> influencers;
  std::vector<std::vector<AuroraWindSource>> windSources;
  uint32_t resets = 0;

  [[nodiscard]] size_t count(uint16_t id) const {
    return static_cast<size_t>(std::ranges::count(events, id, &Event::id));
  }

  aurora::gx::hints::Sink sink() {
    using namespace aurora::gx::hints;
    return {
        .userData = this,
        .reset = [](void* self) noexcept { ++get(self).resets; },
        .camera =
            [](void* self, const Camera& camera) noexcept {
              get(self).note(GX_AURORA_RENDER_CAMERA).cameras.push_back(camera);
            },
        .objectBegin =
            [](void* self, const ObjectBegin& object) noexcept {
              get(self).note(GX_AURORA_RENDER_OBJECT_BEGIN).objects.push_back(object);
            },
        .objectEnd = [](void* self) noexcept { get(self).note(GX_AURORA_RENDER_OBJECT_END); },
        .scopePush =
            [](void* self, const ScopePush& scope) noexcept {
              get(self).note(GX_AURORA_RENDER_SCOPE_PUSH).scopes.push_back(scope);
            },
        .scopePop = [](void* self) noexcept { get(self).note(GX_AURORA_RENDER_SCOPE_POP); },
        .stage =
            [](void* self, aurora::render::Stage stage) noexcept {
              get(self).note(GX_AURORA_RENDER_STAGE).stages.push_back(stage);
            },
        .lights =
            [](void* self, std::span<const AuroraLight> lights) noexcept {
              get(self).note(GX_AURORA_RENDER_LIGHTS).lights.emplace_back(lights.begin(), lights.end());
            },
        .environment =
            [](void* self, const Environment& environment) noexcept {
              get(self).note(GX_AURORA_RENDER_ENVIRONMENT).environments.push_back(environment);
            },
        .material =
            [](void* self, uint32_t materialClass) noexcept {
              get(self).note(GX_AURORA_RENDER_MATERIAL).materials.push_back(materialClass);
            },
        .skeleton =
            [](void* self, const Skeleton& skeleton) noexcept {
              auto& recorder = get(self).note(GX_AURORA_RENDER_SKELETON);
              recorder.skeletonIds.push_back(skeleton.objectId);
              recorder.skeletons.emplace_back(skeleton.joints.begin(), skeleton.joints.end());
            },
        .sceneEpoch =
            [](void* self, uint32_t epoch) noexcept {
              get(self).note(GX_AURORA_RENDER_SCENE_EPOCH).epochs.push_back(epoch);
            },
        .tevPolicy =
            [](void* self, const AuroraTevPolicy& policy) noexcept {
              get(self).note(GX_AURORA_RENDER_TEV_POLICY).policies.push_back(policy);
            },
        .influencers =
            [](void* self, std::span<const AuroraInfluencer> influencers) noexcept {
              get(self)
                  .note(GX_AURORA_RENDER_INFLUENCERS)
                  .influencers.emplace_back(influencers.begin(), influencers.end());
            },
        .windSources =
            [](void* self, std::span<const AuroraWindSource> sources) noexcept {
              get(self).note(GX_AURORA_RENDER_WIND_SOURCES).windSources.emplace_back(sources.begin(), sources.end());
            },
    };
  }

private:
  static HintRecorder& get(void* self) { return *static_cast<HintRecorder*>(self); }

  HintRecorder& note(uint16_t id) {
    events.push_back({id, aurora::gfx::g_testDrawCount});
    return *this;
  }
};

class GXFifoTest : public ::testing::Test {
protected:
  void SetUp() override {
    GXInit(nullptr, 0);
    aurora::gx::fifo::clear_buffer();
    aurora::gx::g_gxState = aurora::gx::GXState{};
    aurora::gfx::depth_peek::testing::reset();
  }

  void TearDown() override { aurora::gx::fifo::shutdown(); }

  // Copy the internal FIFO buffer contents and clear it
  std::vector<u8> capture_fifo() {
    auto size = aurora::gx::fifo::get_buffer_size();
    auto* data = aurora::gx::fifo::get_buffer_data();
    std::vector<u8> bytes(data, data + size);
    aurora::gx::fifo::clear_buffer();
    return bytes;
  }

  // Reset g_gxState to default-constructed state
  void reset_gx_state() { aurora::gx::g_gxState = aurora::gx::GXState{}; }

  // Decode a captured FIFO byte stream through the command processor
  void decode_fifo(const std::vector<u8>& bytes) {
    aurora::gx::fifo::process(bytes.data(), static_cast<u32>(bytes.size()));
  }

  // Flush dirty state, then capture the FIFO buffer
  std::vector<u8> flush_and_capture() {
    __GXSetDirtyState();
    return capture_fifo();
  }

  // Convenience reference to g_gxState
  aurora::gx::GXState& gxState() { return aurora::gx::g_gxState; }
};

template <typename T>
inline bool same_bits(const T& lhs, const T& rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

#define EXPECT_SAME_BITS(field) EXPECT_TRUE(same_bits(want.field, got.field)) << #field

inline void expect_state_eq(const aurora::gx::GXState& want, const aurora::gx::GXState& got) {
  EXPECT_SAME_BITS(pnMtx);
  EXPECT_EQ(want.currentPnMtx, got.currentPnMtx);
  EXPECT_SAME_BITS(proj);
  EXPECT_EQ(want.fog.type, got.fog.type);
  EXPECT_SAME_BITS(fog.a);
  EXPECT_SAME_BITS(fog.b);
  EXPECT_SAME_BITS(fog.c);
  EXPECT_SAME_BITS(fog.color);
  EXPECT_EQ(want.fog.rangeK, got.fog.rangeK);
  EXPECT_EQ(want.fog.rangeCenter, got.fog.rangeCenter);
  EXPECT_EQ(want.fog.rangeEnabled, got.fog.rangeEnabled);
  EXPECT_EQ(want.fog.fog0Raw, got.fog.fog0Raw);
  EXPECT_EQ(want.fog.fog1Raw, got.fog.fog1Raw);
  EXPECT_EQ(want.fog.fog2Raw, got.fog.fog2Raw);
  EXPECT_EQ(want.cullMode, got.cullMode);
  EXPECT_EQ(want.lineWidth, got.lineWidth);
  EXPECT_EQ(want.pointSize, got.pointSize);
  EXPECT_EQ(want.lineTexOffset, got.lineTexOffset);
  EXPECT_EQ(want.pointTexOffset, got.pointTexOffset);
  EXPECT_EQ(want.lineHalfAspect, got.lineHalfAspect);
  EXPECT_EQ(want.blendMode, got.blendMode);
  EXPECT_EQ(want.blendFacSrc, got.blendFacSrc);
  EXPECT_EQ(want.blendFacDst, got.blendFacDst);
  EXPECT_EQ(want.blendOp, got.blendOp);
  EXPECT_EQ(want.depthFunc, got.depthFunc);
  EXPECT_SAME_BITS(clearColor);
  EXPECT_EQ(want.clearDepth, got.clearDepth);
  EXPECT_EQ(want.pixelFmt, got.pixelFmt);
  EXPECT_EQ(want.zFmt, got.zFmt);
  EXPECT_EQ(want.zCompLocBeforeTex, got.zCompLocBeforeTex);
  EXPECT_EQ(want.dstAlpha, got.dstAlpha);
  EXPECT_SAME_BITS(alphaCompare);
  EXPECT_SAME_BITS(colorRegs);
  EXPECT_SAME_BITS(kcolors);
  EXPECT_SAME_BITS(colorChannelConfig);
  for (size_t i = 0; i < want.colorChannelState.size(); ++i) {
    EXPECT_SAME_BITS(colorChannelState[i].matColor) << i;
    EXPECT_SAME_BITS(colorChannelState[i].ambColor) << i;
    EXPECT_EQ(want.colorChannelState[i].lightMask, got.colorChannelState[i].lightMask) << i;
  }
  EXPECT_SAME_BITS(lights);
  EXPECT_SAME_BITS(tevStages);
  for (size_t i = 0; i < want.loadedTextures.size(); ++i) {
    const auto& w = want.loadedTextures[i];
    const auto& g = got.loadedTextures[i];
    EXPECT_EQ(w.mode0, g.mode0) << "texture " << i;
    EXPECT_EQ(w.mode1, g.mode1) << "texture " << i;
    EXPECT_EQ(w.image0, g.image0) << "texture " << i;
    EXPECT_EQ(w.image3, g.image3) << "texture " << i;
    EXPECT_EQ(w.data, g.data) << "texture " << i;
    EXPECT_EQ(w.mWidth, g.mWidth) << "texture " << i;
    EXPECT_EQ(w.mHeight, g.mHeight) << "texture " << i;
    EXPECT_EQ(w.mFormat, g.mFormat) << "texture " << i;
    EXPECT_EQ(w.tlut, g.tlut) << "texture " << i;
    EXPECT_EQ(w.texObjId, g.texObjId) << "texture " << i;
    EXPECT_EQ(w.texDataVersion, g.texDataVersion) << "texture " << i;
    EXPECT_EQ(w.flags & 1, g.flags & 1) << "texture " << i;
  }
  for (size_t i = 0; i < want.loadedTluts.size(); ++i) {
    const auto& w = want.loadedTluts[i];
    const auto& g = got.loadedTluts[i];
    EXPECT_EQ(w.loadTlut0, g.loadTlut0) << "tlut " << i;
    EXPECT_EQ(w.numEntries, g.numEntries) << "tlut " << i;
    EXPECT_EQ(w.data, g.data) << "tlut " << i;
    EXPECT_EQ(w.format, g.format) << "tlut " << i;
    EXPECT_EQ(w.tlutObjId, g.tlutObjId) << "tlut " << i;
    EXPECT_EQ(w.tlutDataVersion, g.tlutDataVersion) << "tlut " << i;
  }
  EXPECT_SAME_BITS(texMtxs);
  EXPECT_SAME_BITS(ptTexMtxs);
  EXPECT_SAME_BITS(tcgs);
  EXPECT_EQ(want.texCoordScales, got.texCoordScales);
  EXPECT_EQ(want.vtxDesc, got.vtxDesc);
  EXPECT_SAME_BITS(vtxFmts);
  EXPECT_SAME_BITS(tevSwapTable);
  EXPECT_SAME_BITS(indStages);
  for (size_t i = 0; i < want.indTexMtxs.size(); ++i) {
    EXPECT_SAME_BITS(indTexMtxs[i].mtx) << i;
    EXPECT_EQ(want.indTexMtxs[i].scaleExp, got.indTexMtxs[i].scaleExp) << i;
    EXPECT_EQ(want.indTexMtxs[i].adjScaleRaw, got.indTexMtxs[i].adjScaleRaw) << i;
  }
  for (size_t i = 0; i < want.arrays.size(); ++i) {
    EXPECT_EQ(want.arrays[i], got.arrays[i]) << "array " << i;
  }
  EXPECT_EQ(want.depthCompare, got.depthCompare);
  EXPECT_EQ(want.depthUpdate, got.depthUpdate);
  EXPECT_EQ(want.colorUpdate, got.colorUpdate);
  EXPECT_EQ(want.alphaUpdate, got.alphaUpdate);
  EXPECT_EQ(want.numChans, got.numChans);
  EXPECT_EQ(want.numIndStages, got.numIndStages);
  EXPECT_EQ(want.numTevStages, got.numTevStages);
  EXPECT_EQ(want.numTexGens, got.numTexGens);
  EXPECT_SAME_BITS(frontOffset);
  EXPECT_SAME_BITS(frontScale);
  EXPECT_SAME_BITS(backOffset);
  EXPECT_SAME_BITS(backScale);
  EXPECT_SAME_BITS(clamp);
  EXPECT_SAME_BITS(logicalViewport);
  EXPECT_SAME_BITS(renderViewport);
  EXPECT_EQ(want.logicalScissor, got.logicalScissor);
  EXPECT_EQ(want.renderScissor, got.renderScissor);
  EXPECT_EQ(want.texCopySrc, got.texCopySrc);
  EXPECT_EQ(want.texCopyFmt, got.texCopyFmt);
  EXPECT_EQ(want.texCopyDstWidth, got.texCopyDstWidth);
  EXPECT_EQ(want.texCopyDstHeight, got.texCopyDstHeight);
  EXPECT_EQ(want.texCopyDstWide, got.texCopyDstWide);
  EXPECT_EQ(want.texCopyDest, got.texCopyDest);

  // The shadows the preamble cannot rebuild are trigger registers and color/konst writes that were overwritten
  for (u32 reg = 0; reg < 0x100; ++reg) {
    if (reg == 0x45 || reg == 0x52 || reg == 0x65 || (reg >= 0xE0 && reg <= 0xE7 && !want.bpRegValid.test(reg))) {
      continue;
    }
    EXPECT_EQ(want.bpRegCache[reg], got.bpRegCache[reg]) << "BP 0x" << std::hex << reg;
  }
  EXPECT_EQ(want.cpRegCache, got.cpRegCache);
  for (u32 reg = 0; reg < aurora::gx::XfRegCount; ++reg) {
    if (want.xfRegValid.test(reg) || (reg >= 0x1A && reg <= 0x1F)) {
      EXPECT_EQ(want.xfRegCache[reg], got.xfRegCache[reg]) << "XF 0x" << std::hex << reg;
    }
  }
}

#undef EXPECT_SAME_BITS
