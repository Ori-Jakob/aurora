#include "state_preamble.hpp"

#include "fifo_encoder.hpp"

#include "../gx/gpu_channel.hpp"
#include "../gx/hints.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace aurora::trace {
namespace {
constexpr u32 BpGenMode = 0x00;
constexpr u32 BpDrawDone = 0x45;
constexpr u32 BpEfbCopy = 0x52;
constexpr u32 BpTlutLoad0 = 0x64;
constexpr u32 BpTlutLoad1 = 0x65;
constexpr u32 BpTevRegFirst = 0xE0;
constexpr u32 BpTevRegLast = 0xE7;
constexpr u32 BpMask = 0xFE;
constexpr u32 CpMatrixIndexA = 0x30;
constexpr u32 CpMatrixIndexB = 0x40;
constexpr u32 XfMatrixIndexA = 0x18;
constexpr u32 XfViewport = 0x1A;
constexpr u32 XfProjection = 0x20;
constexpr u32 XfProjectionLast = 0x26;
// GX_AURORA_LOAD_ARRAYBASE addresses sixteen arrays: the twelve vertex attributes and the four indexed-load arrays
constexpr u32 ArrayBaseCount = 16;

u32 tev_s11(float value) noexcept {
  return static_cast<u32>(std::clamp<long>(std::lround(value * 255.f), -1024, 1023)) & 0x7FF;
}

u32 tev_u8(float value) noexcept { return static_cast<u32>(std::clamp<long>(std::lround(value * 255.f), 0, 255)); }

u32 float_bits(float value) noexcept { return std::bit_cast<u32>(value); }

template <typename T>
void emit_matrix(FifoEncoder& out, u16 addr, const T& matrix, size_t words) {
  std::array<u32, 12> bits{};
  const auto* flat = reinterpret_cast<const float*>(&matrix);
  for (size_t i = 0; i < words; ++i) {
    bits[i] = float_bits(flat[i]);
  }
  out.xf(addr, {bits.data(), words});
}

void emit_normal_matrix(FifoEncoder& out, u16 addr, const Mat3x4<float>& matrix) {
  std::array<u32, 9> bits{};
  const auto* flat = reinterpret_cast<const float*>(&matrix);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      bits[row * 3 + col] = float_bits(flat[row * 4 + col]);
    }
  }
  out.xf(addr, bits);
}

void emit_light(FifoEncoder& out, u16 addr, const gx::Light& light) {
  u32 color = 0;
  for (size_t i = 0; i < 4; ++i) {
    color = color << 8 | tev_u8(light.color[i]);
  }
  const std::array<u32, 16> words{
      0,
      0,
      0,
      color,
      float_bits(light.cosAtt[0]),
      float_bits(light.cosAtt[1]),
      float_bits(light.cosAtt[2]),
      float_bits(light.distAtt[0]),
      float_bits(light.distAtt[1]),
      float_bits(light.distAtt[2]),
      float_bits(light.pos[0]),
      float_bits(light.pos[1]),
      float_bits(light.pos[2]),
      float_bits(light.dir[0]),
      float_bits(light.dir[1]),
      float_bits(light.dir[2]),
  };
  out.xf(addr, words);
}

void emit_xf_reg(FifoEncoder& out, u32 reg, u32 value) { out.xf(static_cast<u16>(0x1000 + reg), {&value, 1}); }

void emit_registers(FifoEncoder& out, const gx::GXState& state) {
  for (u32 addr = 0; addr < state.cpRegCache.size(); ++addr) {
    if (addr != CpMatrixIndexA && addr != CpMatrixIndexB && state.cpRegValid.test(addr)) {
      out.cp(static_cast<u8>(addr), state.cpRegCache[addr]);
    }
  }

  // A decoded genmode always yields at least one stage, so stages prove it was written even after XF reset the bit
  if (state.bpRegValid.test(BpGenMode) || state.numTevStages != 0) {
    out.bp(state.bpRegCache[BpGenMode]);
  }
  for (u32 reg = BpGenMode + 1; reg < BpMask; ++reg) {
    if (reg == BpDrawDone || reg == BpEfbCopy || reg == BpTlutLoad0 || reg == BpTlutLoad1 ||
        (reg >= BpTevRegFirst && reg <= BpTevRegLast)) {
      continue;
    }
    if (state.bpRegValid.test(reg)) {
      out.bp(state.bpRegCache[reg]);
    }
  }

  // Color and konst registers share addresses, so the shadow only holds whichever was written last
  for (u32 i = 0; i < gx::MaxTevRegs; ++i) {
    const u32 ra = (BpTevRegFirst + i * 2) << 24;
    const u32 bg = (BpTevRegFirst + i * 2 + 1) << 24;
    const auto& color = state.colorRegs[i];
    out.bp(ra | tev_s11(color[0]) | tev_s11(color[3]) << 12);
    out.bp(bg | tev_s11(color[2]) | tev_s11(color[1]) << 12);
    const auto& konst = state.kcolors[i];
    out.bp(ra | 1u << 23 | tev_u8(konst[0]) | tev_u8(konst[3]) << 12);
    out.bp(bg | 1u << 23 | tev_u8(konst[2]) | tev_u8(konst[1]) << 12);
  }
  for (u32 reg = BpTevRegFirst; reg <= BpTevRegLast; ++reg) {
    if (state.bpRegValid.test(reg)) {
      out.bp(state.bpRegCache[reg]);
    }
  }

  for (u32 i = 0; i < gx::MaxTluts; ++i) {
    const auto& tlut = state.loadedTluts[i];
    if (tlut.numEntries != 0 && tlut.loadTlut0 != 0) {
      out.bp(BpTlutLoad0 << 24 | (tlut.loadTlut0 & 0xFFFFFF));
      out.bp(BpTlutLoad1 << 24 | i | ((tlut.numEntries - 1u) & 0x3FF) << 10);
    }
  }
  if (state.bpRegValid.test(BpTlutLoad0)) {
    out.bp(state.bpRegCache[BpTlutLoad0]);
  }

  for (u32 reg = 0; reg < gx::XfRegCount; ++reg) {
    if (reg == XfMatrixIndexA || (reg >= XfViewport && reg <= XfProjectionLast)) {
      continue;
    }
    if (state.xfRegValid.test(reg)) {
      emit_xf_reg(out, reg, state.xfRegCache[reg]);
    }
  }

  // CP 0x30 and XF 0x18 both set the current matrix and clear each other's valid bit; the valid one was written last
  const bool xfMatrixValid = state.xfRegValid.test(XfMatrixIndexA);
  const bool xfMatrixWritten = xfMatrixValid || state.xfRegCache[XfMatrixIndexA] != 0;
  const bool cpMatrixWritten = state.cpRegValid.test(CpMatrixIndexA) || state.cpRegCache[CpMatrixIndexA] != 0;
  if (xfMatrixValid) {
    if (cpMatrixWritten) {
      out.cp(CpMatrixIndexA, state.cpRegCache[CpMatrixIndexA]);
    }
    emit_xf_reg(out, XfMatrixIndexA, state.xfRegCache[XfMatrixIndexA]);
  } else {
    if (xfMatrixWritten) {
      emit_xf_reg(out, XfMatrixIndexA, state.xfRegCache[XfMatrixIndexA]);
    }
    if (cpMatrixWritten) {
      out.cp(CpMatrixIndexA, state.cpRegCache[CpMatrixIndexA]);
    }
  }
  if (state.cpRegValid.test(CpMatrixIndexB)) {
    out.cp(CpMatrixIndexB, state.cpRegCache[CpMatrixIndexB]);
  }
}

void emit_xf_memory(FifoEncoder& out, const gx::GXState& state) {
  for (u32 i = 0; i < gx::MaxPnMtx; ++i) {
    emit_matrix(out, static_cast<u16>(i * 12), state.pnMtx[i].pos, 12);
  }
  for (u32 i = 0; i < gx::MaxTexMtx; ++i) {
    emit_matrix(out, static_cast<u16>(0x78 + i * 12), state.texMtxs[i], 12);
  }
  for (u32 i = 0; i < gx::MaxPnMtx; ++i) {
    emit_normal_matrix(out, static_cast<u16>(0x400 + i * 9), state.pnMtx[i].nrm);
  }
  for (u32 i = 0; i < gx::MaxPTTexMtx; ++i) {
    emit_matrix(out, static_cast<u16>(0x500 + i * 12), state.ptTexMtxs[i], 12);
  }
  for (u32 i = 0; i < GX::MaxLights; ++i) {
    emit_light(out, static_cast<u16>(0x600 + i * 16), state.lights[i]);
  }

  if (std::any_of(state.xfRegCache.begin() + XfViewport, state.xfRegCache.begin() + XfViewport + 6,
                  [](u32 word) { return word != 0; })) {
    out.xf(static_cast<u16>(0x1000 + XfViewport), {state.xfRegCache.data() + XfViewport, 6});
  }

  bool projectionValid = true;
  for (u32 reg = XfProjection; reg <= XfProjectionLast; ++reg) {
    projectionValid &= state.xfRegValid.test(reg);
  }
  const gx::GXState defaults{};
  if (projectionValid) {
    out.xf(static_cast<u16>(0x1000 + XfProjection),
           {state.xfRegCache.data() + XfProjection, XfProjectionLast - XfProjection + 1});
  } else if (std::memcmp(&state.proj, &defaults.proj, sizeof(state.proj)) != 0) {
    out.aurora(GX_AURORA_LOAD_PROJECTION_FULL);
    for (u32 row = 0; row < 4; ++row) {
      for (u32 col = 0; col < 4; ++col) {
        out.f32(state.proj[row][col]);
      }
    }
  }
}

void emit_aurora_state(FifoEncoder& out, const gx::GXState& state) {
  for (u32 i = 0; i < ArrayBaseCount && GX_VA_POS + i < gx::MaxVtxAttr; ++i) {
    const auto& array = state.arrays[GX_VA_POS + i];
    if (array.gpu) {
      std::array<u8, GX_AURORA_HINT_HEADER_SIZE + gx::hints::wire::LoadArrayGpuV1> bytes{};
      gx::hints::wire::Writer hint{bytes};
      hint.header(GX_AURORA_LOAD_ARRAY_GPU, gx::hints::wire::LoadArrayGpuV1);
      hint.u16(AURORA_HINT_LOAD_ARRAY_GPU_VERSION);
      hint.u8(static_cast<u8>(GX_VA_POS + i));
      hint.u64(0);
      hint.u32(0);
      out.raw(bytes);
    } else if (array.data != nullptr || array.size != 0 || !array.le) {
      out.aurora(static_cast<u16>(GX_AURORA_LOAD_ARRAYBASE + i));
      out.u64(reinterpret_cast<uintptr_t>(array.data));
      out.u32(array.size);
      out.u8(array.le ? 1 : 0);
    }
  }

  const GXTexObj_ defaultTexture{};
  for (u32 i = 0; i < gx::MaxTextures; ++i) {
    const auto& obj = state.loadedTextures[i];
    if (obj.data == defaultTexture.data && obj.mWidth == defaultTexture.mWidth &&
        obj.mHeight == defaultTexture.mHeight && obj.mFormat == defaultTexture.mFormat &&
        obj.tlut == defaultTexture.tlut && obj.texObjId == defaultTexture.texObjId &&
        obj.texDataVersion == defaultTexture.texDataVersion && (obj.flags & 1) == (defaultTexture.flags & 1)) {
      continue;
    }
    out.aurora(GX_AURORA_LOAD_TEXOBJ);
    out.u8(static_cast<u8>(i));
    out.u64(reinterpret_cast<uintptr_t>(obj.data));
    out.u32(obj.mWidth);
    out.u32(obj.mHeight);
    out.u32(obj.mFormat);
    out.u32(static_cast<u32>(obj.tlut));
    out.u8(obj.flags & 1);
    out.u32(obj.texObjId);
    out.u32(obj.texDataVersion);
  }

  const GXTlutObj_ defaultTlut{};
  for (u32 i = 0; i < gx::MaxTluts; ++i) {
    const auto& tlut = state.loadedTluts[i];
    if (tlut.data == defaultTlut.data && tlut.format == defaultTlut.format &&
        tlut.numEntries == defaultTlut.numEntries && tlut.tlutObjId == defaultTlut.tlutObjId &&
        tlut.tlutDataVersion == defaultTlut.tlutDataVersion) {
      continue;
    }
    out.aurora(GX_AURORA_LOAD_TLUT);
    out.u8(static_cast<u8>(i));
    out.u64(reinterpret_cast<uintptr_t>(tlut.data));
    out.u32(static_cast<u32>(tlut.format));
    out.u16(tlut.numEntries);
    out.u32(tlut.tlutObjId);
    out.u32(tlut.tlutDataVersion);
  }

  if (state.frontOffset != 0.f || state.frontScale != 0.f || state.backOffset != 0.f || state.backScale != 0.f ||
      state.clamp != 0.f) {
    out.aurora(GX2_SET_POLYGON_OFFSET);
    out.f32(state.frontOffset);
    out.f32(state.frontScale);
    out.f32(state.backOffset);
    out.f32(state.backScale);
    out.f32(state.clamp);
  }

  if (state.texCopySrc != gfx::ClipRect{}) {
    out.aurora(GX_AURORA_LOAD_COPY_SRC);
    out.u32(static_cast<u32>(state.texCopySrc.x));
    out.u32(static_cast<u32>(state.texCopySrc.y));
    out.u32(static_cast<u32>(state.texCopySrc.width));
    out.u32(static_cast<u32>(state.texCopySrc.height));
  }
  if (state.texCopyDstWide) {
    out.aurora(GX_AURORA_LOAD_COPY_DST);
    out.u32(state.texCopyDstWidth);
    out.u32(state.texCopyDstHeight);
    out.u32(static_cast<u32>(state.texCopyFmt));
    out.u8(0);
  }
  if (state.texCopyDest != nullptr) {
    out.aurora(GX_AURORA_LOAD_COPY_DEST);
    out.u64(reinterpret_cast<uintptr_t>(state.texCopyDest));
  }

  // Applied last so the exact recorded render rectangles override whatever the register writes above mapped to
  const auto& viewport = state.renderViewport;
  out.aurora(GX_AURORA_LOAD_VIEWPORT_RENDER);
  out.f32(viewport.left);
  out.f32(viewport.top);
  out.f32(viewport.width);
  out.f32(viewport.height);
  out.f32(viewport.znear);
  out.f32(viewport.zfar);
  const auto& scissor = state.renderScissor;
  out.aurora(GX_AURORA_LOAD_SCISSOR_RENDER);
  out.u32(static_cast<u32>(scissor.x));
  out.u32(static_cast<u32>(scissor.y));
  out.u32(static_cast<u32>(scissor.width));
  out.u32(static_cast<u32>(scissor.height));

  // The mask only applies to the next BP write, so it must be the final command
  if (state.bpRegCache[BpMask] != 0x00FFFFFF) {
    out.bp(BpMask << 24 | state.bpRegCache[BpMask]);
  }
}
} // namespace

std::vector<uint8_t> build_state_preamble(const gx::GXState& state) {
  FifoEncoder out;
  emit_registers(out, state);
  emit_xf_memory(out, state);
  emit_aurora_state(out, state);
  gx::gpu_channel::write_preamble(out.bytes());
  return std::move(out.bytes());
}
} // namespace aurora::trace
