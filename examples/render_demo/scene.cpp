#include "scene.hpp"

#include "hints.hpp"
#include "procedural.hpp"

#include <aurora/gpu_systems.hpp>
#include <dolphin/gx.h>
#include <dolphin/mtx.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace render_demo {
namespace {
constexpr f32 Pi = 3.14159265358979f;
constexpr GXColor ClearColor{112, 160, 208, 255};
constexpr GXColor AmbientColor{40, 44, 56, 255};
constexpr GXColor White{255, 255, 255, 255};
constexpr GXColor HudColor{20, 24, 32, 160};
constexpr FogParams Fog{GX_FOG_PERSP_LIN, 40.f, 160.f, 0.5f, 400.f, ClearColor};
constexpr GXColor BandColor{200, 120, 60, 255};
// The default fog stops near f = 0.23 here; these bands cover f up to the clamp and put lock 1 on real pipelines
constexpr std::array<FogParams, 2> FogBands{{
    {GX_FOG_PERSP_LIN, 10.f, 60.f, 0.5f, 400.f, ClearColor},
    {GX_FOG_PERSP_EXP, 10.f, 60.f, 0.5f, 400.f, BandColor},
}};
constexpr f32 PointRefDistance = 10.f;
constexpr f32 PointRefBrightness = 0.5f;
constexpr f32 SpotCutoff = 25.f;
constexpr f32 SpotRefDistance = 15.f;
constexpr f32 SpotRefBrightness = 0.5f;
constexpr Vec WindDirection{1.f, 0.f, 0.f};
constexpr f32 WindPower = 0.3f;
constexpr u32 TentacleBones = 4;
constexpr f32 MarkerScale = 0.35f;
constexpr size_t MotionProbeBox = 2;
constexpr std::array<f32, 2> MotionProbeGroundNdc{0.3f, -0.7f};
constexpr Vec WaterProbe{-18.f, 0.05f, 14.f};

constexpr u64 GroundObject = 1;
constexpr u64 FirstBoxObject = 0x100;
constexpr u64 TentacleObject = 0x200;
constexpr u64 FirstMarkerObject = 0x210;
constexpr u64 CaptureOnlyObject = 0x300;
constexpr u64 WaterObject = 0x400;
constexpr u32 GroundMaterial = 1;
constexpr u32 SceneEpoch = 1;
constexpr u32 GroundTextureSize = 64;
constexpr u32 GroundMipCount = 7;
constexpr u32 GroundCells = 16;
constexpr f32 GroundCellSize = 5.f;
constexpr f32 GroundExtent = GroundCells * GroundCellSize / 2.f;

struct BoxPlacement {
  f32 x;
  f32 z;
  f32 scale;
};
constexpr std::array<BoxPlacement, 6> BoxPlacements{{
    {-8.f, -6.f, 2.f},
    {-3.f, 7.f, 1.5f},
    {5.f, 5.f, 2.5f},
    {9.f, -8.f, 1.2f},
    {0.f, -10.f, 3.f},
    {-11.f, 3.f, 1.8f},
}};

alignas(32) constexpr f32 BoxPositions[8][3] = {
    {-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {-.5f, .5f, -.5f}, {.5f, .5f, -.5f},
    {-.5f, -.5f, .5f},  {.5f, -.5f, .5f},  {-.5f, .5f, .5f},  {.5f, .5f, .5f},
};
alignas(32) constexpr f32 BoxNormals[6][3] = {
    {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, -1.f},
};
alignas(32) constexpr u8 BoxColors[6][4] = {
    {220, 80, 70, 255}, {70, 180, 90, 255},  {240, 230, 200, 255},
    {60, 60, 60, 255},  {80, 120, 220, 255}, {230, 180, 60, 255},
};
constexpr u8 BoxFaces[6][4] = {
    {7, 3, 1, 5}, {2, 6, 4, 0}, {2, 3, 7, 6}, {4, 5, 1, 0}, {6, 7, 5, 4}, {3, 2, 0, 1},
};

struct Lights {
  Vec positions[4];
  GXColor colors[4];
  Vec sunDir;
  Vec spotDir;
};

constexpr u32 BoxNormalsOffset = sizeof(BoxPositions);
constexpr u32 BoxColorsOffset = BoxNormalsOffset + sizeof(BoxNormals);
constexpr u32 BoxArrayBytes = BoxColorsOffset + sizeof(BoxColors);

alignas(32) u8 sBoxDisplayList[256];
u32 sBoxDisplayListSize = 0;
aurora::render::gpu::Handle sBoxArrays = aurora::render::gpu::InvalidHandle;
aurora::render::gpu::Handle sBoxPositions = aurora::render::gpu::InvalidHandle;
std::vector<u8> sGroundTexture;
GXTexObj sGroundTexObj;
std::vector<std::vector<SkinnedVertex>> sTentacle;

// aurora has no GXMatrixIndex1x8; the FIFO byte is identical
void emit_matrix_index(u8 index) { GXParam1u8(index); }

void load_matrices(const Mtx modelView, u32 id) {
  Mtx normal;
  C_MTXInvXpose(modelView, normal);
  GXLoadPosMtxImm(modelView, id);
  GXLoadNrmMtxImm(normal, id);
}

void set_fog(const FogParams& fog) { GXSetFog(fog.type, fog.start, fog.end, fog.nearZ, fog.farZ, fog.color); }

void set_box_format() {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX16);
  GXSetVtxDesc(GX_VA_NRM, GX_INDEX16);
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  if (sBoxArrays != aurora::render::gpu::InvalidHandle) {
    GXSetArrayGPU(GX_VA_POS, sBoxPositions, 0, sizeof(BoxPositions[0]));
    GXSetArrayGPU(GX_VA_NRM, sBoxArrays, BoxNormalsOffset, sizeof(BoxNormals[0]));
    GXSetArrayGPU(GX_VA_CLR0, sBoxArrays, BoxColorsOffset, sizeof(BoxColors[0]));
    return;
  }
  GXSetArray(GX_VA_POS, BoxPositions, sizeof(BoxPositions), sizeof(BoxPositions[0]), true);
  GXSetArray(GX_VA_NRM, BoxNormals, sizeof(BoxNormals), sizeof(BoxNormals[0]), true);
  GXSetArray(GX_VA_CLR0, BoxColors, sizeof(BoxColors), sizeof(BoxColors[0]), true);
}

void copy_box_positions(aurora::render::gpu::SystemId copy) {
  namespace gpu = aurora::render::gpu;
  if (copy == gpu::InvalidSystem || sBoxArrays == gpu::InvalidHandle ||
      gpu::system_status(copy) != gpu::SystemStatus::Ready) {
    return;
  }
  const gpu::Handle copied = gpu::reserve(sizeof(BoxPositions));
  const std::array<gpu::Handle, 1> inputs{sBoxArrays};
  if (copied != gpu::InvalidHandle && gpu::dispatch(copy, 0, inputs, copied, {sizeof(BoxPositions) / 4, 1, 1})) {
    sBoxPositions = copied;
  }
}

void upload_gpu_arrays(const SceneOptions& options) {
  sBoxArrays = aurora::render::gpu::InvalidHandle;
  sBoxPositions = aurora::render::gpu::InvalidHandle;
  if (options.gpuArrays) {
    std::array<u8, BoxArrayBytes> bytes{};
    std::memcpy(bytes.data(), BoxPositions, sizeof(BoxPositions));
    std::memcpy(bytes.data() + BoxNormalsOffset, BoxNormals, sizeof(BoxNormals));
    std::memcpy(bytes.data() + BoxColorsOffset, BoxColors, sizeof(BoxColors));
    sBoxArrays = aurora::render::gpu::upload(bytes.data(), BoxArrayBytes);
    sBoxPositions = sBoxArrays;
    copy_box_positions(options.gpuCopySystem);
  }
  static const std::vector<u8> zeroes(aurora::render::gpu::MaxUploadBytes);
  for (u64 left = u64{options.gpuPadMiB} << 20; left != 0;) {
    const auto piece = static_cast<u32>(std::min<u64>(left, aurora::render::gpu::MaxUploadBytes));
    aurora::render::gpu::upload(zeroes.data(), piece);
    left -= piece;
  }
}

void set_lit_channel(GXColorSrc matSrc) {
  GXSetNumChans(1);
  GXSetChanCtrl(GX_COLOR0, GX_ENABLE, GX_SRC_REG, matSrc, GX_LIGHT0 | GX_LIGHT1 | GX_LIGHT2 | GX_LIGHT3, GX_DF_CLAMP,
                GX_AF_SPOT);
  GXSetChanCtrl(GX_ALPHA0, GX_DISABLE, GX_SRC_REG, matSrc, GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
  GXSetChanAmbColor(GX_COLOR0A0, AmbientColor);
  GXSetChanMatColor(GX_COLOR0A0, White);
}

void set_pass_color_tev() {
  GXSetNumTexGens(0);
  GXSetNumTevStages(1);
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD_NULL, GX_TEXMAP_NULL, GX_COLOR0A0);
  GXSetTevOp(GX_TEVSTAGE0, GX_PASSCLR);
}

void record_box_display_list() {
  set_box_format();
  GXBeginDisplayList(sBoxDisplayList, sizeof(sBoxDisplayList));
  GXBegin(GX_QUADS, GX_VTXFMT1, 24);
  for (u8 face = 0; face < 6; ++face) {
    for (const u8 corner : BoxFaces[face]) {
      GXPosition1x16(corner);
      GXNormal1x16(face);
      GXColor1x8(face);
    }
  }
  GXEnd();
  sBoxDisplayListSize = GXEndDisplayList();
}

Vec to_view(const Mtx view, const Vec& world) {
  Vec out;
  C_MTXMultVec(view, &world, &out);
  return out;
}

Lights setup_lights(const Mtx view, f32 t) {
  Lights lights{};
  // Only the spot light sets a direction; the others must not load stack garbage into XF
  GXLightObj light{};

  Vec& sunDir = lights.sunDir;
  sunDir = Vec{-0.4f, -1.f, -0.3f};
  C_VECNormalize(&sunDir, &sunDir);
  lights.positions[0] = Vec{-sunDir.x * 1e5f, -sunDir.y * 1e5f, -sunDir.z * 1e5f};
  // Dim enough that the point and spot lights still read on sunlit ground
  lights.colors[0] = GXColor{150, 140, 120, 255};
  const Vec sunView = to_view(view, lights.positions[0]);
  GXInitLightPos(&light, sunView.x, sunView.y, sunView.z);
  GXInitLightColor(&light, lights.colors[0]);
  GXInitLightAttn(&light, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f);
  GXLoadLightObjImm(&light, GX_LIGHT0);

  constexpr GXColor PointColors[2] = {{255, 120, 60, 255}, {80, 140, 255, 255}};
  constexpr GXLightID PointIds[2] = {GX_LIGHT1, GX_LIGHT2};
  for (int i = 0; i < 2; ++i) {
    const f32 angle = t + Pi * static_cast<f32>(i);
    lights.positions[1 + i] = Vec{9.f * std::cos(angle), 2.5f, 9.f * std::sin(angle)};
    lights.colors[1 + i] = PointColors[i];
    const Vec pos = to_view(view, lights.positions[1 + i]);
    GXInitLightPos(&light, pos.x, pos.y, pos.z);
    GXInitLightColor(&light, PointColors[i]);
    GXInitLightSpot(&light, 0.f, GX_SP_OFF);
    GXInitLightDistAttn(&light, PointRefDistance, PointRefBrightness, GX_DA_MEDIUM);
    GXLoadLightObjImm(&light, PointIds[i]);
  }

  lights.positions[3] = Vec{0.f, 10.f, 0.f};
  lights.colors[3] = GXColor{255, 250, 220, 255};
  Vec& spotDir = lights.spotDir;
  spotDir = Vec{0.3f * std::sin(0.5f * t), -1.f, 0.3f * std::cos(0.5f * t)};
  C_VECNormalize(&spotDir, &spotDir);
  Vec spotDirView;
  C_MTXMultVecSR(view, &spotDir, &spotDirView);
  const Vec spotPos = to_view(view, lights.positions[3]);
  GXInitLightPos(&light, spotPos.x, spotPos.y, spotPos.z);
  GXInitLightDir(&light, spotDirView.x, spotDirView.y, spotDirView.z);
  GXInitLightColor(&light, lights.colors[3]);
  GXInitLightSpot(&light, SpotCutoff, GX_SP_COS2);
  GXInitLightDistAttn(&light, SpotRefDistance, SpotRefBrightness, GX_DA_GENTLE);
  GXLoadLightObjImm(&light, GX_LIGHT3);
  return lights;
}

void draw_ground(const Mtx view) {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxDesc(GX_VA_NRM, GX_DIRECT);
  GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
  GXSetNumTexGens(1);
  GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
  GXLoadTexObj(&sGroundTexObj, GX_TEXMAP0);
  GXSetNumTevStages(1);
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
  GXSetTevOp(GX_TEVSTAGE0, GX_MODULATE);
  set_lit_channel(GX_SRC_REG);
  load_matrices(view, GX_PNMTX0);
  GXSetCurrentMtx(GX_PNMTX0);
  GXSetCullMode(GX_CULL_BACK);

  GXBegin(GX_QUADS, GX_VTXFMT0, GroundCells * GroundCells * 4);
  for (u32 cz = 0; cz < GroundCells; ++cz) {
    for (u32 cx = 0; cx < GroundCells; ++cx) {
      const f32 x0 = -GroundExtent + static_cast<f32>(cx) * GroundCellSize;
      const f32 z0 = -GroundExtent + static_cast<f32>(cz) * GroundCellSize;
      const f32 x1 = x0 + GroundCellSize;
      const f32 z1 = z0 + GroundCellSize;
      const f32 corners[4][2] = {{x0, z0}, {x1, z0}, {x1, z1}, {x0, z1}};
      for (const auto& [x, z] : corners) {
        GXPosition3f32(x, 0.f, z);
        GXNormal3f32(0.f, 1.f, 0.f);
        GXTexCoord2f32(x / GroundCellSize, z / GroundCellSize);
      }
    }
  }
  GXEnd();
}

void box_model(size_t i, f32 t, Mtx model) {
  const auto& box = BoxPlacements[i];
  Mtx translate;
  Mtx rotate;
  Mtx scale;
  C_MTXTrans(translate, box.x, box.scale / 2.f, box.z);
  C_MTXRotRad(rotate, 'y', t * (0.3f + 0.15f * static_cast<f32>(i)));
  C_MTXScale(scale, box.scale, box.scale, box.scale);
  C_MTXConcat(translate, rotate, model);
  C_MTXConcat(model, scale, model);
}

void draw_boxes(const Mtx view, f32 t) {
  set_lit_channel(GX_SRC_VTX);
  set_pass_color_tev();
  GXSetCurrentMtx(GX_PNMTX0);
  GXSetCullMode(GX_CULL_BACK);
  for (size_t i = 0; i < BoxPlacements.size(); ++i) {
    Mtx model;
    Mtx modelView;
    box_model(i, t, model);
    C_MTXConcat(view, model, modelView);
    begin_object(FirstBoxObject + i, AURORA_CAT_DYNAMIC, 0);
    load_matrices(modelView, GX_PNMTX0);
    set_box_format();
    GXCallDisplayList(sBoxDisplayList, sBoxDisplayListSize);
    end_object();
  }
}

void bone_joint(Mtx out, f32 t, int k) {
  Mtx rz;
  Mtx rx;
  C_MTXRotRad(rz, 'z', 0.35f * std::sin(1.7f * t + 0.8f * static_cast<f32>(k)));
  C_MTXRotRad(rx, 'x', 0.2f * std::sin(1.1f * t + 0.5f * static_cast<f32>(k)));
  C_MTXConcat(rz, rx, out);
}

void tentacle_bones(f32 t, Mtx (&bones)[TentacleBones]) {
  Mtx bone;
  Mtx joint;
  C_MTXTrans(bone, 6.f, 0.f, -2.f);
  for (u32 k = 0; k < TentacleBones; ++k) {
    if (k != 0) {
      Mtx offset;
      C_MTXTrans(offset, 0.f, 1.5f, 0.f);
      C_MTXConcat(bone, offset, bone);
    }
    bone_joint(joint, t, static_cast<int>(k));
    C_MTXConcat(bone, joint, bone);
    C_MTXCopy(bone, bones[k]);
  }
}

Vec tentacle_tip(f32 t) {
  // The last ring sits 1.5 units up the last bone
  Mtx bones[TentacleBones];
  tentacle_bones(t, bones);
  const Vec local{0.f, 1.5f, 0.f};
  Vec tip;
  C_MTXMultVec(bones[TentacleBones - 1], &local, &tip);
  return tip;
}

void draw_tentacle(const Mtx view, const Mtx (&bones)[TentacleBones]) {
  for (u32 k = 0; k < TentacleBones; ++k) {
    Mtx modelView;
    C_MTXConcat(view, bones[k], modelView);
    load_matrices(modelView, GX_PNMTX1 + 3 * k);
  }

  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_PNMTXIDX, GX_DIRECT);
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxDesc(GX_VA_NRM, GX_DIRECT);
  GXSetVtxDesc(GX_VA_CLR0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT2, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT2, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT2, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  set_lit_channel(GX_SRC_VTX);
  set_pass_color_tev();
  GXSetCullMode(GX_CULL_NONE);
  for (const auto& strip : sTentacle) {
    GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT2, static_cast<u16>(strip.size()));
    for (const auto& v : strip) {
      emit_matrix_index(v.matrixIndex);
      GXPosition3f32(v.position[0], v.position[1], v.position[2]);
      GXNormal3f32(v.normal[0], v.normal[1], v.normal[2]);
      GXColor4u8(v.color[0], v.color[1], v.color[2], v.color[3]);
    }
    GXEnd();
  }
}

void draw_light_markers(const Mtx view, const Lights& lights) {
  GXSetNumChans(1);
  GXSetChanCtrl(GX_COLOR0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
  GXSetChanCtrl(GX_ALPHA0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
  set_pass_color_tev();
  GXSetCurrentMtx(GX_PNMTX0);
  GXSetCullMode(GX_CULL_BACK);
  push_scope(AURORA_CAT_EMISSIVE | AURORA_CAT_NO_SHADOW_CAST, 0, AURORA_CAMERA_INHERIT);
  for (int i = 1; i < 4; ++i) {
    Mtx model;
    Mtx scale;
    Mtx modelView;
    C_MTXTrans(model, lights.positions[i].x, lights.positions[i].y, lights.positions[i].z);
    C_MTXScale(scale, MarkerScale, MarkerScale, MarkerScale);
    C_MTXConcat(model, scale, model);
    C_MTXConcat(view, model, modelView);
    begin_object(FirstMarkerObject + static_cast<u64>(i - 1), AURORA_CAT_DYNAMIC, 0);
    load_matrices(modelView, GX_PNMTX0);
    GXSetChanMatColor(GX_COLOR0A0, lights.colors[i]);
    set_box_format();
    GXCallDisplayList(sBoxDisplayList, sBoxDisplayListSize);
    end_object();
  }
  pop_scope();
}

// Reuses the marker state; ten units behind the eye it is clipped even where a skipping replayer draws it
void draw_capture_only_box(const Mtx view, const Point3d& eye, const Point3d& target) {
  Vec back{eye.x - target.x, eye.y - target.y, eye.z - target.z};
  C_VECNormalize(&back, &back);
  Mtx model;
  Mtx scale;
  Mtx modelView;
  C_MTXTrans(model, eye.x + 10.f * back.x, eye.y + 10.f * back.y, eye.z + 10.f * back.z);
  C_MTXScale(scale, MarkerScale, MarkerScale, MarkerScale);
  C_MTXConcat(model, scale, model);
  C_MTXConcat(view, model, modelView);
  begin_object(CaptureOnlyObject, AURORA_CAT_DYNAMIC, AURORA_OBJ_CAPTURE_ONLY);
  load_matrices(modelView, GX_PNMTX0);
  set_box_format();
  GXCallDisplayList(sBoxDisplayList, sBoxDisplayListSize);
  end_object();
  count_capture_only_draw();
}

void draw_water(const Mtx view) {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT3, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetNumChans(0);
  GXSetNumTexGens(0);
  GXSetNumTevStages(1);
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD_NULL, GX_TEXMAP_NULL, GX_COLOR_NULL);
  GXSetTevKColor(GX_KCOLOR0, GXColor{60, 110, 170, 128});
  GXSetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
  GXSetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);
  GXSetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST);
  GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
  GXSetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
  GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
  GXSetBlendMode(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR);
  GXSetZMode(GX_TRUE, GX_LEQUAL, GX_FALSE);
  GXSetCullMode(GX_CULL_NONE);
  load_matrices(view, GX_PNMTX0);
  GXSetCurrentMtx(GX_PNMTX0);

  GXBegin(GX_QUADS, GX_VTXFMT3, 4);
  GXPosition3f32(-24.f, 0.05f, 8.f);
  GXPosition3f32(-12.f, 0.05f, 8.f);
  GXPosition3f32(-12.f, 0.05f, 20.f);
  GXPosition3f32(-24.f, 0.05f, 20.f);
  GXEnd();

  GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
  GXSetZMode(GX_TRUE, GX_LEQUAL, GX_TRUE);
}

// Sets every piece of state it draws with and restores blend and Z, so the depth probe and the next frame see M2 state
void draw_hud_panel() {
  push_scope(AURORA_CAT_UI, 0, AURORA_CAMERA_UI);
  Mtx44 ortho;
  C_MTXOrtho(ortho, 0.f, 480.f, 0.f, 640.f, 0.f, 1.f);
  GXSetProjection(ortho, GX_ORTHOGRAPHIC);
  Mtx identity;
  C_MTXIdentity(identity);
  hint_camera(AURORA_CAMERA_UI, identity, ortho, GX_ORTHOGRAPHIC);
  GXLoadPosMtxImm(identity, GX_PNMTX0);
  GXSetCurrentMtx(GX_PNMTX0);
  GXSetBlendMode(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR);
  GXSetZMode(GX_FALSE, GX_ALWAYS, GX_FALSE);
  GXSetFog(GX_FOG_NONE, 0.f, 0.f, 0.f, 0.f, ClearColor);
  GXSetCullMode(GX_CULL_NONE);
  GXSetNumChans(1);
  GXSetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_VTX, GX_SRC_VTX, GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
  set_pass_color_tev();
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxDesc(GX_VA_CLR0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT4, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT4, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);

  // Top left, clear of the depth probe's centre column
  constexpr f32 Corners[4][2] = {{16.f, 16.f}, {144.f, 16.f}, {144.f, 54.f}, {16.f, 54.f}};
  GXBegin(GX_QUADS, GX_VTXFMT4, 4);
  for (const auto& [x, y] : Corners) {
    GXPosition3f32(x, y, -0.5f);
    GXColor4u8(HudColor.r, HudColor.g, HudColor.b, HudColor.a);
  }
  GXEnd();

  GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
  GXSetZMode(GX_TRUE, GX_LEQUAL, GX_TRUE);
  pop_scope();
}

void emit_frame_hints(u32 frame, const Mtx view, const Mtx44 projection, const Lights& lights) {
  if (!hint_emission()) {
    return;
  }
  hint_camera(AURORA_CAMERA_MAIN, view, projection, GX_PERSPECTIVE);
  hint_scene_epoch(SceneEpoch);
  // Re-sent every frame, so a trace that starts on any frame still declares the arrays its boxes read
  hint_array_immutable(BoxPositions, sizeof(BoxPositions));
  hint_array_immutable(BoxNormals, sizeof(BoxNormals));
  hint_array_immutable(BoxColors, sizeof(BoxColors));

  const std::array<AuroraLight, 4> sceneLights{
      directional_light(1, lights.sunDir, lights.colors[0]),
      point_light(2, lights.positions[1], lights.colors[1], PointRefDistance, PointRefBrightness, GX_DA_MEDIUM),
      point_light(3, lights.positions[2], lights.colors[2], PointRefDistance, PointRefBrightness, GX_DA_MEDIUM),
      spot_light(4, lights.positions[3], lights.spotDir, lights.colors[3], SpotCutoff, SpotRefDistance,
                 SpotRefBrightness, GX_DA_GENTLE),
  };
  hint_lights(sceneLights);
  hint_environment(environment(AmbientColor, Fog, WindDirection, WindPower));

  const Vec tip = tentacle_tip(static_cast<f32>(frame) / 60.f);
  const Vec lastTip = frame == 0 ? tip : tentacle_tip(static_cast<f32>(frame - 1) / 60.f);
  const AuroraInfluencer influencer{
      .pos = {tip.x, tip.y, tip.z},
      .radius = 1.5f,
      .vel = {tip.x - lastTip.x, tip.y - lastTip.y, tip.z - lastTip.z},
      .kind = AURORA_INFLUENCER_PUSH,
  };
  hint_influencers({&influencer, 1});
  const AuroraWindSource wind{
      .pos = {0.f, 0.f, 0.f},
      .radius = 50.f,
      .dir = {WindDirection.x, WindDirection.y, WindDirection.z},
      .power = WindPower,
  };
  hint_wind_sources({&wind, 1});
  const AuroraFogVolume waterMist = fog_volume(Vec{-18.f, 1.f, 14.f}, Vec{6.f, 1.5f, 6.f}, 0.05f);
  hint_fog_volumes({&waterMist, 1});
}
std::array<f32, 2> project(const SceneCamera& camera, const Vec& world) noexcept {
  Vec view;
  C_MTXMultVec(camera.view, &world, &view);
  const auto& p = camera.projection;
  const f32 x = p[0][0] * view.x + p[0][1] * view.y + p[0][2] * view.z + p[0][3];
  const f32 y = p[1][0] * view.x + p[1][1] * view.y + p[1][2] * view.z + p[1][3];
  const f32 w = p[3][0] * view.x + p[3][1] * view.y + p[3][2] * view.z + p[3][3];
  return {x / w, y / w};
}

Vec on_plane(const SceneCamera& camera, std::array<f32, 2> ndc, f32 height) noexcept {
  const auto& p = camera.projection;
  const Vec viewDir{(ndc[0] + p[0][2]) / p[0][0], (ndc[1] + p[1][2]) / p[1][1], -1.f};
  Mtx inverse;
  C_MTXInverse(camera.view, inverse);
  Vec dir;
  C_MTXMultVecSR(inverse, &viewDir, &dir);
  const f32 s = (height - static_cast<f32>(camera.eye.y)) / dir.y;
  return {static_cast<f32>(camera.eye.x) + s * dir.x, height, static_cast<f32>(camera.eye.z) + s * dir.z};
}

std::array<f32, 2> difference(std::array<f32, 2> a, std::array<f32, 2> b) noexcept {
  return {a[0] - b[0], a[1] - b[1]};
}

Vec tentacle_probe_local(const SceneCamera& camera, const Mtx bone) noexcept {
  constexpr f32 Height = 1.f;
  constexpr f32 Sides = 10.f;
  const f32 radius = 0.28f - (Height - 0.75f) / 0.75f * 0.06f;
  Mtx inverse;
  C_MTXInverse(bone, inverse);
  const Vec eye{static_cast<f32>(camera.eye.x), static_cast<f32>(camera.eye.y), static_cast<f32>(camera.eye.z)};
  Vec local;
  C_MTXMultVec(inverse, &eye, &local);
  const f32 phi = std::atan2(local.z, local.x);
  const f32 step = 2.f * Pi / Sides;
  const f32 mid = (std::floor(phi / step) + 0.5f) * step;
  const f32 r = radius * std::cos(step / 2.f) / std::cos(phi - mid);
  return {r * std::cos(phi), Height, r * std::sin(phi)};
}
} // namespace

bool init_scene() {
  sGroundTexture = make_ground_texture(GroundTextureSize, GroundMipCount);
  if (sGroundTexture.size() !=
      GXGetTexBufferSize(GroundTextureSize, GroundTextureSize, GX_TF_RGBA8, GX_TRUE, GroundMipCount)) {
    return false;
  }
  GXInitTexObj(&sGroundTexObj, sGroundTexture.data(), GroundTextureSize, GroundTextureSize, GX_TF_RGBA8, GX_REPEAT,
               GX_REPEAT, GX_TRUE);
  GXInitTexObjLOD(&sGroundTexObj, GX_LIN_MIP_LIN, GX_LINEAR, 0.f, static_cast<f32>(GroundMipCount - 1), 0.f, GX_FALSE,
                  GX_TRUE, GX_ANISO_4);
  sTentacle = make_tentacle(GX_PNMTX1);
  record_box_display_list();
  return sBoxDisplayListSize != 0;
}

uint32_t gpu_uploads_per_frame(const SceneOptions& options) noexcept {
  const u64 padBytes = u64{options.gpuPadMiB} << 20;
  const u64 maxUpload = aurora::render::gpu::MaxUploadBytes;
  return (options.gpuArrays ? 1u : 0u) + static_cast<uint32_t>((padBytes + maxUpload - 1) / maxUpload);
}

SceneCamera scene_camera(u32 frame) noexcept {
  const f32 t = static_cast<f32>(frame) / 60.f;
  SceneCamera camera;
  C_MTXPerspective(camera.projection, 55.f, 4.f / 3.f, 0.5f, 400.f);
  camera.eye = {28.f * std::cos(0.25f * t), 11.f, 28.f * std::sin(0.25f * t)};
  const Vec up{0.f, 1.f, 0.f};
  camera.target = {0.f, 1.5f, 0.f};
  C_MTXLookAt(camera.view, &camera.eye, &up, &camera.target);
  return camera;
}

std::array<std::array<f32, 2>, 4> motion_probe_points(u32 frame) noexcept {
  const f32 t = static_cast<f32>(frame) / 60.f;
  const SceneCamera camera = scene_camera(frame);
  Mtx model;
  box_model(MotionProbeBox, t, model);
  const Vec boxLocal{0.f, 0.5f, 0.f};
  Vec boxTop;
  C_MTXMultVec(model, &boxLocal, &boxTop);
  Mtx bones[TentacleBones];
  tentacle_bones(t, bones);
  const Vec tentacleLocal = tentacle_probe_local(camera, bones[TentacleBones - 1]);
  Vec tentacle;
  C_MTXMultVec(bones[TentacleBones - 1], &tentacleLocal, &tentacle);
  return {MotionProbeGroundNdc, project(camera, boxTop), project(camera, tentacle), project(camera, WaterProbe)};
}

std::optional<std::array<f32, 2>> motion_probe_expected(MotionProbe probe, u32 frame, std::array<f32, 2> ndc) noexcept {
  if (frame == 0) {
    return std::nullopt;
  }
  const f32 t = static_cast<f32>(frame) / 60.f;
  const f32 previousT = static_cast<f32>(frame - 1) / 60.f;
  const SceneCamera camera = scene_camera(frame);
  const SceneCamera previous = scene_camera(frame - 1);
  switch (probe) {
  case MotionProbe::Ground:
    return difference(ndc, project(previous, on_plane(camera, ndc, 0.f)));
  case MotionProbe::Box: {
    Mtx model;
    Mtx previousModel;
    Mtx inverse;
    Mtx motion;
    box_model(MotionProbeBox, t, model);
    box_model(MotionProbeBox, previousT, previousModel);
    C_MTXInverse(model, inverse);
    C_MTXConcat(previousModel, inverse, motion);
    const Vec top = on_plane(camera, ndc, BoxPlacements[MotionProbeBox].scale);
    Vec before;
    C_MTXMultVec(motion, &top, &before);
    return difference(ndc, project(previous, before));
  }
  case MotionProbe::Tentacle: {
    Mtx bones[TentacleBones];
    Mtx previousBones[TentacleBones];
    tentacle_bones(t, bones);
    tentacle_bones(previousT, previousBones);
    const Vec local = tentacle_probe_local(camera, bones[TentacleBones - 1]);
    Vec now;
    Vec before;
    C_MTXMultVec(bones[TentacleBones - 1], &local, &now);
    C_MTXMultVec(previousBones[TentacleBones - 1], &local, &before);
    return difference(project(camera, now), project(previous, before));
  }
  default:
    return std::nullopt;
  }
}

const std::array<FogParams, 2>& fog_bands() noexcept { return FogBands; }

void draw_scene(u32 frame, const SceneOptions& options) {
  // Not gated on hints_enabled() so the off capture proves emitters add only hint bytes; real producers do gate on it
  set_hint_emission(options.hints);
  upload_gpu_arrays(options);
  const f32 t = static_cast<f32>(frame) / 60.f;
  const SceneCamera camera = scene_camera(frame);
  GXSetCopyClear(ClearColor, GX_MAX_Z24);
  GXSetViewport(0.f, 0.f, 640.f, 480.f, 0.f, 1.f);
  GXSetScissor(0, 0, 640, 480);

  GXSetProjection(camera.projection, GX_PERSPECTIVE);
  const auto& projection = camera.projection;
  const auto& view = camera.view;
  const Point3d& eye = camera.eye;
  const Point3d& target = camera.target;

  set_fog(Fog);
  const Lights lights = setup_lights(view, t);
  // Every frame-level hint goes out before SCENE_BEGIN, where aurora freezes the frame's values
  emit_frame_hints(frame, view, projection, lights);
  hint_stage(AURORA_STAGE_SCENE_BEGIN);

  if (options.fogBands) {
    set_fog(FogBands[0]);
  }
  begin_object(GroundObject, AURORA_CAT_STATIC | AURORA_CAT_TERRAIN, AURORA_OBJ_IMMOBILE);
  hint_material(GroundMaterial);
  draw_ground(view);
  end_object();
  if (options.fogBands) {
    set_fog(FogBands[1]);
  }
  if (options.customDrawProbe != aurora::gfx::InvalidDrawType) {
    aurora::gfx::push_custom_draw(options.customDrawProbe, nullptr, 0);
  }
  draw_boxes(view, t);

  Mtx bones[TentacleBones];
  tentacle_bones(t, bones);
  begin_object(TentacleObject, AURORA_CAT_DYNAMIC | AURORA_CAT_SKINNED, 0);
  hint_skeleton(TentacleObject, bones);
  draw_tentacle(view, bones);
  end_object();

  draw_light_markers(view, lights);
  if (options.captureOnlyBox) {
    draw_capture_only_box(view, eye, target);
  }
  hint_stage(AURORA_STAGE_OPAQUE_DONE);

  // The water never reads K1; the policy gives the decoder and tevPolicyDraws a producer that CI runs
  AuroraTevPolicy waterPolicy{};
  waterPolicy.mask = AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K1, AURORA_TEV_CH_R);
  begin_object(WaterObject, AURORA_CAT_WATER | AURORA_CAT_NO_PBR, 0);
  hint_tev_policy(waterPolicy);
  draw_water(view);
  end_object();

  hint_stage(AURORA_STAGE_PRE_POST);
  hint_stage(AURORA_STAGE_SCENE_END);
  hint_stage(AURORA_STAGE_BEFORE_HUD);
  draw_hud_panel();
  hint_stage(AURORA_STAGE_FRAME_END);
}

void shutdown_scene() {
  sBoxArrays = aurora::render::gpu::InvalidHandle;
  sBoxPositions = aurora::render::gpu::InvalidHandle;
  GXDestroyTexObj(&sGroundTexObj);
  sGroundTexture = {};
  sTentacle = {};
  sBoxDisplayListSize = 0;
}
} // namespace render_demo
