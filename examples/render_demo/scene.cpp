#include "scene.hpp"

#include "procedural.hpp"

#include <dolphin/gx.h>
#include <dolphin/mtx.h>

#include <array>
#include <cmath>
#include <vector>

namespace render_demo {
namespace {
constexpr f32 Pi = 3.14159265358979f;
constexpr GXColor ClearColor{112, 160, 208, 255};
constexpr GXColor AmbientColor{40, 44, 56, 255};
constexpr GXColor White{255, 255, 255, 255};
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
};

alignas(32) u8 sBoxDisplayList[256];
u32 sBoxDisplayListSize = 0;
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

void set_box_format() {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX16);
  GXSetVtxDesc(GX_VA_NRM, GX_INDEX16);
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT1, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  GXSetArray(GX_VA_POS, BoxPositions, sizeof(BoxPositions), sizeof(BoxPositions[0]), true);
  GXSetArray(GX_VA_NRM, BoxNormals, sizeof(BoxNormals), sizeof(BoxNormals[0]), true);
  GXSetArray(GX_VA_CLR0, BoxColors, sizeof(BoxColors), sizeof(BoxColors[0]), true);
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
  GXLightObj light;

  Vec sunDir{-0.4f, -1.f, -0.3f};
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
    GXInitLightDistAttn(&light, 10.f, 0.5f, GX_DA_MEDIUM);
    GXLoadLightObjImm(&light, PointIds[i]);
  }

  lights.positions[3] = Vec{0.f, 10.f, 0.f};
  lights.colors[3] = GXColor{255, 250, 220, 255};
  Vec spotDir{0.3f * std::sin(0.5f * t), -1.f, 0.3f * std::cos(0.5f * t)};
  C_VECNormalize(&spotDir, &spotDir);
  Vec spotDirView;
  C_MTXMultVecSR(view, &spotDir, &spotDirView);
  const Vec spotPos = to_view(view, lights.positions[3]);
  GXInitLightPos(&light, spotPos.x, spotPos.y, spotPos.z);
  GXInitLightDir(&light, spotDirView.x, spotDirView.y, spotDirView.z);
  GXInitLightColor(&light, lights.colors[3]);
  GXInitLightSpot(&light, 25.f, GX_SP_COS2);
  GXInitLightDistAttn(&light, 15.f, 0.5f, GX_DA_GENTLE);
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

void draw_boxes(const Mtx view, f32 t) {
  set_lit_channel(GX_SRC_VTX);
  set_pass_color_tev();
  GXSetCurrentMtx(GX_PNMTX0);
  GXSetCullMode(GX_CULL_BACK);
  for (size_t i = 0; i < BoxPlacements.size(); ++i) {
    const auto& box = BoxPlacements[i];
    Mtx translate;
    Mtx rotate;
    Mtx scale;
    Mtx model;
    Mtx modelView;
    C_MTXTrans(translate, box.x, box.scale / 2.f, box.z);
    C_MTXRotRad(rotate, 'y', t * (0.3f + 0.15f * static_cast<f32>(i)));
    C_MTXScale(scale, box.scale, box.scale, box.scale);
    C_MTXConcat(translate, rotate, model);
    C_MTXConcat(model, scale, model);
    C_MTXConcat(view, model, modelView);
    load_matrices(modelView, GX_PNMTX0);
    set_box_format();
    GXCallDisplayList(sBoxDisplayList, sBoxDisplayListSize);
  }
}

void bone_joint(Mtx out, f32 t, int k) {
  Mtx rz;
  Mtx rx;
  C_MTXRotRad(rz, 'z', 0.35f * std::sin(1.7f * t + 0.8f * static_cast<f32>(k)));
  C_MTXRotRad(rx, 'x', 0.2f * std::sin(1.1f * t + 0.5f * static_cast<f32>(k)));
  C_MTXConcat(rz, rx, out);
}

void draw_tentacle(const Mtx view, f32 t) {
  Mtx bone;
  Mtx joint;
  C_MTXTrans(bone, 6.f, 0.f, -2.f);
  for (int k = 0; k < 4; ++k) {
    if (k != 0) {
      Mtx offset;
      C_MTXTrans(offset, 0.f, 1.5f, 0.f);
      C_MTXConcat(bone, offset, bone);
    }
    bone_joint(joint, t, k);
    C_MTXConcat(bone, joint, bone);
    Mtx modelView;
    C_MTXConcat(view, bone, modelView);
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
  for (int i = 1; i < 4; ++i) {
    Mtx model;
    Mtx scale;
    Mtx modelView;
    C_MTXTrans(model, lights.positions[i].x, lights.positions[i].y, lights.positions[i].z);
    C_MTXScale(scale, 0.35f, 0.35f, 0.35f);
    C_MTXConcat(model, scale, model);
    C_MTXConcat(view, model, modelView);
    load_matrices(modelView, GX_PNMTX0);
    GXSetChanMatColor(GX_COLOR0A0, lights.colors[i]);
    set_box_format();
    GXCallDisplayList(sBoxDisplayList, sBoxDisplayListSize);
  }
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

void draw_scene(u32 frame) {
  const f32 t = static_cast<f32>(frame) / 60.f;
  GXSetCopyClear(ClearColor, GX_MAX_Z24);
  GXSetViewport(0.f, 0.f, 640.f, 480.f, 0.f, 1.f);
  GXSetScissor(0, 0, 640, 480);

  Mtx44 projection;
  C_MTXPerspective(projection, 55.f, 4.f / 3.f, 0.5f, 400.f);
  GXSetProjection(projection, GX_PERSPECTIVE);

  const Point3d eye{28.f * std::cos(0.25f * t), 11.f, 28.f * std::sin(0.25f * t)};
  const Vec up{0.f, 1.f, 0.f};
  const Point3d target{0.f, 1.5f, 0.f};
  Mtx view;
  C_MTXLookAt(view, &eye, &up, &target);

  GXSetFog(GX_FOG_PERSP_LIN, 40.f, 160.f, 0.5f, 400.f, ClearColor);
  const Lights lights = setup_lights(view, t);
  draw_ground(view);
  draw_boxes(view, t);
  draw_tentacle(view, t);
  draw_light_markers(view, lights);
  draw_water(view);
}

void shutdown_scene() {
  GXDestroyTexObj(&sGroundTexObj);
  sGroundTexture = {};
  sTentacle = {};
  sBoxDisplayListSize = 0;
}
} // namespace render_demo
