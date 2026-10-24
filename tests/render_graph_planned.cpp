#include "render_graph_planned.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

namespace aurora::render::detail::planned {
namespace {
struct PlannedUse {
  std::string op;
  std::string name;
  bool buffer = false;
  Access access = Access::Sampled;
  Missing missing = Missing::DisablePass;
  bool vertex = false;
  Owner owner = Owner::Dawn;
};

struct PlannedPass {
  std::string_view name;
  Stage stage;
  PassKind kind;
  PassClass cls;
  std::string_view anchor;
  uint16_t order;
  std::string_view uses;
  bool sideEffect = false;
  std::vector<PlannedUse> parsed;
};

struct PlannedResource {
  std::string_view name;
  TexDesc tex;
  BufDesc buf;
  uint32_t handle = 0;
};

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;

TexDesc efb(wgpu::TextureFormat format, uint16_t divisor = 1, uint32_t mips = 1) {
  return {.format = format, .sizeClass = SizeClass::Efb, .divisor = divisor, .mips = mips};
}

TexDesc absolute(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

TexDesc volume(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t depth) {
  return {.format = format,
          .dimension = wgpu::TextureDimension::e3D,
          .sizeClass = SizeClass::Absolute,
          .width = width,
          .height = height,
          .depthOrLayers = depth};
}

BufDesc bytes(uint64_t size) { return {.size = size}; }

using wgpu::TextureFormat;

const std::vector<PlannedResource>& resources() {
  static const std::vector<PlannedResource> table{
      {.name = "terrain.base", .handle = 1},
      {.name = "terrain.records", .handle = 2},
      {.name = "terrain.args", .handle = 3},
      {.name = "scene.clusters", .handle = 4},
      {.name = "scene.vertices", .handle = 5},
      {.name = "grass.patches", .handle = 10},
      {.name = "bvh.nodes", .handle = 11},
      {.name = "bvh.tlas", .handle = 12},
      {.name = "vsm.pt", .handle = 16},
      {.name = "light.list", .handle = 18},
      {.name = "sky.sh9", .handle = 20},
      {.name = "exposure", .handle = 22},
      {.name = "fog.volumes", .handle = 23},
      {.name = "vg.meshes", .handle = 24},
      {.name = "voxel.albedo", .tex = volume(TextureFormat::RGBA8Unorm, 128, 128, 128), .handle = 6},
      {.name = "voxel.opacity", .tex = volume(TextureFormat::R8Unorm, 128, 128, 128), .handle = 7},
      {.name = "voxel.radiance", .tex = volume(TextureFormat::RGBA16Float, 128, 128, 128), .handle = 8},
      {.name = "grass.clipmap", .tex = absolute(TextureFormat::RGBA16Float, 512, 512, 4), .handle = 9},
      {.name = "clouds.weather", .tex = absolute(TextureFormat::RGBA8Unorm, 512, 512), .handle = 13},
      {.name = "clouds.noise", .tex = volume(TextureFormat::RGBA8Unorm, 128, 128, 128), .handle = 14},
      {.name = "csm", .tex = absolute(TextureFormat::RGBA16Float, 512, 512), .handle = 15},
      {.name = "vsm.pool", .tex = absolute(TextureFormat::R32Uint, 4096, 4096), .handle = 17},
      {.name = "sky.probe", .tex = absolute(TextureFormat::RGBA16Float, 64, 64), .handle = 19},
      {.name = "clouds.probe", .tex = absolute(TextureFormat::RGBA16Float, 64, 64), .handle = 21},
      {.name = "rt.native.rays", .handle = 25},
      {.name = "rt.native.hits", .handle = 26},
      {.name = "env.ub", .buf = bytes(1 * KiB)},
      {.name = "terrain.patches", .buf = bytes(64 * KiB)},
      {.name = "vg.visible", .buf = bytes(1 * MiB)},
      {.name = "vg.args", .buf = bytes(4 * KiB)},
      {.name = "grass.tiles", .buf = bytes(256 * KiB)},
      {.name = "grass.blades", .buf = bytes(8 * MiB)},
      {.name = "grass.indices", .buf = bytes(4 * MiB)},
      {.name = "grass.args", .buf = bytes(256)},
      {.name = "grass.long_blades", .buf = bytes(4 * MiB)},
      {.name = "grass.long_args", .buf = bytes(256)},
      {.name = "light.grid", .buf = bytes(2 * MiB)},
      {.name = "vsm.draws", .buf = bytes(1 * MiB)},
      {.name = "rt.rays", .buf = bytes(32 * MiB)},
      {.name = "rt.hits", .buf = bytes(32 * MiB)},
      {.name = "fog.tiles", .buf = bytes(64 * KiB)},
      {.name = "fog.mbar", .buf = bytes(4 * KiB)},
      {.name = "sky.probe.raw", .tex = absolute(TextureFormat::RGBA16Float, 64, 64)},
      {.name = "hzb.depth", .tex = efb(TextureFormat::R32Float, 2, 0)},
      {.name = "ao.hiz", .tex = efb(TextureFormat::R32Float, 1, 0)},
      {.name = "grass.hzb", .tex = efb(TextureFormat::R32Float, 16)},
      {.name = "vg.vis", .tex = efb(TextureFormat::R32Uint)},
      {.name = "lsm.mask", .tex = efb(TextureFormat::R8Unorm)},
      {.name = "lsm.depth", .tex = efb(TextureFormat::Depth32Float)},
      {.name = "legacyTex", .tex = efb(TextureFormat::BGRA8Unorm)},
      {.name = "ao.raw", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "vsm.sun", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "rt.shadow", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "rt.ao", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "vis.sun", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "ssil.raw", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "ssil.irr", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "gi.irr", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "scene.hdr", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "refl.spec", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "clouds.full", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "gi.raw", .tex = efb(TextureFormat::RGBA16Float, 2)},
      {.name = "refl.raw", .tex = efb(TextureFormat::RGBA16Float, 2)},
      {.name = "clouds.trace", .tex = efb(TextureFormat::RGBA16Float, 2)},
      {.name = "godrays.mask", .tex = efb(TextureFormat::RGBA8Unorm, 2)},
      {.name = "godrays.blur", .tex = efb(TextureFormat::RGBA8Unorm, 2)},
      {.name = "fog.integrated", .tex = volume(TextureFormat::RGBA16Float, 160, 90, 64)},
      {.name = "fog.relight", .tex = volume(TextureFormat::RGBA16Float, 160, 90, 64)},
      {.name = "grass.influence", .tex = absolute(TextureFormat::RGBA16Float, 256, 256)},
      {.name = "ao.final", .tex = efb(TextureFormat::RGBA8Unorm)},
      {.name = "rt.moments", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "radiance.defogged", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "taa.color", .tex = efb(TextureFormat::RGBA16Float)},
      {.name = "clouds.hist", .tex = efb(TextureFormat::RGBA16Float, 2)},
      {.name = "fog.inject", .tex = volume(TextureFormat::RGBA16Float, 160, 90, 64)},
  };
  return table;
}

const PlannedResource& resource(std::string_view name) {
  const auto& table = resources();
  const auto it = std::ranges::find(table, name, &PlannedResource::name);
  if (it == table.end()) {
    std::abort();
  }
  return *it;
}

std::string_view trim(std::string_view text) {
  while (!text.empty() && text.front() == ' ') {
    text.remove_prefix(1);
  }
  while (!text.empty() && text.back() == ' ') {
    text.remove_suffix(1);
  }
  return text;
}

bool take_suffix(std::string& text, std::string_view suffix) {
  const auto at = text.find(suffix);
  if (at == std::string::npos) {
    return false;
  }
  text.erase(at, suffix.size());
  return true;
}

Access parse_access(std::string_view token) {
  constexpr std::array<std::pair<std::string_view, Access>, 13> Accesses{{
      {"S", Access::Sampled},
      {"U", Access::Uniform},
      {"SR", Access::StorageRead},
      {"SW", Access::StorageWrite},
      {"SRW", Access::StorageRW},
      {"CW", Access::ColorWrite},
      {"CRMW", Access::ColorRMW},
      {"DR", Access::DepthRead},
      {"DW", Access::DepthWrite},
      {"CS", Access::CopySrc},
      {"CD", Access::CopyDst},
      {"I", Access::Index},
      {"Ind", Access::Indirect},
  }};
  const auto it = std::ranges::find(Accesses, token, &std::pair<std::string_view, Access>::first);
  if (it == Accesses.end()) {
    std::abort();
  }
  return it->second;
}

// Uses keep the planned table's compact notation, for example "u vis.sun S!W; i rt.native.hits SW (NativeD3D12)"
std::vector<PlannedUse> parse_uses(std::string_view text) {
  std::vector<PlannedUse> uses;
  while (!text.empty()) {
    const auto end = text.find(';');
    std::string_view item = trim(text.substr(0, end));
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    PlannedUse use;
    const auto opEnd = item.find(' ');
    use.op = std::string{item.substr(0, opEnd)};
    item = trim(item.substr(opEnd + 1));
    const auto nameEnd = item.find(' ');
    use.name = std::string{item.substr(0, nameEnd)};
    use.buffer = take_suffix(use.name, "[b]");
    if (use.op == "p") {
      use.missing = Missing::BindDummyBlack;
      uses.push_back(std::move(use));
      continue;
    }
    std::string access{trim(item.substr(nameEnd + 1))};
    if (take_suffix(access, " (NativeD3D12)")) {
      use.owner = Owner::NativeD3D12;
    }
    use.vertex = take_suffix(access, "@v");
    if (take_suffix(access, "!B")) {
      use.missing = Missing::BindDummyBlack;
    } else if (take_suffix(access, "!W")) {
      use.missing = Missing::BindDummyWhite;
    }
    use.access = parse_access(access);
    uses.push_back(std::move(use));
  }
  return uses;
}

void declare_uses(InternalPassBuilder& builder, const std::vector<PlannedUse>& uses) {
  for (const PlannedUse& use : uses) {
    const UseOptions options{
        .missing = use.missing,
        .visibility = use.vertex ? wgpu::ShaderStage::Vertex : wgpu::ShaderStage::None,
    };
    if (use.op == "c") {
      const auto& desc = resource(use.name);
      if (use.buffer) {
        builder.create_buffer(use.name, desc.buf, use.access);
      } else {
        builder.create(use.name, desc.tex, use.access);
      }
    } else if (use.op == "h") {
      builder.history(use.name, resource(use.name).tex, use.access);
    } else if (use.op == "u") {
      if (use.buffer) {
        builder.use_buffer(use.name, use.access, options);
      } else {
        builder.use(use.name, use.access, options);
      }
    } else if (use.op == "p") {
      builder.use_prev(use.name, use.missing);
    } else if (use.op == "i") {
      // The import call has no visibility, so a vertex-stage read of an import goes through use_buffer
      if (use.vertex) {
        builder.use_buffer(use.name, use.access, options);
      } else {
        builder.import_persistent(use.name, {resource(use.name).handle}, use.access, use.owner);
      }
    } else if (use.op == "it") {
      const auto& desc = resource(use.name);
      builder.import_persistent_texture(use.name, {desc.handle}, desc.tex, use.access, use.owner);
    } else {
      std::abort();
    }
  }
}

void declare_planned(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& pass = *static_cast<const PlannedPass*>(state);
  builder.after(pass.anchor);
  declare_uses(builder, pass.parsed);
  if (pass.sideEffect) {
    builder.side_effect();
  }
}

struct PlannedGxRead {
  std::string_view node;
  std::string_view uses;
  std::vector<PlannedUse> parsed;
};

void declare_gx_read(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  declare_uses(builder, static_cast<const PlannedGxRead*>(state)->parsed);
}

constexpr auto SceneBegin = Stage::SceneBegin;
constexpr auto OpaqueDone = Stage::OpaqueDone;
constexpr auto PrePost = Stage::PrePost;
constexpr auto BeforeHud = Stage::BeforeHud;
constexpr auto FrameEnd = Stage::FrameEnd;
constexpr auto AfterFrame = Stage::AfterFrame;
constexpr auto Compute = PassKind::Compute;
constexpr auto Render = PassKind::Render;
constexpr auto Copy = PassKind::Copy;
constexpr auto Encoder = PassKind::Encoder;
constexpr auto Native = PassKind::Native;
constexpr auto Required = PassClass::Required;
constexpr auto Optional = PassClass::Optional;

std::vector<PlannedPass>& passes() {
  static std::vector<PlannedPass> table = [] {
    std::vector<PlannedPass> rows{
        {"env.decode", SceneBegin, Compute, Required, "phase.scene_prep", 0, "c env.ub[b] SW"},
        {"grass.influence", SceneBegin, Compute, Required, "phase.scene_prep", 10,
         "u env.ub[b] U; h grass.influence SW"},
        {"sky.probe_project", SceneBegin, Compute, Required, "phase.scene_prep", 20,
         "u gx.color S; c sky.probe.raw SW"},
        {"clouds.probe", SceneBegin, Compute, Required, "phase.scene_prep", 21,
         "it clouds.weather S; it clouds.noise S; it clouds.probe SW"},
        {"sky.probe_filter", SceneBegin, Compute, Required, "phase.scene_prep", 22,
         "u sky.probe.raw S; it clouds.probe S; it sky.probe SW; i sky.sh9 SW"},
        {"terrain.classify", SceneBegin, Compute, Required, "phase.scene_prep", 30,
         "i terrain.base SR; u grass.influence S; u env.ub[b] U; c terrain.patches[b] SW"},
        {"terrain.split", SceneBegin, Compute, Required, "phase.scene_prep", 31, "u terrain.patches[b] SRW"},
        {"terrain.emit", SceneBegin, Compute, Required, "phase.scene_prep", 32,
         "u terrain.patches[b] SR; i terrain.base SR; i terrain.records SW; i terrain.args SW"},
        {"gi.voxelize", SceneBegin, Compute, Required, "phase.scene_prep", 40,
         "i scene.clusters SR; i scene.vertices SR; it voxel.albedo SW; it voxel.opacity SW"},
        {"scene.streamout", OpaqueDone, Compute, Required, "phase.capture", 0,
         "i scene.vertices SW; i scene.clusters SW"},
        {"capture.replay", OpaqueDone, Encoder, Required, "phase.capture", 10,
         "u gx.depth S; c lsm.mask CW; c lsm.depth DW; it grass.clipmap CRMW"},
        {"grass.clipmap", OpaqueDone, Compute, Required, "phase.capture", 20,
         "i scene.vertices SR; it grass.clipmap SW"},
        {"rt.blas_refit", OpaqueDone, Compute, Required, "phase.capture", 30, "i scene.vertices SR; i bvh.nodes SRW"},
        {"rt.tlas", OpaqueDone, Compute, Required, "phase.capture", 31, "i bvh.nodes SR; i bvh.tlas SW"},
        {"hzb.opaque", OpaqueDone, Compute, Required, "phase.geometry", 0, "u gx.depth S; c hzb.depth SW"},
        {"vg.cull", OpaqueDone, Compute, Required, "phase.geometry", 10,
         "i vg.meshes SR; u hzb.depth S; c vg.visible[b] SW; c vg.args[b] SW"},
        {"vg.raster", OpaqueDone, Render, Required, "phase.geometry", 11,
         "u vg.visible[b] SR@v; u vg.args[b] Ind; i vg.meshes SR@v; c vg.vis CW; u gx.depth DW"},
        {"vg.resolve", OpaqueDone, Render, Required, "phase.geometry", 12,
         "u vg.vis S; i vg.meshes SR; u gx.color CRMW; u gx.normal CRMW; u gbuf.motion CRMW; u gbuf.fog CRMW; "
         "u gbuf.surface CRMW; u gx.depth DR"},
        {"grass.hzb", OpaqueDone, Compute, Required, "phase.geometry", 20, "u gx.depth S; c grass.hzb SW"},
        {"grass.cull", OpaqueDone, Compute, Required, "phase.geometry", 21,
         "u grass.hzb S; it grass.clipmap S; u grass.influence S; i grass.patches SR; c grass.tiles[b] SW"},
        {"grass.generate", OpaqueDone, Compute, Required, "phase.geometry", 22,
         "u grass.tiles[b] SR; it grass.clipmap S; c grass.blades[b] SW"},
        {"grass.compact", OpaqueDone, Compute, Required, "phase.geometry", 23,
         "u grass.blades[b] SR; c grass.indices[b] SW; c grass.args[b] SW"},
        {"grass.expand_long", OpaqueDone, Compute, Required, "phase.geometry", 24,
         "i grass.patches SR; u grass.influence S; c grass.long_blades[b] SW; c grass.long_args[b] SW"},
        {"grass.draw", OpaqueDone, Render, Required, "phase.geometry", 25,
         "u grass.blades[b] SR@v; u grass.indices[b] I; u grass.args[b] Ind; u grass.long_blades[b] SR@v; "
         "u grass.long_args[b] Ind; u lsm.mask S!W; u gx.color CRMW; u gx.normal CRMW; u gbuf.motion CRMW; "
         "u gbuf.fog CRMW; u gbuf.surface CRMW; u gx.depth DW"},
        {"grass.far_field", OpaqueDone, Render, Required, "phase.geometry", 26,
         "u gx.depth S; u gbuf.fog S; it grass.clipmap S; u gx.color CRMW; u gbuf.surface CRMW; u gx.normal CRMW"},
        {"legacy.copy", OpaqueDone, Copy, Required, "phase.prepare", 0, "u gx.color CS; c legacyTex CD"},
        {"depth.pyramid", OpaqueDone, Compute, Required, "phase.prepare", 1, "u gx.depth S; c ao.hiz SW"},
        {"light.clusters", OpaqueDone, Compute, Required, "phase.prepare", 2, "i light.list SR; c light.grid[b] SW"},
        {"ao.gtao", OpaqueDone, Compute, Required, "phase.ao", 0,
         "u ao.hiz S; u gx.normal S; p radiance.defogged; c ao.raw SW; c ssil.raw SW"},
        {"ao.denoise", OpaqueDone, Compute, Required, "phase.ao", 1,
         "u ao.raw S; u ssil.raw S; u gx.depth S; h ao.final SW; c ssil.irr SW"},
        {"clouds.weather", OpaqueDone, Compute, Required, "phase.cloud_shadows", 0, "it clouds.weather SW"},
        {"csm.update", OpaqueDone, Compute, Required, "phase.cloud_shadows", 1, "it clouds.weather S; it csm SW"},
        {"vsm.invalidate", OpaqueDone, Compute, Required, "phase.shadows", 0, "i scene.clusters SR; i vsm.pt SRW"},
        {"vsm.mark", OpaqueDone, Compute, Required, "phase.shadows", 1,
         "u gx.depth S; u light.grid[b] SR; i vsm.pt SRW"},
        {"vsm.alloc", OpaqueDone, Compute, Required, "phase.shadows", 2, "i vsm.pt SRW"},
        {"vsm.clear", OpaqueDone, Compute, Required, "phase.shadows", 3, "i vsm.pt SR; it vsm.pool SRW"},
        {"vsm.cull", OpaqueDone, Compute, Required, "phase.shadows", 4,
         "i vsm.pt SR; i scene.clusters SR; c vsm.draws[b] SW"},
        {"vsm.raster", OpaqueDone, Compute, Required, "phase.shadows", 5,
         "u vsm.draws[b] SR; i scene.vertices SR; i vsm.pt SR; it vsm.pool SRW"},
        {"vsm.filter", OpaqueDone, Compute, Required, "phase.shadows", 6, "it vsm.pool S; u gx.depth S; c vsm.sun SW"},
        {"rt.raygen.shadow", OpaqueDone, Compute, Required, "phase.shadows", 10,
         "u gx.depth S; u gx.normal S; c rt.rays[b] SW"},
        {"rt.trace", OpaqueDone, Compute, Required, "phase.shadows", 11,
         "u rt.rays[b] SR; i bvh.nodes SR; i bvh.tlas SR; c rt.hits[b] SW"},
        {"rt.denoise", OpaqueDone, Compute, Required, "phase.shadows", 12,
         "u rt.hits[b] SR; u gx.depth S; h rt.moments SW; c rt.shadow SW; c rt.ao SW"},
        {"ao.combine", OpaqueDone, Render, Required, "phase.shadows", 13, "u rt.ao S; u ao.final CRMW"},
        {"sun.visibility", OpaqueDone, Compute, Required, "phase.shadows", 20,
         "u vsm.sun S!W; u rt.shadow S!W; it csm S; u gx.depth S; u gx.normal S; c vis.sun SW"},
        {"grass.contact", OpaqueDone, Render, Required, "phase.shadows", 21, "u gx.depth S; u vis.sun CRMW"},
        {"gi.inject", OpaqueDone, Compute, Required, "phase.gi", 0,
         "it voxel.albedo S; it voxel.opacity S; it vsm.pool S; it csm S; u vis.sun S!W; it voxel.radiance SW"},
        {"gi.cones", OpaqueDone, Compute, Required, "phase.gi", 1,
         "it voxel.radiance S; u gx.depth S; u gx.normal S; c gi.raw SW"},
        {"gi.upsample", OpaqueDone, Compute, Required, "phase.gi", 2, "u gi.raw S; u gx.depth S; c gi.irr SW"},
        {"light.deferred", OpaqueDone, Compute, Required, "phase.lighting", 0,
         "u gbuf.surface S; u gx.normal S; u gx.depth S; u light.grid[b] SR; i light.list SR; u ao.final S!W; "
         "u ssil.irr S!B; it vsm.pool S; i vsm.pt SR; u vis.sun S!W; u rt.shadow S!W; u gi.irr S!B; "
         "it sky.probe S; i sky.sh9 SR; c scene.hdr SW"},
        {"refl.ssr", OpaqueDone, Compute, Required, "phase.reflections", 0,
         "u ao.hiz S; p radiance.defogged; u gx.normal S; u gbuf.surface S; c refl.raw SW"},
        {"refl.rt", OpaqueDone, Compute, Required, "phase.reflections", 1,
         "i bvh.nodes SR; i bvh.tlas SR; u gx.depth S; u gx.normal S; u refl.raw SW"},
        {"refl.denoise", OpaqueDone, Compute, Required, "phase.reflections", 2, "u refl.raw S; c refl.spec SW"},
        {"refl.apply", OpaqueDone, Render, Required, "phase.reflections", 3, "u refl.spec S; u scene.hdr CRMW"},
        {"exposure.meter", OpaqueDone, Compute, Required, "phase.exposure", 0,
         "u scene.hdr S; u legacyTex S; u gbuf.fog S; i exposure SRW"},
        {"clouds.trace", OpaqueDone, Compute, Required, "phase.sky", 0,
         "it clouds.weather S; it clouds.noise S; it csm S; u gx.depth S; i exposure SR; c clouds.trace SW"},
        {"clouds.reconstruct", OpaqueDone, Compute, Required, "phase.sky", 1, "u clouds.trace S; h clouds.hist SW"},
        {"clouds.upsample", OpaqueDone, Compute, Required, "phase.sky", 2,
         "u clouds.hist S; u gx.depth S; c clouds.full SW"},
        {"godrays.mask", OpaqueDone, Compute, Required, "phase.sky", 10,
         "u legacyTex S; u gx.depth S; u gbuf.fog S; u clouds.full S; c godrays.mask SW"},
        {"fog.cull", OpaqueDone, Compute, Required, "phase.volumetrics", 0, "i fog.volumes SR; c fog.tiles[b] SW"},
        {"fog.inject", OpaqueDone, Compute, Required, "phase.volumetrics", 1,
         "u fog.tiles[b] SR; it vsm.pool S; it csm S; u light.grid[b] SR; i light.list SR; i sky.sh9 SR; "
         "u gi.irr S!B; h fog.inject SW"},
        {"fog.integrate", OpaqueDone, Compute, Required, "phase.volumetrics", 2,
         "u fog.inject S; c fog.integrated SW; c fog.relight SW; c fog.mbar[b] SW"},
        {"composite.main", OpaqueDone, Render, Required, "phase.composite", 0,
         "u legacyTex S; u gbuf.fog S; u gx.normal S; u gx.depth S; u ao.final S!W; u ssil.irr S!B; "
         "u vis.sun S!W; u scene.hdr S!B; u gbuf.surface S!B; u gi.irr S!B; u clouds.full S!B; "
         "u fog.integrated S!B; u fog.relight S!B; u lsm.mask S!W; i exposure SR; u fog.mbar[b] SR!B; "
         "u gx.color CRMW; h radiance.defogged CW"},
        {"godrays.blur", PrePost, Compute, Required, "phase.pre_taa", 0, "u godrays.mask S; c godrays.blur SW"},
        {"godrays.apply", PrePost, Render, Required, "phase.pre_taa", 1, "u godrays.blur S; u gx.color CRMW"},
        {"aurora.taa", PrePost, Encoder, Required, "phase.taa", 0,
         "u gx.color CRMW; u gx.depth S; u gbuf.motion S; u gbuf.reactive S; u clouds.full S!B; h taa.color CW"},
        {"demo.vignette", BeforeHud, Render, Optional, "phase.before_hud", 0, "u gx.color CRMW"},
        {"debug.capture", FrameEnd, Copy, PassClass::Diagnostic, "phase.frame_end", 0, "u gx.color CS", true},
        {"native.dxr", AfterFrame, Native, Optional, "phase.after_frame", 0,
         "i rt.native.rays SR (NativeD3D12); i rt.native.hits SW (NativeD3D12)"},
    };
    for (auto& row : rows) {
      row.parsed = parse_uses(row.uses);
    }
    return rows;
  }();
  return table;
}

std::vector<PlannedGxRead>& gx_reads() {
  static std::vector<PlannedGxRead> table = [] {
    constexpr std::string_view Overlay =
        "u vis.sun S!W; u fog.integrated S!B; u fog.relight S!B; u ao.final S!W; "
        "u csm S!W";
    std::vector<PlannedGxRead> rows{
        {"gx@OpaqueDone", "u terrain.records[b] SR@v!B; u terrain.args[b] Ind!B"},
        {"gx.overlay", Overlay},
        {"gx@PrePost", Overlay},
    };
    for (auto& row : rows) {
      row.parsed = parse_uses(row.uses);
    }
    return rows;
  }();
  return table;
}
} // namespace

std::vector<PassDesc> planned_passes(bool reversed) {
  std::vector<PassDesc> descs;
  for (const PlannedPass& pass : passes()) {
    descs.push_back({
        .name = std::string{pass.name},
        .stage = pass.stage,
        .kind = pass.kind,
        .cls = pass.cls,
        .order = pass.order,
        .declare = declare_planned,
        .state = &pass,
    });
  }
  if (reversed) {
    std::ranges::reverse(descs);
  }
  return descs;
}

std::vector<GxReadDesc> planned_gx_reads() {
  std::vector<GxReadDesc> descs;
  for (const PlannedGxRead& read : gx_reads()) {
    descs.push_back({.node = read.node, .declare = declare_gx_read, .state = &read});
  }
  return descs;
}

GraphLimits full_limits() {
  constexpr BindingCounts PerStage{
      .sampledTextures = 32, .samplers = 16, .storageBuffers = 10, .storageTextures = 8, .uniformBuffers = 12};
  return {
      .vertex = PerStage,
      .fragment = PerStage,
      .compute = PerStage,
      .dynamicUniformBuffers = 8,
      .colorAttachments = 8,
      .colorAttachmentBytesPerSample = 128,
      .textureDimension2D = 16384,
      .textureDimension3D = 2048,
      .textureArrayLayers = 2048,
      .bufferSize = 1ull << 30,
      .storageBufferBindingSize = 1ull << 30,
      .uniformBufferBindingSize = 64 * KiB,
      .features = {.core = true, .tier1Formats = true, .rg11b10Renderable = true, .float32Filterable = true},
  };
}

GraphLimits metal32_limits() {
  GraphLimits limits = full_limits();
  limits.colorAttachmentBytesPerSample = 32;
  limits.features.tier1Formats = false;
  limits.features.rg11b10Renderable = false;
  return limits;
}

GraphLimits compat_limits() {
  GraphLimits limits{
      .vertex = DefaultStageLimits,
      .fragment = DefaultStageLimits,
      .compute = DefaultStageLimits,
      .dynamicUniformBuffers = 8,
      .colorAttachments = 4,
      .colorAttachmentBytesPerSample = 32,
      .textureDimension2D = 4096,
      .textureArrayLayers = 256,
      .uniformBufferBindingSize = 16384,
      .features = {.core = false},
  };
  limits.vertex.storageBuffers = 2;
  limits.vertex.storageTextures = 0;
  limits.fragment.storageBuffers = 4;
  limits.fragment.storageTextures = 4;
  return limits;
}

GraphInputs planned_inputs(const GraphLimits& limits, std::span<const PassDesc> passes,
                           std::span<const GxReadDesc> gxReads) {
  return {
      .passes = passes,
      .gxReads = gxReads,
      .env =
          {
              .limits = limits,
              .efbSize = {2560, 1440},
              .outputSize = {2560, 1440},
              .gxFormats = gx_formats_tier_p(),
              .gxUsage = gx_usage_future(),
              .gxSampleCount = 1,
              .inspectorOpen = false,
          },
      // What the recorder picks for these limits at one sample
      .sceneDepthSource = limits.features.core ? DepthSource::Direct : DepthSource::Unavailable,
  };
}

std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_tier_p() {
  std::array<wgpu::TextureFormat, GxResourceCount> formats{};
  for (size_t i = 0; i < GxResourceCount; ++i) {
    formats[i] = gx_fixed_format(static_cast<GxResource>(i));
  }
  formats[static_cast<size_t>(GxResource::Color)] = wgpu::TextureFormat::BGRA8Unorm;
  formats[static_cast<size_t>(GxResource::Depth)] = wgpu::TextureFormat::Depth32Float;
  return formats;
}

std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_tier_b() {
  auto formats = gx_formats_tier_p();
  formats[static_cast<size_t>(GxResource::Surface)] = wgpu::TextureFormat::Undefined;
  formats[static_cast<size_t>(GxResource::DebugId)] = wgpu::TextureFormat::Undefined;
  return formats;
}

std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_legacy() {
  std::array<wgpu::TextureFormat, GxResourceCount> formats{};
  formats[static_cast<size_t>(GxResource::Color)] = wgpu::TextureFormat::BGRA8Unorm;
  formats[static_cast<size_t>(GxResource::Depth)] = wgpu::TextureFormat::Depth32Float;
  return formats;
}

std::array<wgpu::TextureUsage, GxResourceCount> gx_usage_future() {
  std::array<wgpu::TextureUsage, GxResourceCount> usage{};
  for (auto& entry : usage) {
    entry = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  }
  auto& color = usage[static_cast<size_t>(GxResource::Color)];
  color = color | wgpu::TextureUsage::CopyDst;
  return usage;
}

std::array<wgpu::TextureUsage, GxResourceCount> gx_usage_today() {
  std::array<wgpu::TextureUsage, GxResourceCount> usage{};
  usage[static_cast<size_t>(GxResource::Color)] = wgpu::TextureUsage::RenderAttachment |
                                                  wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
                                                  wgpu::TextureUsage::CopyDst;
  usage[static_cast<size_t>(GxResource::Depth)] =
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
  usage[static_cast<size_t>(GxResource::Normal)] =
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  return usage;
}
} // namespace aurora::render::detail::planned
