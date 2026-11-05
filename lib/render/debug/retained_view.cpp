#include "retained_view.hpp"

#include "gbuf_view.hpp"

#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../scene/affine.hpp"
#include "../scene/identity.hpp"
#include "../scene/meshlets.hpp"
#include "../scene/streamout_layout.hpp"
#include "../../gx/gx.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace aurora::render::debug::retained_view {
namespace {
namespace layout = scene::streamout;
namespace class_bit = scene::retained::class_bit;

constexpr uint64_t LayoutKey = 0x726574766965770Aull;
constexpr uint32_t LocalOffset = sizeof(uint64_t);
constexpr uint32_t ClusterCorners = scene::meshlets::MaxTriangles * 3;
constexpr float MarkerScale = 0.04f;
constexpr double ExtentMargin = 1.05;

enum Ref : uint8_t { Vertices, Indices, Instances, Frame, Clusters, ViewInstances, ViewPairs, RefCount };
static_assert(RefCount == StorageBindings);

struct Local {
  std::array<uint16_t, RefCount> refs{};
  uint16_t color = UINT16_MAX;
};
static_assert(LocalOffset + sizeof(Local) <= detail::PassLocalSize);

struct Entry {
  std::string_view vertex;
  std::string_view fragment;
  bool depth;
};

constexpr std::array<Entry, 3> Entries{{
    {"vs_instance", "fs_view", true},
    {"vs_cluster", "fs_view", true},
    {"vs_marker", "fs_marker", false},
}};

wgpu::BindGroupLayout g_groupLayout;
wgpu::PipelineLayout g_pipelineLayout;
std::atomic<float> g_extent{0.f};

struct Published {
  std::atomic_uint32_t instances{0};
  std::atomic_uint32_t clusters{0};
  std::atomic_uint32_t overInstances{0};
  std::atomic_uint32_t overPairs{0};
  std::atomic<float> extent{0.f};
};

Published g_shown;

// Render worker only
struct WorkerCache {
  std::array<wgpu::Buffer, RefCount> buffers;
  std::array<uint64_t, RefCount> offsets{};
  std::array<uint64_t, RefCount> sizes{};
  wgpu::BindGroup group;
  std::vector<uint64_t> seen;
  std::vector<uint32_t> words;
};

WorkerCache g_worker;

uint32_t f32_bits(float value) noexcept {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void put_vec4(std::span<uint32_t> out, uint32_t word, const std::array<double, 4>& v) noexcept {
  for (uint32_t i = 0; i < 4; ++i) {
    out[word + i] = f32_bits(static_cast<float>(v[i]));
  }
}

void perspective_header(std::span<uint32_t> out, const detail::SceneHints& hints) noexcept {
  std::array<float, 16> proj{};
  std::memcpy(proj.data(), &hints.mainCamera.gxProj, sizeof(proj));
  for (uint32_t col = 0; col < 4; ++col) {
    proj[8 + col] = gx::UseReversedZ ? -proj[8 + col] : proj[8 + col] + proj[12 + col];
  }
  for (uint32_t i = 0; i < 16; ++i) {
    out[header::Proj + i] = f32_bits(proj[i]);
  }
  std::array<float, 12> view{};
  std::memcpy(view.data(), &hints.mainCamera.view, sizeof(view));
  for (uint32_t row = 0; row < 3; ++row) {
    put_vec4(out, header::View + row * 4, {view[row * 4], view[row * 4 + 1], view[row * 4 + 2], 0.0});
  }
}

void coverage_header(std::span<uint32_t> out, const scene::Affine& invView, float extent, double aspect) noexcept {
  const double sx = 1.0 / (extent * std::max(aspect, 1.0));
  const double sy = 1.0 / (extent / std::min(aspect, 1.0));
  put_vec4(out, header::Proj, {sx, 0.0, 0.0, 0.0});
  put_vec4(out, header::Proj + 4, {0.0, sy, 0.0, 0.0});
  put_vec4(out, header::Proj + 8, {0.0, 0.0, 0.5 / MaxCoverageExtent, 0.5});
  put_vec4(out, header::Proj + 12, {0.0, 0.0, 0.0, 1.0});
  put_vec4(out, header::View, {1.0, 0.0, 0.0, 0.0});
  put_vec4(out, header::View + 4, {0.0, 0.0, -1.0, 0.0});
  put_vec4(out, header::View + 8, {0.0, 1.0, 0.0, 0.0});
  double fx = -invView.m[0][2];
  double fz = -invView.m[2][2];
  const double length = std::hypot(fx, fz);
  fx = length > 1e-6 ? fx / length : 0.0;
  fz = length > 1e-6 ? fz / length : -1.0;
  const double size = extent * MarkerScale;
  put_vec4(out, header::Marker, {fx * size * 1.5, 0.0, fz * size * 1.5, 0.0});
  put_vec4(out, header::Marker + 4, {-fz * size * 0.6 - fx * size, 0.0, fx * size * 0.6 - fz * size, 0.0});
  put_vec4(out, header::Marker + 8, {fz * size * 0.6 - fx * size, 0.0, -fx * size * 0.6 - fz * size, 0.0});
}

bool same_views(const std::array<BufferView, RefCount>& views) noexcept {
  if (!g_worker.group) {
    return false;
  }
  for (size_t i = 0; i < RefCount; ++i) {
    if (g_worker.buffers[i].Get() != views[i].buffer.Get() || g_worker.offsets[i] != views[i].offset ||
        g_worker.sizes[i] != views[i].size) {
      return false;
    }
  }
  return true;
}

const wgpu::BindGroup& bind_group(const std::array<BufferView, RefCount>& views) {
  if (same_views(views)) {
    return g_worker.group;
  }
  std::array<wgpu::BindGroupEntry, RefCount> entries{};
  for (uint32_t i = 0; i < RefCount; ++i) {
    entries[i] = {.binding = i, .buffer = views[i].buffer, .offset = views[i].offset, .size = views[i].size};
    g_worker.buffers[i] = views[i].buffer;
    g_worker.offsets[i] = views[i].offset;
    g_worker.sizes[i] = views[i].size;
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = "debug.retained_view",
      .layout = g_groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_worker.group = webgpu::g_device.CreateBindGroup(&descriptor);
  return g_worker.group;
}

bool vertex_storage(const Caps& caps) noexcept {
  return stage_binding_limits(caps, ShaderStage::Vertex).storageBuffers >= StorageBindings;
}

void mark_seen(const scene::retained::Frame& frame, std::vector<uint64_t>& seen) {
  seen.assign(scene::retained::MaxSlots / 64, 0);
  for (const scene::retained::Entry& entry : frame.entries) {
    if (entry.slot < scene::retained::MaxSlots && (entry.flags & scene::retained::entry_flag::CaptureOnly) == 0) {
      seen[entry.slot / 64] |= uint64_t{1} << (entry.slot % 64);
    }
  }
}
} // namespace

RetainedClass classify(const scene::streamout::LiveInstance& instance) noexcept {
  if ((instance.flags & layout::instance_flag::CpuDeformed) != 0) {
    return RetainedClass::Deformed;
  }
  if ((instance.flags & layout::instance_flag::Palette) != 0) {
    return RetainedClass::Palette;
  }
  if ((instance.flags & layout::instance_flag::GpuArrays) != 0) {
    return RetainedClass::Other;
  }
  if ((instance.classBits & (class_bit::Static | class_bit::Immobile)) != 0) {
    return instance.clusterCount != 0 ? RetainedClass::StaticClustered : RetainedClass::StaticUnclustered;
  }
  return (instance.classBits & class_bit::Dynamic) != 0 ? RetainedClass::Dynamic : RetainedClass::Other;
}

uint32_t pack_color(Rgb8 color, bool seen) noexcept {
  const uint32_t shift = seen ? 0 : 1;
  return uint32_t{color.r} >> shift | (uint32_t{color.g} >> shift) << 8 | (uint32_t{color.b} >> shift) << 16 |
         0xFF000000u;
}

Packed pack(std::span<const scene::streamout::LiveInstance> live, std::span<const uint64_t> seen, RetainedMode mode,
            const Limits& limits) {
  Packed out;
  out.instances.reserve(std::min<size_t>(live.size(), limits.instances));
  for (const scene::streamout::LiveInstance& instance : live) {
    if (out.instances.size() >= limits.instances) {
      ++out.overInstances;
      continue;
    }
    const auto index = static_cast<uint32_t>(out.instances.size());
    const bool late = (instance.flags & layout::instance_flag::Late) != 0;
    // Late draws arrive after the view's stage, so their instances cannot be judged and keep full intensity
    const bool drawn =
        late || (instance.slot / 64 < seen.size() && ((seen[instance.slot / 64] >> (instance.slot % 64)) & 1u) != 0);
    ViewInstance view{
        .slot = instance.slot,
        .color = pack_color(retained_class_color(classify(instance)), drawn),
        .indexCount = instance.indexCount,
    };
    if (mode == RetainedMode::Clusters && instance.clusterCount != 0) {
      if (out.pairs.size() + instance.clusterCount <= limits.pairs) {
        view.flags |= instance_flag::Clustered;
        for (uint32_t k = 0; k < instance.clusterCount; ++k) {
          out.pairs.push_back({index, instance.clusterFirst + k});
        }
      } else {
        out.overPairs += instance.clusterCount;
      }
    }
    out.instances.push_back(view);
  }
  return out;
}

float fit_extent(std::span<const scene::streamout::LiveInstance> live, const std::array<double, 3>& eye) noexcept {
  double farthest = 0.0;
  for (const scene::streamout::LiveInstance& instance : live) {
    if ((instance.classBits & (class_bit::Static | class_bit::Immobile)) != 0) {
      farthest = std::max({farthest, std::abs(instance.origin[0] - eye[0]), std::abs(instance.origin[2] - eye[2])});
    }
  }
  return static_cast<float>(std::clamp(farthest * ExtentMargin, double{MinCoverageExtent}, double{MaxCoverageExtent}));
}

bool supported(const Caps& caps) noexcept { return scene::retained::available(caps) && vertex_storage(caps); }

void initialize() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device || !caps().core || !vertex_storage(caps()) || g_groupLayout) {
    return;
  }
  std::array<wgpu::BindGroupLayoutEntry, RefCount> entries{};
  for (uint32_t i = 0; i < RefCount; ++i) {
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Vertex,
        .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
    };
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{
      .label = "debug.retained_view",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "debug.retained_view",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_groupLayout,
  };
  g_pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
}

void shutdown() noexcept {
  g_worker = {};
  g_pipelineLayout = {};
  g_groupLayout = {};
  g_extent.store(0.f, std::memory_order_relaxed);
}

bool declarable(const detail::PassSetup& setup) noexcept {
  return setup.env != nullptr && setup.env->retained && setup.caps != nullptr && supported(*setup.caps) &&
         g_pipelineLayout && scene::streamout::pool_handle(layout::binding::Vertices).id != 0 &&
         scene::meshlets::clusters_handle().id != 0;
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GbufView&) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.manual_bindings();
  Local local;
  const auto pool = [&](std::string_view name, uint32_t binding) {
    return builder.import_persistent(name, scene::streamout::pool_handle(binding), Access::StorageRead).index;
  };
  local.refs[Vertices] = pool("scene.vertices", layout::binding::Vertices);
  local.refs[Indices] = pool("scene.indices", layout::binding::Indices);
  local.refs[Instances] = pool("scene.instances", layout::binding::Instances);
  local.refs[Frame] = pool("scene.frame", layout::binding::Frame);
  local.refs[Clusters] =
      builder.import_persistent("scene.clusters", scene::meshlets::clusters_handle(), Access::StorageRead).index;
  local.refs[ViewInstances] =
      builder
          .create_buffer(InstancesResource, {.size = InstancesBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                         Access::StorageWrite)
          .index;
  local.refs[ViewPairs] =
      builder
          .create_buffer(PairsResource, {.size = PairsBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                         Access::StorageWrite)
          .index;
  local.color =
      builder
          .create(gbuf_view::SnapshotResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                  Access::ColorWrite)
          .index;
  builder.create(DepthResource, {.format = wgpu::TextureFormat::Depth32Float, .sizeClass = SizeClass::Efb},
                 Access::DepthWrite);
  std::memcpy(builder.local().data() + LocalOffset, &local, sizeof(local));
}

bool enabled(const detail::StageData& stage) noexcept {
  return stage.markerSeen && stage.sceneHints != nullptr && stage.sceneHints->has(detail::SceneHasMainCamera) &&
         stage.retained != nullptr && stage.retained->plan != nullptr && stage.retained->plan->live != nullptr;
}

shaders::RenderPipelineDesc pipeline_desc(uint32_t index) {
  const Entry& entry = Entries[std::min<size_t>(index, Entries.size() - 1)];
  shaders::RenderPipelineDesc desc{
      .label = gbuf_view::SnapshotPass,
      .shader = Shader,
      .vertexEntry = entry.vertex,
      .fragmentEntry = entry.fragment,
      .layout = g_pipelineLayout,
      .layoutKey = LayoutKey,
      .colorCount = 1,
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .depthWrite = entry.depth,
      .depthCompare = entry.depth ? (gx::UseReversedZ ? wgpu::CompareFunction::Greater : wgpu::CompareFunction::Less)
                                  : wgpu::CompareFunction::Always,
  };
  desc.colors[0] = {.format = wgpu::TextureFormat::RGBA8Unorm};
  return desc;
}

void pipelines(const detail::PipelineSetup&, const GbufView&, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  for (uint32_t i = 0; i < Entries.size(); ++i) {
    ids[i] = g_pipelineLayout ? shaders::request_render(pipeline_desc(i)).value : shaders::NeverReady.value;
  }
}

void encode(detail::PassContext& ctx, const GbufView& row) {
  const detail::StageData& stage = ctx.stage_data();
  if (!enabled(stage) || !g_groupLayout) {
    return;
  }
  std::array<wgpu::RenderPipeline, Entries.size()> pipes;
  for (uint32_t i = 0; i < Entries.size(); ++i) {
    pipes[i] = shaders::render_pipeline(ctx.pipeline(i));
    if (!pipes[i]) {
      return;
    }
  }
  Local local;
  std::memcpy(&local, ctx.local().data() + LocalOffset, sizeof(local));
  std::array<BufferView, RefCount> views;
  for (uint32_t i = 0; i < RefCount; ++i) {
    views[i] = ctx.buffer(BufRef{local.refs[i]});
    if (!views[i].buffer) {
      return;
    }
  }
  const RetainedMode mode = retained_mode(row);
  const std::vector<scene::streamout::LiveInstance>& live = *stage.retained->plan->live;
  mark_seen(*stage.retained, g_worker.seen);
  const Packed packed = pack(live, g_worker.seen, mode);
  const detail::SceneHints& hints = *stage.sceneHints;

  std::vector<uint32_t>& words = g_worker.words;
  words.assign(HeaderWords + packed.instances.size() * InstanceWords, 0);
  const std::span<uint32_t> head{words.data(), HeaderWords};
  float extent = 0.f;
  if (mode == RetainedMode::Coverage) {
    const scene::Affine invView = scene::inverse(scene::to_affine(hints.mainCamera.view));
    const float chosen = g_extent.load(std::memory_order_relaxed);
    extent = chosen > 0.f ? std::clamp(chosen, MinChosenExtent, MaxCoverageExtent)
                          : fit_extent(live, {invView.m[0][3], invView.m[1][3], invView.m[2][3]});
    const wgpu::Extent3D size = ctx.extent(TexRef{local.color});
    const double aspect = size.height != 0 ? static_cast<double>(size.width) / size.height : 1.0;
    coverage_header(head, invView, extent, aspect);
    head[header::Flags] = header::Ortho;
  } else {
    perspective_header(head, hints);
  }
  head[header::Mode] = static_cast<uint32_t>(mode);
  head[header::Instances] = static_cast<uint32_t>(packed.instances.size());
  head[header::Pairs] = static_cast<uint32_t>(packed.pairs.size());
  std::memcpy(words.data() + HeaderWords, packed.instances.data(), packed.instances.size() * sizeof(ViewInstance));

  // The encoder orders these writes inside the frame; queue writes would land at frame start over pooled transients
  ctx.encoder().WriteBuffer(views[ViewInstances].buffer, views[ViewInstances].offset,
                            reinterpret_cast<const uint8_t*>(words.data()), words.size() * sizeof(uint32_t));
  if (!packed.pairs.empty()) {
    ctx.encoder().WriteBuffer(views[ViewPairs].buffer, views[ViewPairs].offset,
                              reinterpret_cast<const uint8_t*>(packed.pairs.data()),
                              packed.pairs.size() * sizeof(packed.pairs[0]));
  }
  const wgpu::BindGroup& group = bind_group(views);
  const wgpu::RenderPassEncoder pass = ctx.begin_render();
  if (!pass) {
    return;
  }
  // Retained views draw the pools from the latched camera, so the GX draws stay untouched and merges keep their shape
  pass.SetBindGroup(0, group);
  pass.SetPipeline(pipes[0]);
  for (uint32_t i = 0; i < packed.instances.size(); ++i) {
    const ViewInstance& view = packed.instances[i];
    if ((view.flags & instance_flag::Clustered) == 0 && view.indexCount != 0) {
      pass.Draw(view.indexCount, 1, 0, i);
    }
  }
  if (!packed.pairs.empty()) {
    pass.SetPipeline(pipes[1]);
    pass.Draw(ClusterCorners, static_cast<uint32_t>(packed.pairs.size()), 0, 0);
  }
  if (mode == RetainedMode::Coverage) {
    pass.SetPipeline(pipes[2]);
    pass.Draw(3);
  }
  pass.End();
  g_shown.instances.store(static_cast<uint32_t>(packed.instances.size()), std::memory_order_relaxed);
  g_shown.clusters.store(static_cast<uint32_t>(packed.pairs.size()), std::memory_order_relaxed);
  g_shown.overInstances.store(packed.overInstances, std::memory_order_relaxed);
  g_shown.overPairs.store(packed.overPairs, std::memory_order_relaxed);
  g_shown.extent.store(extent, std::memory_order_relaxed);
}

void set_coverage_extent(float extent) noexcept {
  g_extent.store(std::isfinite(extent) ? std::max(extent, 0.f) : 0.f, std::memory_order_relaxed);
}

float coverage_extent() noexcept { return g_extent.load(std::memory_order_relaxed); }

Shown shown() noexcept {
  return {
      .instances = g_shown.instances.load(std::memory_order_relaxed),
      .clusters = g_shown.clusters.load(std::memory_order_relaxed),
      .overInstances = g_shown.overInstances.load(std::memory_order_relaxed),
      .overPairs = g_shown.overPairs.load(std::memory_order_relaxed),
      .extent = g_shown.extent.load(std::memory_order_relaxed),
  };
}
} // namespace aurora::render::debug::retained_view
