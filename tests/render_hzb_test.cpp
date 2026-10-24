#include <gtest/gtest.h>

#include "render_hzb_reference.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "render/frame_state.hpp"
#include "render/geometry/hzb.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/jobs.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/scene_depth.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace ref = aurora::test::hzb;
using wgpu::TextureFormat;
using wgpu::TextureSampleType;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float DepthStep = 0x1p-24f;
constexpr uint32_t Seed = 0x48A5B11Du;

std::vector<float> random_depth(std::mt19937& rng, uint32_t width, uint32_t height) {
  std::uniform_int_distribution<uint32_t> grid(0, (1u << 24) - 1);
  std::vector<float> out(size_t{width} * height);
  for (float& value : out) {
    value = static_cast<float>(grid(rng)) * DepthStep;
  }
  return out;
}

uint32_t uniform(std::mt19937& rng, uint32_t lo, uint32_t hi) {
  return std::uniform_int_distribution<uint32_t>(lo, hi)(rng);
}

float uniform(std::mt19937& rng, float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

ref::Rect random_rect(std::mt19937& rng, uint32_t width, uint32_t height) {
  const auto span = [&](uint32_t size) {
    uint32_t extent = 1;
    switch (uniform(rng, 0u, 3u)) {
    case 0:
      break;
    case 1:
      extent = uniform(rng, 1u, std::min(size, 4u));
      break;
    case 2:
      extent = uniform(rng, 1u, std::max(1u, size / 4));
      break;
    default:
      extent = uniform(rng, 1u, size);
      break;
    }
    const uint32_t first = uniform(rng, 0u, size - extent);
    return std::pair{first, first + extent - 1};
  };
  const auto [x0, x1] = span(width);
  const auto [y0, y1] = span(height);
  return {x0, y0, x1, y1};
}

float pick_nearest(std::mt19937& rng, float exact) {
  switch (uniform(rng, 0u, 4u)) {
  case 0:
    return exact;
  case 1:
    return std::nextafter(exact, -1.f);
  case 2:
    return std::nextafter(exact, 2.f);
  case 3:
    return exact - DepthStep * static_cast<float>(uniform(rng, 0u, 1024u));
  default:
    return uniform(rng, 0.f, 1.f);
  }
}

uint32_t pyramid_levels(uint32_t width, uint32_t height) {
  return static_cast<uint32_t>(std::bit_width(std::max((width + 1) / 2, (height + 1) / 2)));
}

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> texel_spans(uint32_t pixels, uint32_t mip0, uint32_t levels) {
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> out;
  uint32_t inSize = pixels;
  for (uint32_t level = 0; level < levels; ++level) {
    const uint32_t size = hzb::level_size(mip0, level);
    std::vector<std::pair<uint32_t, uint32_t>> spans;
    for (uint32_t texel = 0; texel < size; ++texel) {
      const hzb::Footprint f = hzb::footprint(texel, size, inSize);
      if (level == 0) {
        spans.emplace_back(f.first, f.last);
      } else {
        spans.emplace_back(out.back()[f.first].first, out.back()[f.last].second);
      }
    }
    out.push_back(std::move(spans));
    inSize = size;
  }
  return out;
}

TEST(HzbRules, FootprintsPartitionSource) {
  for (uint32_t in = 1; in <= 64; ++in) {
    for (const uint32_t out : {(in + 1) / 2, std::max(1u, in / 2)}) {
      uint32_t next = 0;
      for (uint32_t texel = 0; texel < out; ++texel) {
        const hzb::Footprint f = hzb::footprint(texel, out, in);
        ASSERT_EQ(f.first, next) << in << " -> " << out << " texel " << texel;
        ASSERT_TRUE(hzb::reducible(f, f)) << in << " -> " << out << " texel " << texel;
        next = f.last + 1;
      }
      EXPECT_EQ(next, in) << in << " -> " << out;
    }
  }
  EXPECT_FALSE(hzb::reducible({2, 1}, {0, 0}));
  EXPECT_FALSE(hzb::reducible({0, 3}, {0, 0}));
  EXPECT_FALSE(hzb::reducible(hzb::footprint(0, 1, 0), {0, 0}));
}

TEST(HzbRules, MappingFindsCoveringTexel) {
  for (uint32_t width = 1; width <= 40; ++width) {
    for (uint32_t height = 1; height <= 40; ++height) {
      const uint32_t levels = pyramid_levels(width, height);
      for (const auto [pixels, mip0] : {std::pair{width, (width + 1) / 2}, std::pair{height, (height + 1) / 2}}) {
        const auto spans = texel_spans(pixels, mip0, levels);
        for (uint32_t level = 0; level < levels; ++level) {
          ASSERT_EQ(spans[level].front().first, 0u);
          ASSERT_EQ(spans[level].back().second, pixels - 1);
          for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
            const uint32_t texel = hzb::texel_of(pixel, level, hzb::level_size(mip0, level));
            const auto [first, last] = spans[level][texel];
            ASSERT_TRUE(pixel >= first && pixel <= last)
                << width << "x" << height << " level " << level << " pixel " << pixel << " texel " << texel;
          }
        }
      }
    }
  }
}

TEST(HzbRules, RangeMinMatchesBruteForce) {
  std::mt19937 rng{Seed};
  for (uint32_t width = 1; width <= 33; ++width) {
    for (uint32_t height = 1; height <= 33; ++height) {
      const std::vector<float> depth = random_depth(rng, width, height);
      const ref::RangeMin range{depth, width, height};
      for (int i = 0; i < 16; ++i) {
        const ref::Rect rect = random_rect(rng, width, height);
        ASSERT_EQ(range.min(rect), ref::brute_min(depth, width, rect)) << width << "x" << height;
      }
    }
  }
  for (const auto [width, height] : {std::pair{1280u, 720u}, std::pair{1279u, 719u}}) {
    const std::vector<float> depth = random_depth(rng, width, height);
    const ref::RangeMin range{depth, width, height};
    for (int i = 0; i < 200; ++i) {
      const ref::Rect rect = random_rect(rng, width, height);
      ASSERT_EQ(range.min(rect), ref::brute_min(depth, width, rect)) << width << "x" << height;
    }
  }
}

TEST(HzbRules, QueryIsConservative) {
  std::mt19937 rng{Seed + 1};
  const auto check = [&](uint32_t width, uint32_t height, uint32_t rects) {
    const std::vector<float> depth = random_depth(rng, width, height);
    const ref::CpuPyramid pyramid = ref::build_pyramid(depth, width, height);
    const ref::RangeMin range{depth, width, height};
    uint32_t culled = 0;
    for (uint32_t i = 0; i < rects; ++i) {
      const ref::Rect rect = random_rect(rng, width, height);
      const float exact = range.min(rect);
      const float nearest = pick_nearest(rng, exact);
      if (!ref::occluded_rect(pyramid, rect, nearest)) {
        continue;
      }
      ++culled;
      if (!(nearest < exact)) {
        ADD_FAILURE() << width << "x" << height << " rect " << rect.x0 << "," << rect.y0 << "-" << rect.x1 << ","
                      << rect.y1 << " culled at " << nearest << " over " << exact;
        return culled;
      }
    }
    return culled;
  };
  uint32_t culled = 0;
  for (uint32_t width = 1; width <= 33; ++width) {
    for (uint32_t height = 1; height <= 33; ++height) {
      culled += check(width, height, 500);
    }
  }
  culled += check(1280, 720, 10000);
  culled += check(1279, 719, 10000);
  EXPECT_GT(culled, 0u);
  EXPECT_FALSE(ref::occluded_rect(ref::build_pyramid(std::vector<float>(4, 1.f), 2, 2), {1, 0, 0, 0}, 0.f));

  const std::vector<float> nearWall(64 * 64, 1.f);
  const ref::CpuPyramid wall = ref::build_pyramid(nearWall, 64, 64);
  const ref::Viewport full{.width = 64.f, .height = 64.f};
  const auto bounds_of = [](std::initializer_list<std::array<float, 4>> corners) {
    ref::Bounds bounds;
    for (const auto& clip : corners) {
      bounds = ref::bounds_add(bounds, clip);
    }
    return bounds;
  };
  const ref::Bounds behind = bounds_of({{-0.1f, -0.1f, 0.5f, 1.f}, {0.1f, 0.1f, 0.5f, 1.f}});
  EXPECT_TRUE(ref::occluded_bounds(wall, behind, full, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(wall, ref::Bounds{}, full, 64, 64));
  EXPECT_FALSE(
      ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 0.5f, 1.f}, {0.1f, 0.1f, 0.f, 1e-6f}}), full, 64, 64));
  EXPECT_FALSE(
      ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 0.5f, 1.f}, {0.1f, 0.1f, 0.f, -1.f}}), full, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 0.5f, 1.f}, {0.1f, 0.1f, 0.f, std::nanf("")}}),
                                    full, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 0.5f, 1.f}, {std::nanf(""), 0.1f, 0.f, 1.f}}), full,
                                    64, 64));
  EXPECT_FALSE(ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 1.5f, 1.f}}), full, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(wall, bounds_of({{-0.1f, -0.1f, 1.f, 1.f}}), full, 64, 64));

  const std::vector<float> rangeWall(64 * 64, 0.75f);
  const ref::CpuPyramid ranged = ref::build_pyramid(rangeWall, 64, 64);
  ref::Viewport sub = full;
  sub.minDepth = 0.25f;
  sub.maxDepth = 0.75f;
  EXPECT_TRUE(ref::occluded_bounds(ranged, bounds_of({{0.f, 0.f, 0.99f, 1.f}}), sub, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(ranged, bounds_of({{0.f, 0.f, 1.f, 1.f}}), sub, 64, 64));
  EXPECT_FALSE(ref::occluded_bounds(ranged, bounds_of({{0.f, 0.f, 0.99f, 1.f}}), full, 64, 64));
}

TEST(HzbRules, QueryIsNotVacuous) {
  std::mt19937 rng{Seed + 2};
  for (const auto [width, height] :
       {std::pair{64u, 64u}, std::pair{61u, 37u}, std::pair{1280u, 720u}, std::pair{1279u, 719u}}) {
    std::vector<float> depth(size_t{width} * height, 0.9f);
    const ref::CpuPyramid wall = ref::build_pyramid(depth, width, height);
    uint32_t culled = 0;
    constexpr uint32_t Rects = 2000;
    for (uint32_t i = 0; i < Rects; ++i) {
      culled += ref::occluded_rect(wall, random_rect(rng, width, height), 0.5f) ? 1 : 0;
    }
    EXPECT_GE(culled, Rects * 95 / 100) << width << "x" << height;
    for (uint32_t y = 0; y < height; ++y) {
      depth[size_t{y} * width + width - 1] = 0.f;
    }
    const ref::CpuPyramid edge = ref::build_pyramid(depth, width, height);
    for (uint32_t i = 0; i < Rects; ++i) {
      ref::Rect rect = random_rect(rng, width, height);
      rect.x1 = width - 1;
      ASSERT_FALSE(ref::occluded_rect(edge, rect, 0.5f)) << width << "x" << height << " x0 " << rect.x0;
    }
  }
}

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  uint16_t order = 1;
  std::function<void(InternalPassBuilder&)> declare;
  std::function<void(PassContext&)> encode;
  std::function<void(const PipelineSetup&, std::span<uint32_t, MaxPassPipelines>)> pipelines;
  mutable int calls = 0;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  static_cast<const TestPass*>(state)->declare(builder);
}

void encode_test_pass(PassContext& ctx, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  ++pass.calls;
  if (pass.encode) {
    pass.encode(ctx);
  }
}

void pipelines_test_pass(const PipelineSetup& setup, const void* state, std::span<uint32_t, MaxPassPipelines> ids) {
  static_cast<const TestPass*>(state)->pipelines(setup, ids);
}

PassDesc desc_of(const TestPass& pass) {
  return {
      .name = pass.name,
      .stage = pass.stage,
      .kind = pass.kind,
      .order = pass.order,
      .declare = declare_test_pass,
      .encode = encode_test_pass,
      .pipelines = pass.pipelines ? pipelines_test_pass : nullptr,
      .state = &pass,
  };
}

void declare_r32_build(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Geometry);
  hzb::declare_build(builder, "t.src", hzb::Resource);
}

PassDesc r32_build_desc() {
  PassDesc desc = hzb::opaque_pass_desc();
  desc.name = "t.build";
  desc.order = 1;
  desc.declare = declare_r32_build;
  return desc;
}

struct TestParams {
  ref::Mat4 clipFromWorld{};
  std::array<float, 4> viewport{};
  std::array<float, 2> depthRange{};
  std::array<uint32_t, 2> efbSize{};
  uint32_t count = 0;
  std::array<uint32_t, 3> pad{};
};
static_assert(sizeof(TestParams) == 112);

struct RectQuery {
  std::array<uint32_t, 4> rect{};
  float nearest = 0.f;
  std::array<float, 3> pad{};
};
static_assert(sizeof(RectQuery) == 32);

constexpr std::string_view ConsumerHeader = R"(//#import aurora/hzb

struct HzbTestParams {
  clip_from_world: mat4x4f,
  viewport: vec4f,
  depth_range: vec2f,
  efb_size: vec2u,
  count: u32,
}

@group(1) @binding(0) var hzb: texture_2d<f32>;
@group(2) @binding(0) var<uniform> params: HzbTestParams;
)";

constexpr std::string_view CullBody = R"(
struct HzbTestBox {
  bmin: vec4f,
  bmax: vec4f,
}

@group(2) @binding(1) var<storage, read> boxes: array<HzbTestBox>;
@group(2) @binding(2) var<storage, read_write> verdicts: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.count) {
    return;
  }
  let item = boxes[id.x];
  let bounds = aurora_hzb_bounds_aabb(params.clip_from_world, item.bmin.xyz, item.bmax.xyz);
  let culled = aurora_hzb_occluded(hzb, bounds, params.viewport, params.depth_range, params.efb_size);
  verdicts[id.x] = select(0u, 1u, culled);
}
)";

constexpr std::string_view RectBody = R"(
struct HzbTestRect {
  rect: vec4u,
  nearest: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(2) @binding(1) var<storage, read> rects: array<HzbTestRect>;
@group(2) @binding(2) var<storage, read_write> verdicts: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.count) {
    return;
  }
  let query = rects[id.x];
  verdicts[id.x] = select(0u, 1u, aurora_hzb_occluded_rect(hzb, query.rect.xy, query.rect.zw, query.nearest));
}
)";

constexpr std::string_view DepthFillWgsl = R"(
@vertex
fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f {
  return vec4f(point, 1.0);
}
)";

constexpr std::string_view GeometryWgsl = R"(
@group(0) @binding(0) var<uniform> clip_from_world: mat4x4f;

@vertex
fn vs(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return clip_from_world * vec4f(position, 1.0);
}
)";

class HzbDeviceTest : public ::testing::Test {
protected:
  bool open(const test::TestDeviceOptions& options) {
    m_options = options;
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      return false;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    gpu_heap::shutdown();
    reset_module_state();
    jobs::initialize(1);
    shaders::initialize();
    register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
    EXPECT_TRUE(shaders::register_source("test/hzb_cull", std::string{ConsumerHeader} + std::string{CullBody}));
    EXPECT_TRUE(shaders::register_source("test/hzb_rects", std::string{ConsumerHeader} + std::string{RectBody}));
    const std::array entries{
        wgpu::BindGroupLayoutEntry{.binding = 0,
                                   .visibility = wgpu::ShaderStage::Compute,
                                   .buffer = {.type = wgpu::BufferBindingType::Uniform}},
        wgpu::BindGroupLayoutEntry{.binding = 1,
                                   .visibility = wgpu::ShaderStage::Compute,
                                   .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
        wgpu::BindGroupLayoutEntry{.binding = 2,
                                   .visibility = wgpu::ShaderStage::Compute,
                                   .buffer = {.type = wgpu::BufferBindingType::Storage}},
    };
    const wgpu::BindGroupLayoutDescriptor layout{.entryCount = entries.size(), .entries = entries.data()};
    m_extraLayout = m_device->device.CreateBindGroupLayout(&layout);
    m_started = true;
    return true;
  }

  void TearDown() override {
    if (m_started) {
      hzb::shutdown();
      hzb::testing::reset();
      m_extraGroup = {};
      m_extraLayout = {};
      shaders::shutdown();
      jobs::shutdown();
      reset_module_state();
      gpu_heap::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  void use_globals(const test::TestDeviceOptions& options) {
    hzb::testing::reset();
    m_globals.reset();
    m_options = options;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
  }

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(std::span<const PassDesc> passes,
                                                       DepthSource source = DepthSource::Direct) const {
    GraphInputs inputs;
    inputs.env = gather_graph_environment();
    inputs.caps = &caps();
    inputs.sceneDepthSource = source;
    inputs.passes = passes;
    std::shared_ptr<CompiledGraph> graph = compile_graph(inputs);
    prepare_graph_bindings(*graph, {.caps = &caps(), .env = &inputs.env, .sceneDepth = source});
    return graph;
  }

  [[nodiscard]] wgpu::Texture make_depth(uint32_t width, uint32_t height) const {
    const wgpu::TextureDescriptor descriptor{
        .usage =
            wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .size = {width, height, 1},
        .format = TextureFormat::Depth32Float,
    };
    return m_device->device.CreateTexture(&descriptor);
  }

  static std::shared_ptr<const SceneDepth> scene_depth(const wgpu::Texture& texture) {
    auto depth = std::make_shared<SceneDepth>();
    depth->source = DepthSource::Direct;
    depth->sampledTexture = texture;
    const wgpu::TextureViewDescriptor view{.aspect = wgpu::TextureAspect::DepthOnly};
    depth->sampled = texture.CreateView(&view);
    depth->sampledFormat = texture.GetFormat();
    depth->size = {texture.GetWidth(), texture.GetHeight(), 1};
    return depth;
  }

  [[nodiscard]] wgpu::Buffer make_buffer(wgpu::BufferUsage usage, uint64_t size, const void* data = nullptr) const {
    const wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst,
                                            .size = std::max<uint64_t>((size + 3) & ~uint64_t{3}, 16)};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    if (data != nullptr && size != 0) {
      m_device->queue.WriteBuffer(buffer, 0, data, (size + 3) & ~uint64_t{3});
    }
    return buffer;
  }

  StageData build_opaque(const FrameState& state, uint64_t frameId, std::shared_ptr<const SceneDepth> depth) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    StageData opaque{
        .stage = Stage::OpaqueDone,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = true,
        .frame = &state,
        .sceneDepth = std::move(depth),
    };
    build_stage_data(opaque);
    return opaque;
  }

  std::optional<std::string> run(const FrameState& state, uint64_t frameId, std::shared_ptr<const SceneDepth> depth,
                                 const std::function<void(const wgpu::CommandEncoder&)>& after = {}) const {
    const StageData opaque = build_opaque(state, frameId, std::move(depth));
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    if (opaque.any_enabled()) {
      run_stage(encoder, opaque);
    }
    if (after) {
      after(encoder);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    return test::pop_validation_error(*m_device);
  }

  static TestPass reader(std::string name = "t.cull") {
    return {
        .name = std::move(name),
        .declare =
            [](InternalPassBuilder& b) {
              b.after(phase::Geometry);
              hzb::use(b);
              b.side_effect();
            },
    };
  }

  TestPass shader_reader(std::string_view shader) {
    TestPass pass = reader();
    pass.pipelines = [this, shader](const PipelineSetup& setup, std::span<uint32_t, MaxPassPipelines> ids) {
      const std::array layouts{setup.frameLayout, setup.passLayout, m_extraLayout};
      const wgpu::PipelineLayoutDescriptor descriptor{.bindGroupLayoutCount = layouts.size(),
                                                      .bindGroupLayouts = layouts.data()};
      const wgpu::PipelineLayout layout = m_device->device.CreatePipelineLayout(&descriptor);
      ids[0] = shaders::request_compute(
                   {.label = shader, .shader = shader, .layout = layout, .layoutKey = setup.layoutHash ^ 0x485A42ull})
                   .value;
    };
    pass.encode = [this](PassContext& ctx) {
      const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
      ASSERT_TRUE(pipeline);
      const wgpu::ComputePassEncoder compute = ctx.begin_compute();
      compute.SetPipeline(pipeline);
      compute.SetBindGroup(2, m_extraGroup);
      compute.DispatchWorkgroups((m_count + 63) / 64);
      compute.End();
    };
    return pass;
  }

  static TestPass capture_reader(wgpu::Texture& out) {
    return {
        .name = "t.capture",
        .kind = PassKind::Copy,
        .order = 2,
        .declare =
            [](InternalPassBuilder& b) {
              b.after(phase::Geometry);
              b.use(hzb::Resource, Access::CopySrc);
              b.side_effect();
            },
        .encode = [&out](PassContext& ctx) { out = ctx.texture(TexRef{0}); },
    };
  }

  void bind_extra(const wgpu::Buffer& params, const wgpu::Buffer& input, const wgpu::Buffer& verdicts) {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = params},
        wgpu::BindGroupEntry{.binding = 1, .buffer = input},
        wgpu::BindGroupEntry{.binding = 2, .buffer = verdicts},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .layout = m_extraLayout, .entryCount = entries.size(), .entries = entries.data()};
    m_extraGroup = m_device->device.CreateBindGroup(&descriptor);
  }

  void wait_ready(const CompiledGraph& graph, std::string_view pass) const {
    const NodeId node = graph.find_node(pass);
    ASSERT_NE(node, InvalidNode) << pass;
    const uint32_t id = graph.nodes[node].pipelineIds[0];
    ASSERT_NE(id, 0u) << pass;
    ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << pass;
  }

  void fill_depth(const wgpu::Texture& texture, std::span<const float> values) const {
    const wgpu::Device& device = m_device->device;
    const std::string code{DepthFillWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    const wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    std::vector<float> points;
    points.reserve(values.size() * 3);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        points.push_back(static_cast<float>((x + 0.5) / width * 2.0 - 1.0));
        points.push_back(static_cast<float>(1.0 - (y + 0.5) / height * 2.0));
        points.push_back(values[size_t{y} * width + x]);
      }
    }
    const wgpu::VertexAttribute attribute{.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0};
    const wgpu::VertexBufferLayout vertexLayout{
        .stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = 12, .attributeCount = 1, .attributes = &attribute};
    const wgpu::DepthStencilState depthState{
        .format = TextureFormat::Depth32Float,
        .depthWriteEnabled = wgpu::OptionalBool::True,
        .depthCompare = wgpu::CompareFunction::Always,
    };
    const wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .vertex = {.module = module, .entryPoint = "vs", .bufferCount = 1, .buffers = &vertexLayout},
        .primitive = {.topology = wgpu::PrimitiveTopology::PointList},
        .depthStencil = &depthState,
    };
    const wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDescriptor);
    const wgpu::Buffer buffer = make_buffer(wgpu::BufferUsage::Vertex, points.size() * sizeof(float), points.data());
    const wgpu::RenderPassDepthStencilAttachment attachment{
        .view = texture.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.f,
    };
    const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    pass.SetVertexBuffer(0, buffer);
    pass.Draw(static_cast<uint32_t>(values.size()));
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const std::vector<float> stored = test::read_texture_mip(*m_device, texture, 0);
    ASSERT_EQ(stored.size(), values.size());
    size_t bad = 0;
    for (size_t i = 0; i < stored.size(); ++i) {
      bad += std::bit_cast<uint32_t>(stored[i]) != std::bit_cast<uint32_t>(values[i]) ? 1 : 0;
    }
    EXPECT_EQ(bad, 0u) << "the depth fill did not store every value exactly";
  }

  void expect_pyramid(const wgpu::Texture& texture, const ref::CpuPyramid& want, std::string_view what) const {
    ASSERT_TRUE(texture) << what;
    ASSERT_EQ(texture.GetMipLevelCount(), want.level_count()) << what;
    for (uint32_t level = 0; level < want.level_count(); ++level) {
      const std::vector<float> got = test::read_texture_mip(*m_device, texture, level);
      ASSERT_EQ(got.size(), want.levels[level].size()) << what << " level " << level;
      size_t bad = 0;
      size_t first = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (std::bit_cast<uint32_t>(got[i]) != std::bit_cast<uint32_t>(want.levels[level][i])) {
          first = bad == 0 ? i : first;
          ++bad;
        }
      }
      EXPECT_EQ(bad, 0u) << what << " level " << level << ": first mismatch at texel " << first << ", got "
                         << got[first] << ", want " << want.levels[level][first];
    }
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  wgpu::BindGroupLayout m_extraLayout;
  wgpu::BindGroup m_extraGroup;
  uint32_t m_count = 0;
};

class HzbGraph : public HzbDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
  }

  void encode_sizes() {
    for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{2u, 1u}, std::pair{1u, 3u}, std::pair{61u, 37u},
                                       std::pair{64u, 64u}, std::pair{1280u, 720u}}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      test::TestDeviceOptions options = m_options;
      options.efbSize = {width, height, 1};
      use_globals(options);
      const TestPass consumer = reader();
      const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
      const auto graph = compile(descs);
      wait_ready(*graph, hzb::OpaquePass);
      const FrameState state{.graph = graph};
      const auto depth = scene_depth(make_depth(width, height));
      const uint32_t levels = pyramid_levels(width, height);
      EXPECT_EQ(run(state, 1, depth), std::nullopt);
      EXPECT_EQ(hzb::testing::counters().dispatches, levels);
      EXPECT_EQ(hzb::testing::counters().groupsCreated, levels - 1);
      EXPECT_EQ(hzb::testing::counters().cacheEntries, 1u);
      EXPECT_EQ(run(state, 2, depth), std::nullopt);
      EXPECT_EQ(hzb::testing::counters().dispatches, 2 * levels);
      EXPECT_EQ(hzb::testing::counters().groupsCreated, levels - 1);
      on_worker_frame_begin(2 + hzb::RetainFrames, &state);
      EXPECT_EQ(hzb::testing::counters().cacheEntries, 1u);
      on_worker_frame_begin(3 + hzb::RetainFrames, &state);
      EXPECT_EQ(hzb::testing::counters().cacheEntries, 0u);
      EXPECT_EQ(consumer.calls, 2);
    }
  }
};

TEST_F(HzbGraph, CulledWithoutConsumer) {
  hzb::initialize();
  const GraphEnvironment env = gather_graph_environment();
  const std::shared_ptr<CompiledGraph> graph =
      compile_graph(Settings{}, caps(), env, {.sceneDepthSource = DepthSource::Direct});
  ASSERT_TRUE(graph);
  prepare_graph_bindings(*graph, {.caps = &caps(), .env = &env, .sceneDepth = DepthSource::Direct});
  const NodeId node = graph->find_node(hzb::OpaquePass);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, CullReason::NoConsumer);
  EXPECT_EQ(graph->nodes[node].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
  for (const StageProgram& program : graph->stages) {
    EXPECT_TRUE(program.passes.empty());
  }
  EXPECT_TRUE(graph->slots.slots.empty());
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1, scene_depth(make_depth(64, 64))), std::nullopt);
  EXPECT_EQ(hzb::testing::counters().dispatches, 0u);
  EXPECT_EQ(shaders::stats().ready + shaders::stats().pending + shaders::stats().failed, 0u);
}

TEST_F(HzbGraph, ConsumerMakesItLive) {
  const TestPass consumer = reader();
  const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
  const auto graph = compile(descs);
  const NodeId node = graph->find_node(hzb::OpaquePass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_NE(graph->nodes[node].pipelineIds[0], 0u);
  const BindingPlan& plan = graph->bindingPlans[node];
  ASSERT_EQ(plan.slots.size(), 2u);
  EXPECT_EQ(plan.slots[0].type, BindingType::SampledTexture);
  EXPECT_EQ(plan.slots[0].sampleType, TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(plan.slots[1].type, BindingType::WriteOnlyStorageTexture);
  EXPECT_EQ(plan.slots[1].storageFormat, TextureFormat::R32Float);
  const CompiledResource& pyramid = graph->resources[graph->find_resource(hzb::Resource)];
  const uint32_t width = (m_options.efbSize.width + 1) / 2;
  const uint32_t height = (m_options.efbSize.height + 1) / 2;
  EXPECT_EQ(pyramid.extent.width, width);
  EXPECT_EQ(pyramid.extent.height, height);
  EXPECT_EQ(pyramid.mipLevels, static_cast<uint32_t>(std::bit_width(std::max(width, height))));
  EXPECT_TRUE(pyramid.textureUsage & wgpu::TextureUsage::TextureBinding);
  EXPECT_TRUE(pyramid.textureUsage & wgpu::TextureUsage::StorageBinding);
  const BindingPlan& readerPlan = graph->bindingPlans[graph->find_node("t.cull")];
  ASSERT_EQ(readerPlan.slots.size(), 1u);
  EXPECT_EQ(readerPlan.slots[0].sampleType, TextureSampleType::UnfilterableFloat);
}

TEST_F(HzbGraph, SkippedUnlessDirect) {
  for (const auto [source, samples] :
       {std::pair{DepthSource::Unavailable, 1u}, std::pair{DepthSource::Copy, 1u}, std::pair{DepthSource::Copy, 4u}}) {
    SCOPED_TRACE(std::string{scene_depth_source_name(source)} + " x" + std::to_string(samples));
    test::TestDeviceOptions options = m_options;
    options.msaaSamples = samples;
    use_globals(options);
    const TestPass consumer = reader();
    const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
    const auto graph = compile(descs, source);
    EXPECT_EQ(graph->nodes[graph->find_node(hzb::OpaquePass)].status, CullReason::Skipped);
    const NodeId readerNode = graph->find_node("t.cull");
    ASSERT_EQ(graph->nodes[readerNode].status, CullReason::None);
    const CompiledUse& use = graph->node_uses(readerNode)[0];
    EXPECT_TRUE(use.staticDummy);
    ASSERT_TRUE(use.hasFallback);
    EXPECT_EQ(use.fallback.color, DummyColor::Black);
    EXPECT_EQ(use.fallback.sampleType, TextureSampleType::UnfilterableFloat);
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, 1, nullptr), std::nullopt);
    EXPECT_EQ(consumer.calls, 1);
    EXPECT_EQ(hzb::testing::counters().dispatches, 0u);
  }
}

TEST_F(HzbGraph, EncodesValidationClean) { encode_sizes(); }

TEST_F(HzbGraph, EncodesValidationCleanWithProductionToggles) {
  TearDown();
  m_started = false;
  if (!open({.backend = wgpu::BackendType::Null, .productionToggles = true})) {
    GTEST_SKIP() << "no null device: " << m_error;
  }
  encode_sizes();
}

TEST_F(HzbGraph, MissingDepthAtBindTimeBindsBlack) {
  const TestPass consumer = reader();
  const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
  const auto graph = compile(descs);
  wait_ready(*graph, hzb::OpaquePass);
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1, nullptr), std::nullopt);
  EXPECT_EQ(hzb::testing::counters().dispatches, pyramid_levels(m_options.efbSize.width, m_options.efbSize.height));
  EXPECT_EQ(consumer.calls, 1);
}

TEST_F(HzbGraph, PendingPipelineGivesConsumerDummy) {
  std::atomic_bool release = false;
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!release && !context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const TestPass consumer = reader();
  const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
  const auto graph = compile(descs);
  const NodeId node = graph->find_node(hzb::OpaquePass);
  const uint32_t id = graph->nodes[node].pipelineIds[0];
  ASSERT_NE(id, 0u);
  EXPECT_EQ(shaders::status({id}), shaders::PipelineStatus::Pending);
  const FrameState state{.graph = graph};
  const NodeId readerNode = graph->find_node("t.cull");
  const StageData stage = build_opaque(state, 1, scene_depth(make_depth(64, 64)));
  EXPECT_FALSE(stage.is_enabled(graph->nodes[node].programIndex));
  EXPECT_TRUE(stage.is_enabled(graph->nodes[readerNode].programIndex));
  EXPECT_TRUE(stage.dummy_bound(graph->node_uses(readerNode)[0].stageUse));
  EXPECT_EQ(run(state, 2, scene_depth(make_depth(64, 64))), std::nullopt);
  EXPECT_EQ(hzb::testing::counters().dispatches, 0u);
  EXPECT_EQ(consumer.calls, 1);
  release = true;
  gate.wait();
}

TEST_F(HzbGraph, UnrequestedPipelineGivesConsumerDummy) {
  shaders::quiesce();
  const TestPass consumer = reader();
  const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
  const auto graph = compile(descs);
  const NodeId node = graph->find_node(hzb::OpaquePass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_EQ(graph->nodes[node].pipelineIds[0], 0u);
  EXPECT_FALSE(pass_pipelines_ready(graph->nodes[node]));
  const FrameState state{.graph = graph};
  const NodeId readerNode = graph->find_node("t.cull");
  const StageData stage = build_opaque(state, 1, scene_depth(make_depth(64, 64)));
  EXPECT_FALSE(stage.is_enabled(graph->nodes[node].programIndex));
  EXPECT_TRUE(stage.is_enabled(graph->nodes[readerNode].programIndex));
  EXPECT_TRUE(stage.dummy_bound(graph->node_uses(readerNode)[0].stageUse));
  EXPECT_EQ(run(state, 2, scene_depth(make_depth(64, 64))), std::nullopt);
  EXPECT_EQ(hzb::testing::counters().dispatches, 0u);
  EXPECT_EQ(consumer.calls, 1);
}

class HzbGpu : public HzbDeviceTest {
protected:
  void SetUp() override {
    m_backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!reopen(false)) {
      GTEST_SKIP() << m_skip;
    }
  }

  bool reopen(bool productionToggles, bool preciseTimestamps = false) {
    if (m_started) {
      TearDown();
      m_started = false;
    }
    if (!open({.backend = m_backend, .productionToggles = productionToggles, .preciseTimestamps = preciseTimestamps})) {
      m_skip = "no adapter: " + m_error;
      return false;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      m_skip = "the null backend runs no shaders";
      return false;
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(m_device->adapterInfo.backendType));
    return true;
  }

  [[nodiscard]] DepthSource device_source() const noexcept {
    return select_scene_depth_source(m_device->core, 1, true);
  }

  [[nodiscard]] bool cpu_adapter() const noexcept {
    return m_device->adapterInfo.adapterType == wgpu::AdapterType::CPU;
  }

  wgpu::BackendType m_backend = wgpu::BackendType::Undefined;
  std::string m_skip;
};

TEST_F(HzbGpu, PyramidMatchesCpuBitExact) {
  if (device_source() != DepthSource::Direct) {
    GTEST_SKIP() << "scene depth is not direct on this device";
  }
  set_debug_poison(true);
  std::mt19937 rng{Seed + 3};
  for (const auto [width, height] :
       {std::pair{3u, 5u}, std::pair{61u, 37u}, std::pair{64u, 64u}, std::pair{1279u, 719u}, std::pair{1280u, 720u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    test::TestDeviceOptions options = m_options;
    options.efbSize = {width, height, 1};
    use_globals(options);
    const std::vector<float> values = random_depth(rng, width, height);
    const ref::CpuPyramid want = ref::build_pyramid(values, width, height);

    wgpu::Texture fromDepth;
    {
      const TestPass capture = capture_reader(fromDepth);
      const std::vector descs{hzb::opaque_pass_desc(), desc_of(capture)};
      const auto graph = compile(descs);
      wait_ready(*graph, hzb::OpaquePass);
      const wgpu::Texture depth = make_depth(width, height);
      fill_depth(depth, values);
      const FrameState state{.graph = graph};
      EXPECT_EQ(run(state, 1, scene_depth(depth)), std::nullopt);
      expect_pyramid(fromDepth, want, "Depth32Float source");
    }

    use_globals(options);
    wgpu::Texture fromR32;
    const uint32_t rowBytes = (width * 4 + 255) & ~255u;
    std::vector<float> staged(size_t{rowBytes / 4} * height);
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(staged.data() + size_t{y} * (rowBytes / 4), values.data() + size_t{y} * width, width * 4);
    }
    const wgpu::Buffer staging = make_buffer(wgpu::BufferUsage::CopySrc, staged.size() * 4, staged.data());
    TestPass source{
        .name = "t.src",
        .kind = PassKind::Copy,
        .order = 0,
        .declare =
            [](InternalPassBuilder& b) {
              b.after(phase::Geometry);
              b.create("t.src", {.format = TextureFormat::R32Float, .sizeClass = SizeClass::Efb}, Access::CopyDst);
            },
        .encode =
            [&](PassContext& ctx) {
              const wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = rowBytes, .rowsPerImage = height},
                                                   .buffer = staging};
              const wgpu::TexelCopyTextureInfo to{.texture = ctx.texture(TexRef{0})};
              const wgpu::Extent3D extent{width, height, 1};
              ctx.encoder().CopyBufferToTexture(&from, &to, &extent);
            },
    };
    const TestPass capture = capture_reader(fromR32);
    const std::vector descs{desc_of(source), r32_build_desc(), desc_of(capture)};
    const auto graph = compile(descs);
    wait_ready(*graph, "t.build");
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, 1, nullptr), std::nullopt);
    expect_pyramid(fromR32, want, "R32Float source");
  }
}

TEST_F(HzbGpu, QueryMatchesCpuBitExact) {
  if (device_source() != DepthSource::Direct) {
    GTEST_SKIP() << "scene depth is not direct on this device";
  }
  std::mt19937 rng{Seed + 4};
  for (const auto [width, height] : {std::pair{61u, 37u}, std::pair{1280u, 720u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    test::TestDeviceOptions options = m_options;
    options.efbSize = {width, height, 1};
    use_globals(options);
    std::vector<float> values = random_depth(rng, width, height);
    for (size_t i = 0; i < values.size(); ++i) {
      const uint32_t x = static_cast<uint32_t>(i % width) / 16;
      const uint32_t y = static_cast<uint32_t>(i / width) / 16;
      values[i] = std::max(values[i] * 0x1p-8f + static_cast<float>((x * 7 + y * 13) % 64) / 64.f, 0.f);
      values[i] = std::min(values[i], 1.f - DepthStep);
      values[i] = std::floor(values[i] / DepthStep) * DepthStep;
    }
    const ref::CpuPyramid pyramid = ref::build_pyramid(values, width, height);
    const ref::RangeMin range{values, width, height};
    constexpr uint32_t Queries = 65536;
    std::vector<RectQuery> queries(Queries);
    std::vector<uint32_t> want(Queries);
    uint32_t culled = 0;
    for (uint32_t i = 0; i < Queries; ++i) {
      const ref::Rect rect = random_rect(rng, width, height);
      const float nearest = pick_nearest(rng, range.min(rect));
      queries[i] = {.rect = {rect.x0, rect.y0, rect.x1, rect.y1}, .nearest = nearest};
      want[i] = ref::occluded_rect(pyramid, rect, nearest) ? 1 : 0;
      culled += want[i];
    }
    EXPECT_GT(culled, Queries / 50);
    m_count = Queries;
    const TestParams params{.count = Queries};
    const wgpu::Buffer paramsBuffer = make_buffer(wgpu::BufferUsage::Uniform, sizeof(params), &params);
    const wgpu::Buffer queryBuffer =
        make_buffer(wgpu::BufferUsage::Storage, queries.size() * sizeof(RectQuery), queries.data());
    const wgpu::Buffer verdicts =
        make_buffer(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, Queries * sizeof(uint32_t));
    bind_extra(paramsBuffer, queryBuffer, verdicts);
    const TestPass consumer = shader_reader("test/hzb_rects");
    const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
    const auto graph = compile(descs);
    wait_ready(*graph, hzb::OpaquePass);
    wait_ready(*graph, "t.cull");
    const wgpu::Texture depth = make_depth(width, height);
    fill_depth(depth, values);
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, 1, scene_depth(depth)), std::nullopt);
    const std::vector<uint32_t> got =
        test::read_buffer(*m_device, verdicts, 0, Queries * sizeof(uint32_t)).value_or(std::vector<uint32_t>{});
    ASSERT_EQ(got.size(), Queries);
    size_t bad = 0;
    for (uint32_t i = 0; i < Queries; ++i) {
      if (got[i] != want[i] && bad++ < 8) {
        ADD_FAILURE() << "query " << i << " rect " << queries[i].rect[0] << "," << queries[i].rect[1] << "-"
                      << queries[i].rect[2] << "," << queries[i].rect[3] << " nearest " << queries[i].nearest
                      << ": gpu " << got[i] << ", cpu " << want[i];
      }
    }
    EXPECT_EQ(bad, 0u);
  }
}

constexpr float Near = 0.5f;
constexpr float Far = 100.f;

struct Camera {
  float sx = 1.f;
  float sy = 1.f;
  ref::Mat4 clipFromWorld{};
};

Camera make_camera(uint32_t width, uint32_t height) {
  Camera camera;
  camera.sy = 1.f / std::tan(std::numbers::pi_v<float> / 6.f);
  camera.sx = camera.sy * static_cast<float>(height) / static_cast<float>(width);
  camera.clipFromWorld[0] = camera.sx;
  camera.clipFromWorld[5] = camera.sy;
  camera.clipFromWorld[10] = Near / (Far - Near);
  camera.clipFromWorld[11] = -1.f;
  camera.clipFromWorld[14] = Near * Far / (Far - Near);
  return camera;
}

ref::Vec3 at_ndc(const Camera& camera, float x, float y, float distance) {
  return {x * distance / camera.sx, y * distance / camera.sy, -distance};
}

enum class BoxKind : uint8_t { Behind, Coplanar, Epsilon, Peek, NearPlane, OffScreen, SubPixel, Huge, Edge, Random };

constexpr std::array<std::string_view, 10> BoxKindNames{
    "behind", "coplanar", "epsilon", "peek", "near-plane", "off-screen", "sub-pixel", "huge", "edge", "random",
};

struct Box {
  ref::Vec3 bmin{};
  ref::Vec3 bmax{};
  BoxKind kind = BoxKind::Random;
};

struct BoxScene {
  Camera camera;
  std::vector<float> occluders;
  std::vector<Box> boxes;
};

void push_quad(std::vector<float>& out, const std::array<ref::Vec3, 4>& corners) {
  for (const size_t index : {0, 1, 2, 0, 2, 3}) {
    out.insert(out.end(), corners[index].begin(), corners[index].end());
  }
}

Box box_from_ndc(const Camera& camera, float x0, float y0, float x1, float y1, float front, float depth, BoxKind kind) {
  const ref::Vec3 a = at_ndc(camera, x0, y0, front);
  const ref::Vec3 b = at_ndc(camera, x1, y1, front);
  return {
      .bmin = {std::min(a[0], b[0]), std::min(a[1], b[1]), -(front + depth)},
      .bmax = {std::max(a[0], b[0]), std::max(a[1], b[1]), -front},
      .kind = kind,
  };
}

BoxScene make_box_scene(uint32_t width, uint32_t height, uint32_t count, std::mt19937& rng) {
  BoxScene scene{.camera = make_camera(width, height)};
  const Camera& camera = scene.camera;
  const float wall = uniform(rng, 4.f, 8.f);
  constexpr float WallEdge = 0.8f;
  const auto flat_quad = [&](float x0, float y0, float x1, float y1, float distance) {
    push_quad(scene.occluders, {at_ndc(camera, x0, y0, distance), at_ndc(camera, x1, y0, distance),
                                at_ndc(camera, x1, y1, distance), at_ndc(camera, x0, y1, distance)});
  };
  flat_quad(-WallEdge, -WallEdge, WallEdge, WallEdge, wall);
  const float edge = uniform(rng, 3.f, 6.f);
  flat_quad(0.9f, -1.5f, 1.5f, 1.5f, edge);
  flat_quad(-1.5f, -1.5f, 1.5f, -0.9f, edge);
  for (int i = 0; i < 16; ++i) {
    const float cx = uniform(rng, -1.f, 1.f);
    const float cy = uniform(rng, -1.f, 1.f);
    const float hx = uniform(rng, 0.05f, 0.35f);
    const float hy = uniform(rng, 0.05f, 0.35f);
    const float distance = uniform(rng, 1.5f, 40.f);
    const float tilt = uniform(rng, 0.f, 0.3f);
    push_quad(scene.occluders, {at_ndc(camera, cx - hx, cy - hy, distance * (1.f - tilt)),
                                at_ndc(camera, cx + hx, cy - hy, distance * (1.f + tilt)),
                                at_ndc(camera, cx + hx, cy + hy, distance * (1.f + tilt)),
                                at_ndc(camera, cx - hx, cy + hy, distance * (1.f - tilt))});
  }
  const float pixelX = 2.f / static_cast<float>(width);
  const float pixelY = 2.f / static_cast<float>(height);
  constexpr std::array PeekPixels{-2.f, -1.f, -0.5f, 0.f, 0.25f, 0.5f, 1.f, 2.f};
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t slot = i % 20;
    const float cx = uniform(rng, -0.55f, 0.55f);
    const float cy = uniform(rng, -0.55f, 0.55f);
    const float hx = uniform(rng, 0.003f, 0.15f);
    const float hy = uniform(rng, 0.003f, 0.15f);
    const float depth = uniform(rng, 0.05f, 5.f);
    if (slot < 6) {
      scene.boxes.push_back(box_from_ndc(camera, cx - hx, cy - hy, cx + hx, cy + hy, wall * uniform(rng, 1.001f, 4.f),
                                         depth, BoxKind::Behind));
    } else if (slot < 8) {
      scene.boxes.push_back(box_from_ndc(camera, cx - hx, cy - hy, cx + hx, cy + hy, wall, depth, BoxKind::Coplanar));
    } else if (slot < 10) {
      const float epsilon = std::ldexp(1.f, -static_cast<int>(uniform(rng, 10u, 24u)));
      scene.boxes.push_back(
          box_from_ndc(camera, cx - hx, cy - hy, cx + hx, cy + hy, wall * (1.f + epsilon), depth, BoxKind::Epsilon));
    } else if (slot < 12) {
      const float offset = PeekPixels[uniform(rng, 0u, static_cast<uint32_t>(PeekPixels.size() - 1))];
      const float inner = WallEdge - uniform(rng, 0.05f, 0.3f);
      const float front = wall * uniform(rng, 1.0001f, 3.f);
      const float across = uniform(rng, -0.5f, 0.5f);
      const float halfAcross = uniform(rng, 0.01f, 0.1f);
      switch (uniform(rng, 0u, 3u)) {
      case 0:
        scene.boxes.push_back(box_from_ndc(camera, inner, across - halfAcross, WallEdge + offset * pixelX,
                                           across + halfAcross, front, depth, BoxKind::Peek));
        break;
      case 1:
        scene.boxes.push_back(box_from_ndc(camera, -WallEdge - offset * pixelX, across - halfAcross, -inner,
                                           across + halfAcross, front, depth, BoxKind::Peek));
        break;
      case 2:
        scene.boxes.push_back(box_from_ndc(camera, across - halfAcross, inner, across + halfAcross,
                                           WallEdge + offset * pixelY, front, depth, BoxKind::Peek));
        break;
      default:
        scene.boxes.push_back(box_from_ndc(camera, across - halfAcross, -WallEdge - offset * pixelY,
                                           across + halfAcross, -inner, front, depth, BoxKind::Peek));
        break;
      }
    } else if (slot < 14) {
      const float front = uniform(rng, -3.f, 0.6f);
      const float x = uniform(rng, -1.f, 1.f) * 2.f / camera.sx;
      const float y = uniform(rng, -1.f, 1.f) * 2.f / camera.sy;
      const float size = uniform(rng, 0.05f, 1.5f);
      scene.boxes.push_back({.bmin = {x - size, y - size, -(front + uniform(rng, 0.2f, 4.f))},
                             .bmax = {x + size, y + size, -front},
                             .kind = BoxKind::NearPlane});
    } else if (slot == 14) {
      const float side = uniform(rng, 0u, 1u) == 0 ? -1.f : 1.f;
      const float x = side * uniform(rng, 1.3f, 3.f);
      scene.boxes.push_back(
          box_from_ndc(camera, x - hx, cy - hy, x + hx, cy + hy, uniform(rng, 2.f, 50.f), depth, BoxKind::OffScreen));
    } else if (slot == 15) {
      const float sub = uniform(rng, 0.05f, 0.9f) * 0.5f * pixelX;
      const float front = uniform(rng, 0u, 1u) == 0 ? wall * uniform(rng, 1.0001f, 2.f) : uniform(rng, 1.f, wall);
      scene.boxes.push_back(
          box_from_ndc(camera, cx - sub, cy - sub, cx + sub, cy + sub, front, depth * 0.01f, BoxKind::SubPixel));
    } else if (slot == 16) {
      const float big = uniform(rng, 0.7f, 2.5f);
      const float front = uniform(rng, 0u, 1u) == 0 ? wall * uniform(rng, 1.f, 1.5f) : uniform(rng, 1.f, 60.f);
      scene.boxes.push_back(box_from_ndc(camera, cx - big, cy - big, cx + big, cy + big, front, depth, BoxKind::Huge));
    } else if (slot == 17) {
      const float offset = std::array{-1.f, 0.f, 0.5f, 1.f}[uniform(rng, 0u, 3u)];
      const float front = edge * uniform(rng, 1.0001f, 3.f);
      const float across = uniform(rng, -0.6f, 0.6f);
      if (uniform(rng, 0u, 1u) == 0) {
        scene.boxes.push_back(box_from_ndc(camera, 0.92f, across - 0.05f, 1.f + offset * pixelX, across + 0.05f, front,
                                           depth, BoxKind::Edge));
      } else {
        scene.boxes.push_back(box_from_ndc(camera, across - 0.05f, -1.f - offset * pixelY, across + 0.05f, -0.92f,
                                           front, depth, BoxKind::Edge));
      }
    } else {
      const float x = uniform(rng, -1.2f, 1.2f);
      const float y = uniform(rng, -1.2f, 1.2f);
      const float half = uniform(rng, 0.01f, 0.4f);
      scene.boxes.push_back(box_from_ndc(camera, x - half, y - half, x + half, y + half, uniform(rng, 0.6f, 90.f),
                                         uniform(rng, 0.05f, 10.f), BoxKind::Random));
    }
  }
  return scene;
}

std::vector<float> box_vertices(std::span<const Box> boxes) {
  constexpr std::array<std::array<uint32_t, 4>, 6> Faces{{
      {0, 1, 3, 2},
      {4, 5, 7, 6},
      {0, 1, 5, 4},
      {2, 3, 7, 6},
      {0, 2, 6, 4},
      {1, 3, 7, 5},
  }};
  std::vector<float> out;
  out.reserve(boxes.size() * 36 * 3);
  for (const Box& box : boxes) {
    std::array<ref::Vec3, 8> corners{};
    for (uint32_t i = 0; i < 8; ++i) {
      corners[i] = {(i & 1) != 0 ? box.bmax[0] : box.bmin[0], (i & 2) != 0 ? box.bmax[1] : box.bmin[1],
                    (i & 4) != 0 ? box.bmax[2] : box.bmin[2]};
    }
    for (const auto& face : Faces) {
      push_quad(out, {corners[face[0]], corners[face[1]], corners[face[2]], corners[face[3]]});
    }
  }
  return out;
}

TEST_F(HzbGpu, NeverCullsVisibleBox) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 3> Sizes{{{64, 64}, {61, 37}, {1280, 720}}};
  constexpr std::array<std::pair<float, float>, 2> Ranges{{{0.f, 1.f}, {0.25f, 0.75f}}};
  std::mt19937 rng{Seed + 5};
  size_t totalCulled = 0;
  for (const bool productionToggles : {false, true}) {
    ASSERT_TRUE(reopen(productionToggles)) << m_skip;
    const DepthSource source = device_source();
    const uint32_t count = cpu_adapter() ? 500 : 2000;
    const wgpu::Device& device = m_device->device;
    const std::string code{GeometryWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    const wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    const wgpu::VertexAttribute attribute{.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0};
    const wgpu::VertexBufferLayout vertexLayout{
        .stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = 12, .attributeCount = 1, .attributes = &attribute};
    const auto make_pipeline = [&](bool write) {
      const wgpu::DepthStencilState depthState{
          .format = TextureFormat::Depth32Float,
          .depthWriteEnabled = write ? wgpu::OptionalBool::True : wgpu::OptionalBool::False,
          .depthCompare = wgpu::CompareFunction::GreaterEqual,
      };
      const wgpu::RenderPipelineDescriptor descriptor{
          .vertex = {.module = module, .entryPoint = "vs", .bufferCount = 1, .buffers = &vertexLayout},
          .depthStencil = &depthState,
      };
      return device.CreateRenderPipeline(&descriptor);
    };
    const wgpu::RenderPipeline occluderPipeline = make_pipeline(true);
    const wgpu::RenderPipeline boxPipeline = make_pipeline(false);

    for (const auto [width, height] : Sizes) {
      for (const auto [minDepth, maxDepth] : Ranges) {
        const std::string config = std::string{productionToggles ? "release" : "validated"} + " " +
                                   std::to_string(width) + "x" + std::to_string(height) + " depth " +
                                   std::to_string(minDepth) + "-" + std::to_string(maxDepth);
        SCOPED_TRACE(config);
        test::TestDeviceOptions options = m_options;
        options.efbSize = {width, height, 1};
        use_globals(options);
        const BoxScene scene = make_box_scene(width, height, count, rng);
        const auto& matrix = scene.camera.clipFromWorld;
        const wgpu::Buffer matrixBuffer = make_buffer(wgpu::BufferUsage::Uniform, sizeof(matrix), matrix.data());
        const wgpu::Buffer occluderBuffer =
            make_buffer(wgpu::BufferUsage::Vertex, scene.occluders.size() * sizeof(float), scene.occluders.data());
        const std::vector<float> boxVertices = box_vertices(scene.boxes);
        const wgpu::Buffer boxVertexBuffer =
            make_buffer(wgpu::BufferUsage::Vertex, boxVertices.size() * sizeof(float), boxVertices.data());
        std::vector<std::array<float, 8>> boxData;
        for (const Box& box : scene.boxes) {
          boxData.push_back({box.bmin[0], box.bmin[1], box.bmin[2], 1.f, box.bmax[0], box.bmax[1], box.bmax[2], 1.f});
        }
        const wgpu::Buffer boxBuffer =
            make_buffer(wgpu::BufferUsage::Storage, boxData.size() * sizeof(boxData[0]), boxData.data());
        const TestParams params{
            .clipFromWorld = matrix,
            .viewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height)},
            .depthRange = {minDepth, maxDepth},
            .efbSize = {width, height},
            .count = count,
        };
        const wgpu::Buffer paramsBuffer = make_buffer(wgpu::BufferUsage::Uniform, sizeof(params), &params);
        const wgpu::Buffer verdicts =
            make_buffer(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, count * sizeof(uint32_t));
        bind_extra(paramsBuffer, boxBuffer, verdicts);
        m_count = count;

        const wgpu::Texture depth = make_depth(width, height);
        const auto geometry_group = [&](const wgpu::RenderPipeline& pipeline) {
          const wgpu::BindGroupEntry entry{.binding = 0, .buffer = matrixBuffer};
          const wgpu::BindGroupDescriptor descriptor{
              .layout = pipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &entry};
          return device.CreateBindGroup(&descriptor);
        };
        {
          const wgpu::RenderPassDepthStencilAttachment attachment{
              .view = depth.CreateView(),
              .depthLoadOp = wgpu::LoadOp::Clear,
              .depthStoreOp = wgpu::StoreOp::Store,
              .depthClearValue = 0.f,
          };
          const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
          const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
          const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
          pass.SetViewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height), minDepth, maxDepth);
          pass.SetPipeline(occluderPipeline);
          pass.SetBindGroup(0, geometry_group(occluderPipeline));
          pass.SetVertexBuffer(0, occluderBuffer);
          pass.Draw(static_cast<uint32_t>(scene.occluders.size() / 3));
          pass.End();
          const wgpu::CommandBuffer commands = encoder.Finish();
          ASSERT_TRUE(test::submit_and_wait(*m_device, commands));
        }

        const TestPass consumer = shader_reader("test/hzb_cull");
        const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
        const auto graph = compile(descs, source);
        const CompiledNode& opaque = graph->nodes[graph->find_node(hzb::OpaquePass)];
        if (source != DepthSource::Direct) {
          EXPECT_EQ(opaque.status, CullReason::Skipped);
        } else {
          ASSERT_EQ(opaque.status, CullReason::None);
          wait_ready(*graph, hzb::OpaquePass);
        }
        wait_ready(*graph, "t.cull");
        const FrameState state{.graph = graph};
        ASSERT_EQ(run(state, 1, scene_depth(depth)), std::nullopt);
        EXPECT_EQ(consumer.calls, 1);

        const wgpu::QuerySetDescriptor queryDescriptor{.type = wgpu::QueryType::Occlusion, .count = count};
        const wgpu::QuerySet queries = device.CreateQuerySet(&queryDescriptor);
        const wgpu::Buffer resolved =
            make_buffer(wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, uint64_t{count} * 8);
        {
          const wgpu::RenderPassDepthStencilAttachment attachment{.view = depth.CreateView(), .depthReadOnly = true};
          const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment,
                                                          .occlusionQuerySet = queries};
          const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
          const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
          pass.SetViewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height), minDepth, maxDepth);
          pass.SetPipeline(boxPipeline);
          pass.SetBindGroup(0, geometry_group(boxPipeline));
          pass.SetVertexBuffer(0, boxVertexBuffer);
          for (uint32_t i = 0; i < count; ++i) {
            pass.BeginOcclusionQuery(i);
            pass.Draw(36, 1, 36 * i, 0);
            pass.EndOcclusionQuery();
          }
          pass.End();
          encoder.ResolveQuerySet(queries, 0, count, resolved, 0);
          const wgpu::CommandBuffer commands = encoder.Finish();
          ASSERT_TRUE(test::submit_and_wait(*m_device, commands));
        }
        const std::vector<uint32_t> culled =
            test::read_buffer(*m_device, verdicts, 0, uint64_t{count} * 4).value_or(std::vector<uint32_t>{});
        const std::vector<uint32_t> samples =
            test::read_buffer(*m_device, resolved, 0, uint64_t{count} * 8).value_or(std::vector<uint32_t>{});
        ASSERT_EQ(culled.size(), count);
        ASSERT_EQ(samples.size(), size_t{count} * 2);
        size_t zero = 0;
        size_t hidden = 0;
        size_t failures = 0;
        std::array<size_t, BoxKindNames.size()> culledByKind{};
        std::array<size_t, BoxKindNames.size()> zeroByKind{};
        for (uint32_t i = 0; i < count; ++i) {
          const uint64_t passed = samples[2 * i] | (uint64_t{samples[2 * i + 1]} << 32);
          const Box& box = scene.boxes[i];
          zero += passed == 0 ? 1 : 0;
          zeroByKind[static_cast<size_t>(box.kind)] += passed == 0 ? 1 : 0;
          if (culled[i] == 0) {
            continue;
          }
          ++hidden;
          ++culledByKind[static_cast<size_t>(box.kind)];
          if (passed != 0 && failures++ < 16) {
            ADD_FAILURE() << "culled " << BoxKindNames[static_cast<size_t>(box.kind)] << " box " << i << " ("
                          << box.bmin[0] << ", " << box.bmin[1] << ", " << box.bmin[2] << ")-(" << box.bmax[0] << ", "
                          << box.bmax[1] << ", " << box.bmax[2] << ") passed " << passed << " samples";
          }
        }
        EXPECT_EQ(failures, 0u);
        totalCulled += hidden;
        if (source == DepthSource::Direct) {
          EXPECT_GT(zero, 0u);
          EXPECT_GE(hidden * 10, zero * 3) << hidden << " culled of " << zero << " hidden boxes";
        } else {
          EXPECT_EQ(hidden, 0u);
        }
        std::string kinds;
        for (size_t kind = 0; kind < BoxKindNames.size(); ++kind) {
          kinds += " " + std::string{BoxKindNames[kind]} + "=" + std::to_string(culledByKind[kind]) + "/" +
                   std::to_string(zeroByKind[kind]);
        }
        std::printf("hzb boxes %s backend %d: %u boxes, %zu with no samples, %zu culled (%s )\n", config.c_str(),
                    static_cast<int>(m_device->adapterInfo.backendType), count, zero, hidden, kinds.c_str());
      }
    }
  }
  ::testing::Test::RecordProperty("culled", static_cast<int>(totalCulled));
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  const auto index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1) + 0.5);
  return values[std::min(index, values.size() - 1)];
}

TEST_F(HzbGpu, Bench) {
  const char* enabled = std::getenv("AURORA_TEST_BENCH");
  if (enabled == nullptr || std::string_view{enabled} != "1") {
    GTEST_SKIP() << "set AURORA_TEST_BENCH=1 to run";
  }
  ASSERT_TRUE(reopen(true, true)) << m_skip;
  if (device_source() != DepthSource::Direct || !m_device->device.HasFeature(wgpu::FeatureName::TimestampQuery)) {
    GTEST_SKIP() << "needs direct scene depth and timestamp queries";
  }
  constexpr uint32_t Warmup = 50;
  constexpr uint32_t Timed = 200;
  for (const auto [width, height] :
       {std::pair{1280u, 720u}, std::pair{1920u, 1080u}, std::pair{2560u, 1440u}, std::pair{3840u, 2160u}}) {
    test::TestDeviceOptions options = m_options;
    options.efbSize = {width, height, 1};
    use_globals(options);
    const TestPass consumer = reader();
    const std::vector descs{hzb::opaque_pass_desc(), desc_of(consumer)};
    const auto graph = compile(descs);
    wait_ready(*graph, hzb::OpaquePass);
    const wgpu::Texture depth = make_depth(width, height);
    fill_depth(depth, std::vector<float>(size_t{width} * height, 0.5f));
    const auto sceneDepth = scene_depth(depth);
    const FrameState state{.graph = graph};
    const wgpu::QuerySetDescriptor queryDescriptor{.type = wgpu::QueryType::Timestamp, .count = 2};
    const wgpu::QuerySet queries = m_device->device.CreateQuerySet(&queryDescriptor);
    const wgpu::Buffer resolved = make_buffer(wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, 16);
    std::vector<double> gpuMs;
    std::vector<double> cpuUs;
    double coldUs = 0.0;
    for (uint32_t frame = 1; frame <= Warmup + Timed; ++frame) {
      const StageData opaque = build_opaque(state, frame, sceneDepth);
      const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
      encoder.WriteTimestamp(queries, 0);
      const auto start = std::chrono::steady_clock::now();
      run_stage(encoder, opaque);
      const auto stop = std::chrono::steady_clock::now();
      encoder.WriteTimestamp(queries, 1);
      encoder.ResolveQuerySet(queries, 0, 2, resolved, 0);
      const wgpu::CommandBuffer commands = encoder.Finish();
      ASSERT_TRUE(test::submit_and_wait(*m_device, commands));
      const double us = std::chrono::duration<double, std::micro>(stop - start).count();
      if (frame == 1) {
        coldUs = us;
      }
      if (frame <= Warmup) {
        continue;
      }
      const std::vector<uint32_t> stamps =
          test::read_buffer(*m_device, resolved, 0, 16).value_or(std::vector<uint32_t>{});
      ASSERT_EQ(stamps.size(), 4u);
      const uint64_t begin = stamps[0] | (uint64_t{stamps[1]} << 32);
      const uint64_t end = stamps[2] | (uint64_t{stamps[3]} << 32);
      gpuMs.push_back(static_cast<double>(end - begin) / 1e6);
      cpuUs.push_back(us);
    }
    const uint64_t bytes = [&] {
      uint64_t total = 0;
      for (uint32_t level = 0; level < pyramid_levels(width, height); ++level) {
        total += uint64_t{hzb::level_size((width + 1) / 2, level)} * hzb::level_size((height + 1) / 2, level) * 4;
      }
      return total;
    }();
    std::printf(
        "hzb bench %ux%u backend %d adapter %s: gpu median %.4f ms p95 %.4f ms, encode cold %.1f us warm "
        "median %.1f us, pyramid %llu bytes\n",
        width, height, static_cast<int>(m_device->adapterInfo.backendType),
        std::string{std::string_view{m_device->adapterInfo.device}}.c_str(), percentile(gpuMs, 0.5),
        percentile(gpuMs, 0.95), coldUs, percentile(cpuUs, 0.5), static_cast<unsigned long long>(bytes));
  }
}
} // namespace
} // namespace aurora::render::detail
