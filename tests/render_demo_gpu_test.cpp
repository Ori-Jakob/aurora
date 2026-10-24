#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "cull_reference.hpp"
#include "render/graph_layout.hpp"

#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace aurora;
using render_demo::CullInstance;
using render_demo::CullLayout;
using render_demo::Plane;

enum class DeviceKind { Core, Compat, CoreProduction };

constexpr uint64_t ParamsOffset = 256;
constexpr uint64_t FrameBufferSize = 512;

std::array<Plane, 6> uniform_planes(Plane plane) { return {plane, plane, plane, plane, plane, plane}; }

std::array<f32, 3> sub(std::array<f32, 3> a, std::array<f32, 3> b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

std::array<f32, 3> cross(std::array<f32, 3> a, std::array<f32, 3> b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

f32 dot(std::array<f32, 3> a, std::array<f32, 3> b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

std::array<f32, 3> normalize(std::array<f32, 3> v) {
  const f32 length = std::sqrt(dot(v, v));
  return {v[0] / length, v[1] / length, v[2] / length};
}

// The demo camera at frame 30, built the way C_MTXPerspective and C_MTXLookAt build it
std::array<Plane, 6> frame30_planes() {
  const f32 t = 30.f / 60.f;
  const std::array<f32, 3> eye{28.f * std::cos(0.25f * t), 11.f, 28.f * std::sin(0.25f * t)};
  const std::array<f32, 3> target{0.f, 1.5f, 0.f};
  const std::array<f32, 3> look = normalize(sub(eye, target));
  const std::array<f32, 3> right = normalize(cross({0.f, 1.f, 0.f}, look));
  const std::array<f32, 3> up = cross(look, right);
  f32 view[3][4];
  for (u32 i = 0; i < 3; ++i) {
    view[0][i] = right[i];
    view[1][i] = up[i];
    view[2][i] = look[i];
  }
  view[0][3] = -dot(right, eye);
  view[1][3] = -dot(up, eye);
  view[2][3] = -dot(look, eye);
  constexpr f32 Near = 0.5f;
  constexpr f32 Far = 400.f;
  const f32 cot = 1.f / std::tan(55.f * 0.5f * 3.14159265358979f / 180.f);
  f32 proj[4][4]{};
  proj[0][0] = cot / (4.f / 3.f);
  proj[1][1] = cot;
  proj[2][2] = -Near / (Far - Near);
  proj[2][3] = -(Far * Near) / (Far - Near);
  proj[3][2] = -1.f;
  return render_demo::frustum_planes(render_demo::view_projection(proj, view, true));
}

class RenderDemoGpu : public ::testing::TestWithParam<DeviceKind> {
protected:
  void SetUp() override {
    test::TestDeviceOptions options{
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
    };
    switch (GetParam()) {
    case DeviceKind::Core:
      options.profile = FEATURE_PROFILE_FULL;
      break;
    case DeviceKind::Compat:
      options.core = false;
      options.profile = FEATURE_PROFILE_COMPAT;
      break;
    case DeviceKind::CoreProduction:
      options.profile = FEATURE_PROFILE_FULL;
      options.productionToggles = true;
      break;
    }
    std::string error;
    m_device = test::create_test_device(options, error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(m_device->adapterInfo.backendType));
    const auto frameEntries = render::detail::frame_layout_entries();
    m_frameLayout = layout(frameEntries);
    m_outLayout = layout(std::array{wgpu::BindGroupLayoutEntry{
        .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage}}});
    m_instanceLayout = layout(
        std::array{wgpu::BindGroupLayoutEntry{.binding = 0,
                                              .visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex,
                                              .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}}});
    m_frameBuffer = buffer(FrameBufferSize, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    const auto sampler = [&](wgpu::FilterMode filter, wgpu::AddressMode address, wgpu::CompareFunction compare) {
      const wgpu::SamplerDescriptor descriptor{
          .addressModeU = address,
          .addressModeV = address,
          .magFilter = filter,
          .minFilter = filter,
          .compare = compare,
      };
      return m_device->device.CreateSampler(&descriptor);
    };
    const std::array<wgpu::BindGroupEntry, 6> entries{{
        {.binding = 0, .buffer = m_frameBuffer, .size = sizeof(render::FrameConstants)},
        {.binding = 1, .buffer = m_frameBuffer, .offset = ParamsOffset, .size = sizeof(render_demo::CullParams)},
        {.binding = 2,
         .sampler =
             sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge, wgpu::CompareFunction::Undefined)},
        {.binding = 3,
         .sampler =
             sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge, wgpu::CompareFunction::Undefined)},
        {.binding = 4,
         .sampler = sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat, wgpu::CompareFunction::Undefined)},
        {.binding = 5,
         .sampler = sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge, wgpu::CompareFunction::Greater)},
    }};
    const wgpu::BindGroupDescriptor groupDescriptor{
        .layout = m_frameLayout, .entryCount = entries.size(), .entries = entries.data()};
    m_frameGroup = m_device->device.CreateBindGroup(&groupDescriptor);
  }

  void TearDown() override {
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
  }

  template <size_t N>
  wgpu::BindGroupLayout layout(const std::array<wgpu::BindGroupLayoutEntry, N>& entries) const {
    return layout(std::span<const wgpu::BindGroupLayoutEntry>{entries});
  }

  wgpu::BindGroupLayout layout(std::span<const wgpu::BindGroupLayoutEntry> entries) const {
    const wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroupLayout(&descriptor);
  }

  wgpu::Buffer buffer(uint64_t size, wgpu::BufferUsage usage) const {
    const wgpu::BufferDescriptor descriptor{.usage = usage, .size = size};
    return m_device->device.CreateBuffer(&descriptor);
  }

  wgpu::BindGroup group(const wgpu::BindGroupLayout& groupLayout, const wgpu::Buffer& resource, uint64_t size) const {
    const wgpu::BindGroupEntry entry{.binding = 0, .buffer = resource, .size = size};
    const wgpu::BindGroupDescriptor descriptor{.layout = groupLayout, .entryCount = 1, .entries = &entry};
    return m_device->device.CreateBindGroup(&descriptor);
  }

  wgpu::PipelineLayout pipeline_layout(const wgpu::BindGroupLayout& middle) const {
    const std::array layouts{m_frameLayout, middle, m_instanceLayout};
    const wgpu::PipelineLayoutDescriptor descriptor{.bindGroupLayoutCount = layouts.size(),
                                                    .bindGroupLayouts = layouts.data()};
    return m_device->device.CreatePipelineLayout(&descriptor);
  }

  wgpu::ShaderModule module(const std::string& body) const {
    const std::string code = std::string{render::FrameWgsl} + body;
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
    return m_device->device.CreateShaderModule(&descriptor);
  }

  std::vector<u32> run(std::span<const CullInstance> instances, const std::array<Plane, 6>& planes, u32 frameTag) {
    const CullLayout cull = render_demo::cull_layout(static_cast<u32>(instances.size()));
    const wgpu::ShaderModule shader = module(render_demo::cull_compute_wgsl(cull));
    const wgpu::PipelineLayout pipelineLayout = pipeline_layout(m_outLayout);
    std::array<wgpu::ComputePipeline, 3> pipelines;
    const std::array entryPoints{"cs_count", "cs_scan", "cs_scatter"};
    for (size_t i = 0; i < pipelines.size(); ++i) {
      const wgpu::ComputePipelineDescriptor descriptor{.layout = pipelineLayout,
                                                       .compute = {.module = shader, .entryPoint = entryPoints[i]}};
      pipelines[i] = m_device->device.CreateComputePipeline(&descriptor);
    }
    const render_demo::CullParams params{.planes = planes, .frameTag = frameTag, .pad = {}};
    m_device->queue.WriteBuffer(m_frameBuffer, ParamsOffset, &params, sizeof(params));
    const uint64_t instanceBytes = instances.size_bytes();
    const wgpu::Buffer instanceBuffer = buffer(instanceBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    m_device->queue.WriteBuffer(instanceBuffer, 0, instances.data(), instanceBytes);
    const wgpu::Buffer out = buffer(cull.bytes(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    const wgpu::BindGroup outGroup = group(m_outLayout, out, cull.bytes());
    const wgpu::BindGroup instanceGroup = group(m_instanceLayout, instanceBuffer, instanceBytes);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    const std::array<uint32_t, 2> offsets{0, 0};
    pass.SetBindGroup(0, m_frameGroup, offsets.size(), offsets.data());
    pass.SetBindGroup(1, outGroup);
    pass.SetBindGroup(2, instanceGroup);
    pass.SetPipeline(pipelines[0]);
    pass.DispatchWorkgroups(cull.groups);
    pass.SetPipeline(pipelines[1]);
    pass.DispatchWorkgroups(1);
    pass.SetPipeline(pipelines[2]);
    pass.DispatchWorkgroups(cull.groups);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    return test::read_buffer(*m_device, out, 0, cull.bytes()).value_or(std::vector<u32>{});
  }

  std::unique_ptr<test::TestDevice> m_device;
  wgpu::BindGroupLayout m_frameLayout;
  wgpu::BindGroupLayout m_outLayout;
  wgpu::BindGroupLayout m_instanceLayout;
  wgpu::Buffer m_frameBuffer;
  wgpu::BindGroup m_frameGroup;
};

TEST_P(RenderDemoGpu, CullKernelsMatchSimulation) {
  const std::array<std::pair<const char*, std::array<Plane, 6>>, 3> planeSets{{
      {"box",
       {{{1.f, 0.f, 0.f, 100.f},
         {-1.f, 0.f, 0.f, 50.5f},
         {0.f, 1.f, 0.f, 1.f},
         {0.f, -1.f, 0.f, 1.f},
         {0.f, 0.f, 1.f, 200.f},
         {0.f, 0.f, -1.f, 75.25f}}}},
      {"all in", uniform_planes({0.f, 0.f, 0.f, 1.f})},
      {"all out", uniform_planes({0.f, 0.f, 0.f, -1.f})},
  }};
  for (const u32 count : {1u, 127u, 129u, 70001u, 1u << 20}) {
    std::vector<CullInstance> instances(count);
    render_demo::fill_cull_instances(instances);
    for (const auto& [name, planes] : planeSets) {
      SCOPED_TRACE(std::string{name} + ", " + std::to_string(count) + " instances");
      const std::vector<u32> expected = render_demo::simulate_cull(instances, planes, count);
      const std::vector<u32> words = run(instances, planes, count);
      ASSERT_EQ(words.size(), expected.size());
      size_t mismatches = 0;
      size_t first = words.size();
      for (size_t i = 0; i < words.size(); ++i) {
        if (words[i] != expected[i]) {
          first = std::min(first, i);
          ++mismatches;
        }
      }
      EXPECT_EQ(mismatches, 0u) << "first mismatch at word " << first;
    }
  }
  std::vector<CullInstance> instances(1u << 20);
  render_demo::fill_cull_instances(instances);
  const std::array<Plane, 6> planes = frame30_planes();
  const std::vector<u32> words = run(instances, planes, render_demo::CullCheckFrame);
  const auto result = render_demo::check_cull_readback(words, render_demo::cull_layout(1u << 20), instances, planes,
                                                       render_demo::CullCheckFrame);
  EXPECT_EQ(result.failure, std::nullopt);
  EXPECT_GT(result.visible, 0u);
}

TEST_P(RenderDemoGpu, DrawPipelineCreates) {
  const wgpu::BindGroupLayout empty = layout(std::span<const wgpu::BindGroupLayoutEntry>{});
  const wgpu::ShaderModule shader = module(render_demo::cull_draw_wgsl(1u << 20));
  const std::array<wgpu::VertexAttribute, 2> attributes{{
      {.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0},
      {.format = wgpu::VertexFormat::Uint32, .offset = 0, .shaderLocation = 1},
  }};
  const std::array<wgpu::VertexBufferLayout, 2> buffers{{
      {.stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = 12, .attributeCount = 1, .attributes = &attributes[0]},
      {.stepMode = wgpu::VertexStepMode::Instance, .arrayStride = 4, .attributeCount = 1, .attributes = &attributes[1]},
  }};
  const wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA8Unorm};
  const wgpu::FragmentState fragment{.module = shader, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
  const wgpu::DepthStencilState depth{
      .format = wgpu::TextureFormat::Depth32Float,
      .depthWriteEnabled = wgpu::OptionalBool::False,
      .depthCompare = wgpu::CompareFunction::GreaterEqual,
  };
  m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
  const wgpu::RenderPipelineDescriptor descriptor{
      .layout = pipeline_layout(empty),
      .vertex = {.module = shader, .entryPoint = "vs_main", .bufferCount = buffers.size(), .buffers = buffers.data()},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depth,
      .fragment = &fragment,
  };
  const wgpu::RenderPipeline pipeline = m_device->device.CreateRenderPipeline(&descriptor);
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  EXPECT_TRUE(pipeline);
}

INSTANTIATE_TEST_SUITE_P(Devices, RenderDemoGpu,
                         ::testing::Values(DeviceKind::Core, DeviceKind::Compat, DeviceKind::CoreProduction),
                         [](const ::testing::TestParamInfo<DeviceKind>& info) {
                           switch (info.param) {
                           case DeviceKind::Core:
                             return std::string{"Core"};
                           case DeviceKind::Compat:
                             return std::string{"Compat"};
                           default:
                             return std::string{"CoreProduction"};
                           }
                         });
} // namespace
