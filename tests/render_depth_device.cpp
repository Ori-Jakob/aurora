#include "render_depth_device.hpp"

#include <gtest/gtest.h>

#include "render/scene_depth.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::test {
namespace {
constexpr std::string_view DepthFillWgsl = R"(
@vertex
fn vs(@location(0) point: vec3f) -> @builtin(position) vec4f {
  return vec4f(point, 1.0);
}
)";

wgpu::Buffer make_buffer(TestDevice& device, wgpu::BufferUsage usage, uint64_t size, const void* data) {
  const wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst,
                                          .size = std::max<uint64_t>((size + 3) & ~uint64_t{3}, 16)};
  const wgpu::Buffer buffer = device.device.CreateBuffer(&descriptor);
  if (data != nullptr && size != 0) {
    device.queue.WriteBuffer(buffer, 0, data, (size + 3) & ~uint64_t{3});
  }
  return buffer;
}
} // namespace

wgpu::Texture make_depth(TestDevice& device, uint32_t width, uint32_t height) {
  const wgpu::TextureDescriptor descriptor{
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
      .size = {width, height, 1},
      .format = wgpu::TextureFormat::Depth32Float,
  };
  return device.device.CreateTexture(&descriptor);
}

std::shared_ptr<const render::detail::SceneDepth> scene_depth(const wgpu::Texture& texture) {
  auto depth = std::make_shared<render::detail::SceneDepth>();
  depth->source = render::DepthSource::Direct;
  depth->sampledTexture = texture;
  const wgpu::TextureViewDescriptor view{.aspect = wgpu::TextureAspect::DepthOnly};
  depth->sampled = texture.CreateView(&view);
  depth->sampledFormat = texture.GetFormat();
  depth->size = {texture.GetWidth(), texture.GetHeight(), 1};
  return depth;
}

void fill_depth(TestDevice& device, const wgpu::Texture& texture, std::span<const float> values) {
  const std::string code{DepthFillWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
  const wgpu::ShaderModule module = device.device.CreateShaderModule(&moduleDescriptor);
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
      .format = wgpu::TextureFormat::Depth32Float,
      .depthWriteEnabled = wgpu::OptionalBool::True,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .vertex = {.module = module, .entryPoint = "vs", .bufferCount = 1, .buffers = &vertexLayout},
      .primitive = {.topology = wgpu::PrimitiveTopology::PointList},
      .depthStencil = &depthState,
  };
  const wgpu::RenderPipeline pipeline = device.device.CreateRenderPipeline(&pipelineDescriptor);
  const wgpu::Buffer buffer =
      make_buffer(device, wgpu::BufferUsage::Vertex, points.size() * sizeof(float), points.data());
  const wgpu::RenderPassDepthStencilAttachment attachment{
      .view = texture.CreateView(),
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 0.f,
  };
  const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
  const wgpu::CommandEncoder encoder = device.device.CreateCommandEncoder();
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(pipeline);
  pass.SetVertexBuffer(0, buffer);
  pass.Draw(static_cast<uint32_t>(values.size()));
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  EXPECT_TRUE(submit_and_wait(device, commands));
  const std::vector<float> stored = read_texture_mip(device, texture, 0);
  ASSERT_EQ(stored.size(), values.size());
  size_t bad = 0;
  for (size_t i = 0; i < stored.size(); ++i) {
    bad += std::bit_cast<uint32_t>(stored[i]) != std::bit_cast<uint32_t>(values[i]) ? 1 : 0;
  }
  EXPECT_EQ(bad, 0u) << "the depth fill did not store every value exactly";
}
} // namespace aurora::test
