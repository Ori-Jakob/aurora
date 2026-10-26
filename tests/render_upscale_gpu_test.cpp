#include <gtest/gtest.h>

#include "render_fsr_rig.hpp"
#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "render/jobs.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/post/upscale/fsr/fsr.hpp"
#include "render/post/upscale/fsr/fsr_pass.hpp"
#include "render/shaders.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if AURORA_ENABLE_NATIVE_D3D12
#include "gfx/encoding.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/tex_copy_conv.hpp"
#include "render/native/d3d12/device.hpp"
#include "render/native/d3d12/native_step.hpp"
#include "render/native/d3d12/shared_texture.hpp"
#include "render/native/d3d12/teardown.hpp"
#include "render/native/d3d12/upscale_bridge.hpp"
#include "render/native/native.hpp"
#include "render/post/upscale/native_upscale.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu_prof.hpp"

#include <wrl/client.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#endif

namespace aurora::render::detail {
namespace {
namespace fsr_pass = upscale::fsr_pass;
namespace compose = upscale::compose;
using wgpu::TextureFormat;

constexpr TextureFormat ColorFormat = TextureFormat::BGRA8Unorm;
constexpr TextureFormat DepthFormat = TextureFormat::Depth32Float;
constexpr uint32_t OutputWidth = 96;
constexpr uint32_t OutputHeight = 48;

constexpr const char* DepthFillWgsl = R"(
@group(0) @binding(0) var depth_source: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4f) -> @builtin(frag_depth) f32 {
  return textureLoad(depth_source, vec2i(position.xy), 0).r;
}
)";

struct Rgba {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 0;
};

struct Extent {
  uint32_t width = 0;
  uint32_t height = 0;
};

float halton(uint32_t index, uint32_t base) {
  float f = 1.f;
  float r = 0.f;
  while (index > 0) {
    f /= static_cast<float>(base);
    r += f * static_cast<float>(index % base);
    index /= base;
  }
  return r;
}

uint32_t nearest(uint32_t output, float scale, float jitter, uint32_t size) {
  const float position = (static_cast<float>(output) + 0.5f) * scale + jitter;
  return static_cast<uint32_t>(
      std::clamp(static_cast<int64_t>(std::floor(position)), int64_t{0}, static_cast<int64_t>(size) - 1));
}

class SceneComposeGpu : public ::testing::Test {
protected:
  void SetUp() override {
    const wgpu::BackendType backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    m_options = {.backend = backend, .efbSize = {OutputWidth, OutputHeight, 1}, .createEfb = true};
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      m_device.reset();
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    jobs::initialize(1);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    compose::testing::reset();
    ASSERT_TRUE(compose::prepare_pipelines(ComposeMode::Split, ColorFormat, DepthFormat));
    ASSERT_TRUE(compose::prepare_pipelines(ComposeMode::InPlace, ColorFormat, DepthFormat));
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      compose::shutdown();
      compose::testing::reset();
      shaders::shutdown();
      jobs::shutdown();
      gfx::set_blocking_pipelines(false);
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_depthFill = {};
    m_depthFillLayout = {};
    m_globals.reset();
    m_device.reset();
  }

  [[nodiscard]] wgpu::Texture color_texture(Extent size, TextureFormat format = ColorFormat) const {
    const wgpu::TextureDescriptor descriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = {size.width, size.height, 1},
        .format = format,
    };
    return m_device->device.CreateTexture(&descriptor);
  }

  [[nodiscard]] wgpu::Texture depth_texture(Extent size) const {
    const wgpu::TextureDescriptor descriptor{
        .usage =
            wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .size = {size.width, size.height, 1},
        .format = DepthFormat,
    };
    return m_device->device.CreateTexture(&descriptor);
  }

  void write_color(const wgpu::Texture& texture, const std::vector<Rgba>& pixels, bool bgra = true) const {
    std::vector<uint8_t> bytes(pixels.size() * 4);
    for (size_t i = 0; i < pixels.size(); ++i) {
      bytes[i * 4 + 0] = bgra ? pixels[i].b : pixels[i].r;
      bytes[i * 4 + 1] = pixels[i].g;
      bytes[i * 4 + 2] = bgra ? pixels[i].r : pixels[i].b;
      bytes[i * 4 + 3] = pixels[i].a;
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = texture.GetWidth() * 4,
                                             .rowsPerImage = texture.GetHeight()};
    const wgpu::Extent3D size{texture.GetWidth(), texture.GetHeight(), 1};
    m_device->queue.WriteTexture(&destination, bytes.data(), bytes.size(), &layout, &size);
  }

  [[nodiscard]] std::vector<Rgba> read_color(const wgpu::Texture& texture) const {
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    const uint32_t row = (width * 4 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<Rgba> out(size_t{width} * height);
    if (!words) {
      ADD_FAILURE() << "colour readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint8_t* p = bytes + size_t{y} * row + size_t{x} * 4;
        out[size_t{y} * width + x] = {.r = p[2], .g = p[1], .b = p[0], .a = p[3]};
      }
    }
    return out;
  }

  void fill_depth(const wgpu::Texture& target, const std::vector<float>& values) {
    const wgpu::Device& device = m_device->device;
    if (!m_depthFill) {
      const wgpu::BindGroupLayoutEntry entry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      };
      const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.entryCount = 1, .entries = &entry};
      m_depthFillLayout = device.CreateBindGroupLayout(&layoutDescriptor);
      const wgpu::PipelineLayoutDescriptor pipelineLayout{.bindGroupLayoutCount = 1,
                                                          .bindGroupLayouts = &m_depthFillLayout};
      const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = DepthFillWgsl}};
      const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
      const wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
      const wgpu::DepthStencilState depth{
          .format = DepthFormat,
          .depthWriteEnabled = wgpu::OptionalBool::True,
          .depthCompare = wgpu::CompareFunction::Always,
      };
      const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs_main"};
      const wgpu::RenderPipelineDescriptor descriptor{
          .layout = device.CreatePipelineLayout(&pipelineLayout),
          .vertex = {.module = module, .entryPoint = "vs_main"},
          .depthStencil = &depth,
          .fragment = &fragment,
      };
      m_depthFill = device.CreateRenderPipeline(&descriptor);
    }
    const wgpu::TextureDescriptor sourceDescriptor{
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .size = {target.GetWidth(), target.GetHeight(), 1},
        .format = TextureFormat::R32Float,
    };
    const wgpu::Texture source = device.CreateTexture(&sourceDescriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = source};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = target.GetWidth() * 4, .rowsPerImage = target.GetHeight()};
    m_device->queue.WriteTexture(&destination, values.data(), values.size() * sizeof(float), &layout,
                                 &sourceDescriptor.size);
    const wgpu::BindGroupEntry groupEntry{.binding = 0, .textureView = source.CreateView()};
    const wgpu::BindGroupDescriptor groupDescriptor{
        .layout = m_depthFillLayout, .entryCount = 1, .entries = &groupEntry};
    const wgpu::BindGroup group = device.CreateBindGroup(&groupDescriptor);
    const wgpu::RenderPassDepthStencilAttachment attachment{
        .view = target.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.f,
    };
    const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(m_depthFill);
    pass.SetBindGroup(0, group);
    pass.Draw(3);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  }

  [[nodiscard]] std::vector<float> read_depth(const wgpu::Texture& texture) const {
    return test::read_texture_mip(*m_device, texture, 0);
  }

  bool run(const SceneComposeData& data, Extent scene, Extent output, const compose::Targets& targets,
           const std::optional<compose::Source>& source = std::nullopt) {
    const compose::Params params =
        compose::make_params(data, {scene.width, scene.height}, {output.width, output.height});
    m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, data.uniformOffset, &params, sizeof(params));
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const bool encoded = compose::testing::encode(encoder, data, targets, source);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const std::optional<std::string> error = test::pop_validation_error(*m_device);
    EXPECT_EQ(error, std::nullopt);
    return encoded && !error;
  }

  struct Split {
    Extent scene;
    wgpu::Texture sceneColor;
    wgpu::Texture sceneDepth;
    wgpu::Texture color;
    wgpu::Texture depth;

    [[nodiscard]] compose::Targets targets() const {
      return {
          .sceneColor = sceneColor.CreateView(),
          .sceneDepth = sceneDepth.CreateView(),
          .colorTexture = color,
          .color = color.CreateView(),
          .colorFormat = ColorFormat,
          .depth = depth.CreateView(),
          .depthFormat = DepthFormat,
      };
    }
  };

  [[nodiscard]] Split split_targets(Extent scene) const {
    return {
        .scene = scene,
        .sceneColor = color_texture(scene),
        .sceneDepth = depth_texture(scene),
        .color = color_texture({OutputWidth, OutputHeight}),
        .depth = depth_texture({OutputWidth, OutputHeight}),
    };
  }

  static SceneComposeData split_data(Extent scene, float jitterX, float jitterY) {
    return {
        .mode = ComposeMode::Split,
        .ratio = {static_cast<float>(OutputWidth) / static_cast<float>(scene.width),
                  static_cast<float>(OutputHeight) / static_cast<float>(scene.height)},
        .jitterPx = {jitterX, jitterY},
    };
  }

  static float scale_x(Extent scene) { return static_cast<float>(scene.width) / static_cast<float>(OutputWidth); }

  static float scale_y(Extent scene) { return static_cast<float>(scene.height) / static_cast<float>(OutputHeight); }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  wgpu::BindGroupLayout m_depthFillLayout;
  wgpu::RenderPipeline m_depthFill;
};

constexpr std::array<Extent, 2> SceneSizes{{{48, 24}, {72, 36}}};
constexpr std::array<std::array<float, 2>, 3> Jitters{{{0.f, 0.f}, {0.3125f, -0.1875f}, {-0.4375f, 0.375f}}};

TEST_F(SceneComposeGpu, AlphaIsNearestSceneAlpha) {
  for (const Extent scene : SceneSizes) {
    const Split split = split_targets(scene);
    std::vector<Rgba> pixels(size_t{scene.width} * scene.height);
    for (uint32_t y = 0; y < scene.height; ++y) {
      for (uint32_t x = 0; x < scene.width; ++x) {
        pixels[size_t{y} * scene.width + x] = {
            .r = static_cast<uint8_t>(x * 4),
            .g = static_cast<uint8_t>(y * 4),
            .b = 128,
            .a = static_cast<uint8_t>((x * 7 + y * 13 + 5) & 255),
        };
      }
    }
    write_color(split.sceneColor, pixels);
    fill_depth(split.sceneDepth, std::vector<float>(pixels.size(), 0.5f));
    for (const auto& [jx, jy] : Jitters) {
      SCOPED_TRACE(std::to_string(scene.width) + " jitter " + std::to_string(jx) + "," + std::to_string(jy));
      ASSERT_TRUE(run(split_data(scene, jx, jy), scene, {OutputWidth, OutputHeight}, split.targets()));
      const std::vector<Rgba> out = read_color(split.color);
      uint32_t mismatches = 0;
      for (uint32_t py = 0; py < OutputHeight; ++py) {
        for (uint32_t px = 0; px < OutputWidth; ++px) {
          const uint32_t tx = nearest(px, scale_x(scene), jx, scene.width);
          const uint32_t ty = nearest(py, scale_y(scene), jy, scene.height);
          mismatches += out[size_t{py} * OutputWidth + px].a != pixels[size_t{ty} * scene.width + tx].a ? 1 : 0;
        }
      }
      EXPECT_EQ(mismatches, 0u);
    }
  }
}

TEST_F(SceneComposeGpu, DepthIsNearestSceneDepth) {
  for (const Extent scene : SceneSizes) {
    const Split split = split_targets(scene);
    std::vector<float> depth(size_t{scene.width} * scene.height);
    for (uint32_t y = 0; y < scene.height; ++y) {
      for (uint32_t x = 0; x < scene.width; ++x) {
        depth[size_t{y} * scene.width + x] = static_cast<float>((x * 5 + y * 11) % 97 + 1) / 128.f;
      }
    }
    write_color(split.sceneColor, std::vector<Rgba>(depth.size(), Rgba{.r = 1, .g = 2, .b = 3, .a = 4}));
    fill_depth(split.sceneDepth, depth);
    for (const auto& [jx, jy] : Jitters) {
      SCOPED_TRACE(std::to_string(scene.width) + " jitter " + std::to_string(jx) + "," + std::to_string(jy));
      ASSERT_TRUE(run(split_data(scene, jx, jy), scene, {OutputWidth, OutputHeight}, split.targets()));
      const std::vector<float> out = read_depth(split.depth);
      ASSERT_EQ(out.size(), size_t{OutputWidth} * OutputHeight);
      uint32_t mismatches = 0;
      for (uint32_t py = 0; py < OutputHeight; ++py) {
        for (uint32_t px = 0; px < OutputWidth; ++px) {
          const uint32_t tx = nearest(px, scale_x(scene), jx, scene.width);
          const uint32_t ty = nearest(py, scale_y(scene), jy, scene.height);
          mismatches += out[size_t{py} * OutputWidth + px] != depth[size_t{ty} * scene.width + tx] ? 1 : 0;
        }
      }
      EXPECT_EQ(mismatches, 0u);
    }
  }
}

TEST_F(SceneComposeGpu, JitterCompensatedEdge) {
  constexpr Extent Scene{48, 24};
  constexpr float Near = 0.75f;
  constexpr float Far = 0.25f;
  const Split split = split_targets(Scene);
  write_color(split.sceneColor, std::vector<Rgba>(size_t{Scene.width} * Scene.height, Rgba{.a = 255}));
  const auto edge_at = [](uint32_t row) { return 20.f + (static_cast<float>(row) + 0.5f) / Scene.height * 3.f; };
  const auto measure = [&](bool compensated) {
    std::vector<float> errors;
    for (uint32_t phase = 1; phase <= 8; ++phase) {
      const float jx = halton(phase, 2) - 0.5f;
      std::vector<float> depth(size_t{Scene.width} * Scene.height);
      for (uint32_t y = 0; y < Scene.height; ++y) {
        for (uint32_t x = 0; x < Scene.width; ++x) {
          depth[size_t{y} * Scene.width + x] = static_cast<float>(x) + 0.5f - jx < edge_at(y) ? Near : Far;
        }
      }
      fill_depth(split.sceneDepth, depth);
      EXPECT_TRUE(
          run(split_data(Scene, compensated ? jx : 0.f, 0.f), Scene, {OutputWidth, OutputHeight}, split.targets()));
      const std::vector<float> out = read_depth(split.depth);
      if (out.size() != size_t{OutputWidth} * OutputHeight) {
        ADD_FAILURE() << "depth readback failed";
        return errors;
      }
      for (uint32_t py = 0; py < OutputHeight; ++py) {
        uint32_t boundary = OutputWidth;
        for (uint32_t px = 0; px < OutputWidth; ++px) {
          if (out[size_t{py} * OutputWidth + px] != Near) {
            boundary = px;
            break;
          }
        }
        const float ideal = std::ceil(edge_at(py / 2) * 2.f - 0.5f);
        errors.push_back((static_cast<float>(boundary) - ideal) / 2.f);
      }
    }
    return errors;
  };
  const auto max_abs = [](const std::vector<float>& values) {
    float out = 0.f;
    for (const float value : values) {
      out = std::max(out, std::abs(value));
    }
    return out;
  };
  const auto mean = [](const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) {
      sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
  };
  const std::vector<float> compensated = measure(true);
  ASSERT_FALSE(compensated.empty());
  EXPECT_LE(max_abs(compensated), 0.5f);
  EXPECT_LE(std::abs(mean(compensated)), 0.1);
  const std::vector<float> control = measure(false);
  ASSERT_FALSE(control.empty());
  EXPECT_GT(max_abs(control), 0.5f);
}

TEST_F(SceneComposeGpu, DefaultSourceIsSceneColour) {
  constexpr Extent Scene{48, 24};
  const Split split = split_targets(Scene);
  std::vector<Rgba> pixels(size_t{Scene.width} * Scene.height);
  for (uint32_t y = 0; y < Scene.height; ++y) {
    for (uint32_t x = 0; x < Scene.width; ++x) {
      pixels[size_t{y} * Scene.width + x] = {
          .r = static_cast<uint8_t>(x * 4),
          .g = static_cast<uint8_t>(y * 8),
          .b = static_cast<uint8_t>(64 + x * 2 + y * 2),
          .a = static_cast<uint8_t>(x + y),
      };
    }
  }
  write_color(split.sceneColor, pixels);
  fill_depth(split.sceneDepth, std::vector<float>(pixels.size(), 0.5f));
  constexpr float Jx = 0.25f;
  constexpr float Jy = -0.125f;
  ASSERT_TRUE(run(split_data(Scene, Jx, Jy), Scene, {OutputWidth, OutputHeight}, split.targets()));
  EXPECT_EQ(compose::testing::counters().sourced, 0u);
  const std::vector<Rgba> out = read_color(split.color);
  const auto linear = [](float position, float step, float base, uint32_t size) {
    const float texel = std::clamp(position - 0.5f, 0.f, static_cast<float>(size - 1));
    return base + step * texel;
  };
  int worst = 0;
  for (uint32_t py = 0; py < OutputHeight; ++py) {
    for (uint32_t px = 0; px < OutputWidth; ++px) {
      const float qx = (static_cast<float>(px) + 0.5f) * scale_x(Scene) + Jx;
      const float qy = (static_cast<float>(py) + 0.5f) * scale_y(Scene) + Jy;
      const float tx = std::clamp(qx - 0.5f, 0.f, static_cast<float>(Scene.width - 1));
      const float ty = std::clamp(qy - 0.5f, 0.f, static_cast<float>(Scene.height - 1));
      const Rgba& got = out[size_t{py} * OutputWidth + px];
      worst = std::max(worst, std::abs(got.r - static_cast<int>(std::lround(linear(qx, 4.f, 0.f, Scene.width)))));
      worst = std::max(worst, std::abs(got.g - static_cast<int>(std::lround(linear(qy, 8.f, 0.f, Scene.height)))));
      worst = std::max(worst, std::abs(got.b - static_cast<int>(std::lround(64.f + 2.f * tx + 2.f * ty))));
      const uint32_t nx = nearest(px, scale_x(Scene), Jx, Scene.width);
      const uint32_t ny = nearest(py, scale_y(Scene), Jy, Scene.height);
      EXPECT_EQ(got.a, pixels[size_t{ny} * Scene.width + nx].a);
    }
  }
  EXPECT_LE(worst, 1);
}

TEST_F(SceneComposeGpu, InplaceWritesRgbOnly) {
  constexpr Extent Size{OutputWidth, OutputHeight};
  const wgpu::Texture target = color_texture(Size);
  const wgpu::Texture source = color_texture(Size, TextureFormat::RGBA8Unorm);
  std::vector<Rgba> base(size_t{Size.width} * Size.height);
  std::vector<Rgba> resolved(base.size());
  for (uint32_t y = 0; y < Size.height; ++y) {
    for (uint32_t x = 0; x < Size.width; ++x) {
      const size_t i = size_t{y} * Size.width + x;
      base[i] = {.r = 10, .g = 20, .b = 30, .a = static_cast<uint8_t>((x * 3 + y * 5) & 255)};
      resolved[i] = {.r = static_cast<uint8_t>(x), .g = static_cast<uint8_t>(y), .b = 200, .a = 77};
    }
  }
  write_color(target, base);
  write_color(source, resolved, false);
  const compose::Targets targets{.colorTexture = target, .color = target.CreateView(), .colorFormat = ColorFormat};
  const SceneComposeData data{.mode = ComposeMode::InPlace};
  ASSERT_TRUE(run(data, Size, Size, targets));
  const std::vector<Rgba> untouched = read_color(target);
  for (size_t i = 0; i < base.size(); ++i) {
    ASSERT_EQ(untouched[i].r, base[i].r) << i;
    ASSERT_EQ(untouched[i].a, base[i].a) << i;
  }
  ASSERT_TRUE(run(data, Size, Size, targets,
                  compose::Source{.view = source.CreateView(), .format = TextureFormat::RGBA8Unorm}));
  EXPECT_EQ(compose::testing::counters().sourced, 1u);
  const std::vector<Rgba> out = read_color(target);
  for (size_t i = 0; i < base.size(); ++i) {
    ASSERT_EQ(out[i].r, resolved[i].r) << i;
    ASSERT_EQ(out[i].g, resolved[i].g) << i;
    ASSERT_EQ(out[i].b, resolved[i].b) << i;
    ASSERT_EQ(out[i].a, base[i].a) << i;
  }
}

namespace fsr = upscale::fsr;

float half_to_float(uint16_t half) {
  const uint32_t sign = (half >> 15) & 1u;
  const uint32_t exponent = (half >> 10) & 31u;
  const uint32_t mantissa = half & 1023u;
  float value = 0.f;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa != 0 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
  } else {
    value = std::ldexp(static_cast<float>(mantissa | 1024u), static_cast<int>(exponent) - 25);
  }
  return sign != 0 ? -value : value;
}

uint16_t float_to_half(float value) {
  if (value == 0.f) {
    return 0;
  }
  int exponent = 0;
  const float fraction = std::frexp(std::fabs(value), &exponent);
  const uint16_t sign = value < 0.f ? 0x8000u : 0u;
  if (exponent < -13) {
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(std::lrint(std::fabs(value) * 16777216.f)));
  }
  const auto mantissa = static_cast<uint32_t>(std::lrint(fraction * 2048.f));
  const uint32_t bits = (static_cast<uint32_t>(exponent + 14) << 10) + (mantissa - 1024u);
  return static_cast<uint16_t>(sign | bits);
}

float round_half(float value) { return half_to_float(float_to_half(value)); }

std::array<float, 3> rgb_to_ycocg(const std::array<float, 3>& c) {
  return {0.25f * c[0] + 0.5f * c[1] + 0.25f * c[2], 0.5f * c[0] - 0.5f * c[2],
          -0.25f * c[0] + 0.5f * c[1] - 0.25f * c[2]};
}

std::array<float, 3> ycocg_to_rgb(const std::array<float, 3>& c) {
  return {c[0] + c[1] - c[2], c[0] + c[2], c[0] - c[1] - c[2]};
}

uint32_t mix_bits(uint32_t x, uint32_t y) {
  uint32_t h = x * 374761393u + y * 668265263u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return h ^ (h >> 16);
}

class FsrGpu : public ::testing::Test {
protected:
  struct Inputs {
    Extent size;
    std::vector<Rgba> color;
    std::vector<float> depth;
    std::vector<std::array<float, 2>> motion;
    std::vector<uint8_t> reactive;
  };

  void SetUp() override {
    const wgpu::BackendType backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    m_options = {.backend = backend, .efbSize = {64, 64, 1}, .createEfb = true};
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      m_device.reset();
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    if (!m_device->device.HasFeature(wgpu::FeatureName::TextureFormatsTier1)) {
      m_device.reset();
      GTEST_SKIP() << "FSR needs texture-formats-tier1";
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    jobs::initialize(1);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    m_state.locksR8 = m_device->device.HasFeature(wgpu::FeatureName::TextureFormatsTier2);
    RecordProperty("locks_r8", m_state.locksR8 ? 1 : 0);
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      m_state = {};
      fsr::shutdown();
      shaders::shutdown();
      jobs::shutdown();
      gfx::set_blocking_pipelines(false);
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_depthFill = {};
    m_depthFillLayout = {};
    m_globals.reset();
    m_device.reset();
  }

  [[nodiscard]] wgpu::Texture texture(Extent size, TextureFormat format) const {
    const wgpu::TextureDescriptor descriptor{
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc |
                 (format == DepthFormat ? wgpu::TextureUsage::RenderAttachment : wgpu::TextureUsage::None),
        .size = {size.width, size.height, 1},
        .format = format,
    };
    return m_device->device.CreateTexture(&descriptor);
  }

  void write(const wgpu::Texture& target, const void* data, uint32_t bytesPerTexel) const {
    const wgpu::TexelCopyTextureInfo destination{.texture = target};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = target.GetWidth() * bytesPerTexel,
                                             .rowsPerImage = target.GetHeight()};
    const wgpu::Extent3D size{target.GetWidth(), target.GetHeight(), 1};
    m_device->queue.WriteTexture(&destination, data, size_t{target.GetWidth()} * target.GetHeight() * bytesPerTexel,
                                 &layout, &size);
  }

  [[nodiscard]] std::vector<uint8_t> read_raw(const wgpu::Texture& source, uint32_t bytesPerTexel) const {
    const uint32_t width = source.GetWidth();
    const uint32_t height = source.GetHeight();
    const uint32_t row = (width * bytesPerTexel + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo from{.texture = source};
    const wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = row, .rowsPerImage = height}, .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&from, &to, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<uint8_t> out(size_t{width} * height * bytesPerTexel);
    if (!words) {
      ADD_FAILURE() << "texture readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(out.data() + size_t{y} * width * bytesPerTexel, bytes + size_t{y} * row, width * bytesPerTexel);
    }
    return out;
  }

  [[nodiscard]] std::vector<float> read_halves(const wgpu::Texture& source, uint32_t channels) const {
    const std::vector<uint8_t> bytes = read_raw(source, channels * 2);
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
      uint16_t half = 0;
      std::memcpy(&half, bytes.data() + i * 2, 2);
      out[i] = half_to_float(half);
    }
    return out;
  }

  [[nodiscard]] std::vector<float> read_floats(const wgpu::Texture& source, uint32_t channels) const {
    const std::vector<uint8_t> bytes = read_raw(source, channels * 4);
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  }

  [[nodiscard]] std::vector<float> read_reconstruction() const {
    const uint64_t size = m_state.recon.GetSize();
    const auto words = test::read_buffer(*m_device, m_state.recon, 0, size);
    std::vector<float> out(size / 4);
    if (words) {
      std::memcpy(out.data(), words->data(), size);
    }
    return out;
  }

  void fill_depth(const wgpu::Texture& target, const std::vector<float>& values) {
    const wgpu::Device& device = m_device->device;
    if (!m_depthFill) {
      const wgpu::BindGroupLayoutEntry entry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      };
      const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.entryCount = 1, .entries = &entry};
      m_depthFillLayout = device.CreateBindGroupLayout(&layoutDescriptor);
      const wgpu::PipelineLayoutDescriptor pipelineLayout{.bindGroupLayoutCount = 1,
                                                          .bindGroupLayouts = &m_depthFillLayout};
      const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = DepthFillWgsl}};
      const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
      const wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
      const wgpu::DepthStencilState depth{
          .format = DepthFormat,
          .depthWriteEnabled = wgpu::OptionalBool::True,
          .depthCompare = wgpu::CompareFunction::Always,
      };
      const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs_main"};
      const wgpu::RenderPipelineDescriptor descriptor{
          .layout = device.CreatePipelineLayout(&pipelineLayout),
          .vertex = {.module = module, .entryPoint = "vs_main"},
          .depthStencil = &depth,
          .fragment = &fragment,
      };
      m_depthFill = device.CreateRenderPipeline(&descriptor);
    }
    const wgpu::Texture source = texture({target.GetWidth(), target.GetHeight()}, TextureFormat::R32Float);
    write(source, values.data(), 4);
    const wgpu::BindGroupEntry groupEntry{.binding = 0, .textureView = source.CreateView()};
    const wgpu::BindGroupDescriptor groupDescriptor{
        .layout = m_depthFillLayout, .entryCount = 1, .entries = &groupEntry};
    const wgpu::BindGroup group = device.CreateBindGroup(&groupDescriptor);
    const wgpu::RenderPassDepthStencilAttachment attachment{
        .view = target.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.f,
    };
    const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(m_depthFill);
    pass.SetBindGroup(0, group);
    pass.Draw(3);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  }

  [[nodiscard]] static Inputs flat_inputs(Extent size) {
    const size_t count = size_t{size.width} * size.height;
    return {
        .size = size,
        .color = std::vector<Rgba>(count, Rgba{.r = 128, .g = 128, .b = 128, .a = 255}),
        .depth = std::vector<float>(count, 0.5f),
        .motion = std::vector<std::array<float, 2>>(count, {0.f, 0.f}),
        .reactive = std::vector<uint8_t>(count, 0),
    };
  }

  bool run(const Inputs& inputs, uint32_t framesSinceReset, fsr::Stage last, std::array<float, 2> jitter = {}) {
    fsr::testing::Inputs gpu{
        .color = texture(inputs.size, ColorFormat),
        .depth = texture(inputs.size, DepthFormat),
        .motion = texture(inputs.size, TextureFormat::RG16Float),
        .reactive = texture(inputs.size, TextureFormat::R8Unorm),
        .frame =
            {
                .jitter = {jitter[0], jitter[1]},
                .prevJitter = {jitter[0], jitter[1]},
                .camera = {.nearZ = 1.f, .farZ = 1000.f, .fovY = 1.f, .viewToMeters = 0.01f},
                .framesSinceReset = framesSinceReset,
            },
    };
    std::vector<uint8_t> bgra(inputs.color.size() * 4);
    for (size_t i = 0; i < inputs.color.size(); ++i) {
      bgra[i * 4 + 0] = inputs.color[i].b;
      bgra[i * 4 + 1] = inputs.color[i].g;
      bgra[i * 4 + 2] = inputs.color[i].r;
      bgra[i * 4 + 3] = inputs.color[i].a;
    }
    write(gpu.color, bgra.data(), 4);
    std::vector<uint16_t> motion(inputs.motion.size() * 2);
    for (size_t i = 0; i < inputs.motion.size(); ++i) {
      motion[i * 2 + 0] = float_to_half(inputs.motion[i][0]);
      motion[i * 2 + 1] = float_to_half(inputs.motion[i][1]);
    }
    write(gpu.motion, motion.data(), 4);
    write(gpu.reactive, inputs.reactive.data(), 1);
    fill_depth(gpu.depth, inputs.depth);
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const bool ran = fsr::testing::run(m_state, gpu, last);
    const std::optional<std::string> error = test::pop_validation_error(*m_device);
    EXPECT_EQ(error, std::nullopt);
    return ran && !error;
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  fsr::testing::State m_state;
  wgpu::BindGroupLayout m_depthFillLayout;
  wgpu::RenderPipeline m_depthFill;
};

TEST_F(FsrGpu, PrepareInputsTakesClosestReversedDepth) {
  constexpr Extent Size{40, 24};
  Inputs inputs = flat_inputs(Size);
  for (uint32_t y = 0; y < Size.height; ++y) {
    for (uint32_t x = 0; x < Size.width; ++x) {
      const size_t i = size_t{y} * Size.width + x;
      inputs.depth[i] = 0.05f + 0.9f * static_cast<float>(mix_bits(x, y) % 4096) / 4096.f;
      inputs.motion[i] = {static_cast<float>(static_cast<int>(x) - 20) / 256.f,
                          static_cast<float>(static_cast<int>(y) - 12) / 256.f};
    }
  }
  ASSERT_TRUE(run(inputs, 0, fsr::Stage::PrepareInputs));
  const std::vector<float> dilated = read_floats(m_state.dilatedDepth, 1);
  const std::vector<float> motion = read_halves(m_state.dilatedMotion, 2);
  constexpr std::array<std::array<int, 2>, 8> Offsets{
      {{1, 0}, {0, 1}, {0, -1}, {-1, 0}, {-1, 1}, {1, 1}, {-1, -1}, {1, -1}}};
  for (int y = 0; y < static_cast<int>(Size.height); ++y) {
    for (int x = 0; x < static_cast<int>(Size.width); ++x) {
      const size_t i = static_cast<size_t>(y) * Size.width + static_cast<size_t>(x);
      float nearest = inputs.depth[i];
      size_t from = i;
      for (const auto& [dx, dy] : Offsets) {
        const int qx = x + dx;
        const int qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= static_cast<int>(Size.width) || qy >= static_cast<int>(Size.height)) {
          continue;
        }
        const size_t q = static_cast<size_t>(qy) * Size.width + static_cast<size_t>(qx);
        if (inputs.depth[q] > nearest) {
          nearest = inputs.depth[q];
          from = q;
        }
      }
      ASSERT_EQ(dilated[i], nearest) << x << "," << y;
      EXPECT_NEAR(motion[i * 2 + 0], -0.5f * round_half(inputs.motion[from][0]), 1e-4f) << x << "," << y;
      EXPECT_NEAR(motion[i * 2 + 1], 0.5f * round_half(inputs.motion[from][1]), 1e-4f) << x << "," << y;
    }
  }
}

TEST_F(FsrGpu, ReconstructionBufferClearedEachFrame) {
  constexpr Extent Size{32, 32};
  Inputs inputs = flat_inputs(Size);
  std::ranges::fill(inputs.depth, 0.9f);
  ASSERT_TRUE(run(inputs, 0, fsr::Stage::PrepareInputs));
  for (const float value : read_reconstruction()) {
    ASSERT_EQ(value, 0.9f);
  }
  for (uint32_t y = 0; y < Size.height; ++y) {
    for (uint32_t x = 0; x < Size.width; ++x) {
      inputs.depth[size_t{y} * Size.width + x] = 0.2f + 0.1f * static_cast<float>(mix_bits(x, y) % 256) / 256.f;
    }
  }
  ASSERT_TRUE(run(inputs, 1, fsr::Stage::PrepareInputs));
  const std::vector<float> reconstructed = read_reconstruction();
  const std::vector<float> dilated = read_floats(m_state.dilatedDepth, 1);
  ASSERT_EQ(reconstructed.size(), dilated.size());
  for (size_t i = 0; i < dilated.size(); ++i) {
    ASSERT_EQ(reconstructed[i], dilated[i]) << i;
    ASSERT_LT(reconstructed[i], 0.9f) << i;
  }
}

TEST_F(FsrGpu, InvalidMotionBecomesReactive) {
  constexpr Extent Size{32, 32};
  Inputs inputs = flat_inputs(Size);
  for (uint32_t y = 8; y < 16; ++y) {
    for (uint32_t x = 8; x < 16; ++x) {
      inputs.motion[size_t{y} * Size.width + x] = {4.f, 4.f};
    }
  }
  ASSERT_TRUE(run(inputs, 0, fsr::Stage::PrepareReactivity));
  ASSERT_TRUE(run(inputs, 1, fsr::Stage::PrepareReactivity));
  const std::vector<uint8_t> masks = read_raw(m_state.dilatedReactive, 4);
  const std::vector<float> motion = read_halves(m_state.dilatedMotion, 2);
  for (uint32_t y = 0; y < Size.height; ++y) {
    for (uint32_t x = 0; x < Size.width; ++x) {
      const size_t i = size_t{y} * Size.width + x;
      const uint8_t change = masks[i * 4 + 2];
      if (x >= 8 && x < 16 && y >= 8 && y < 16) {
        EXPECT_EQ(motion[i * 2], 2.f) << x << "," << y;
        EXPECT_GE(change, 229) << x << "," << y;
      } else if (x + 2 < 8 || x >= 18 || y + 2 < 8 || y >= 18) {
        EXPECT_LE(change, 3) << x << "," << y;
      }
    }
  }
}

TEST_F(FsrGpu, LumaPyramidFrameInfoMatchesCpu) {
  constexpr Extent Size{256, 256};
  Inputs inputs = flat_inputs(Size);
  double lumaSum = 0.0;
  for (uint32_t y = 0; y < Size.height; ++y) {
    for (uint32_t x = 0; x < Size.width; ++x) {
      const Rgba color{
          .r = static_cast<uint8_t>(x), .g = static_cast<uint8_t>(y), .b = static_cast<uint8_t>(x ^ y), .a = 255};
      inputs.color[size_t{y} * Size.width + x] = color;
      const float luma = 0.2126f * color.r / 255.f + 0.7152f * color.g / 255.f + 0.0722f * color.b / 255.f;
      lumaSum += round_half(luma);
    }
  }
  ASSERT_TRUE(run(inputs, 0, fsr::Stage::LumaPyramidB));
  const std::vector<float> info = read_floats(m_state.frameInfo[m_state.current()], 4);
  ASSERT_EQ(info.size(), 4u);
  const double averageLuma = lumaSum / (double{Size.width} * Size.height);
  const float deltaTime = fsr::NominalFrameMs / 1000.f;
  const float logLuma = std::max(0.f, 1.f + (6.10e-05f - 1.f) * (1.f - std::exp(-deltaTime)));
  const float exposure = 1.f / ((78.f / (0.65f * 100.f)) * std::pow(2.f, std::log2(std::exp(logLuma) * 100.f / 12.5f)));
  EXPECT_NEAR(info[2], averageLuma, 2e-3);
  EXPECT_NEAR(info[1], logLuma, 1e-4f);
  EXPECT_NEAR(info[0], exposure, exposure * 1e-3f);
}

TEST_F(FsrGpu, ResetEqualsLanczosUpsample) {
  constexpr Extent Render{32, 24};
  constexpr Extent Output{64, 48};
  constexpr std::array<float, 2> Jitter{0.25f, -0.125f};
  Inputs inputs = flat_inputs(Render);
  for (uint32_t y = 0; y < Render.height; ++y) {
    for (uint32_t x = 0; x < Render.width; ++x) {
      inputs.color[size_t{y} * Render.width + x] = {
          .r = static_cast<uint8_t>(x * 8),
          .g = static_cast<uint8_t>(y * 10),
          .b = static_cast<uint8_t>(mix_bits(x, y) & 255),
          .a = 255,
      };
    }
  }
  m_state.output = {Output.width, Output.height};
  ASSERT_TRUE(run(inputs, 0, fsr::Stage::Accumulate, Jitter));
  const std::vector<float> upscaled = read_halves(m_state.upscaled[m_state.current()], 4);
  const auto load = [&](int x, int y) {
    x = std::clamp(x, 0, static_cast<int>(Render.width) - 1);
    y = std::clamp(y, 0, static_cast<int>(Render.height) - 1);
    const Rgba c = inputs.color[static_cast<size_t>(y) * Render.width + static_cast<size_t>(x)];
    return rgb_to_ycocg({c.r / 255.f, c.g / 255.f, c.b / 255.f});
  };
  for (uint32_t py = 0; py < Output.height; ++py) {
    for (uint32_t px = 0; px < Output.width; ++px) {
      const float outX = (static_cast<float>(px) + 0.5f) * 0.5f;
      const float outY = (static_cast<float>(py) + 0.5f) * 0.5f;
      const int inX = static_cast<int>(std::floor(outX));
      const int inY = static_cast<int>(std::floor(outY));
      const float unjitteredX = static_cast<float>(inX) + 0.5f - Jitter[0];
      const float unjitteredY = static_cast<float>(inY) + 0.5f - Jitter[1];
      const bool flipCol = unjitteredX > outX;
      const bool flipRow = unjitteredY > outY;
      const int left = flipCol ? -2 : -1;
      const int top = flipRow ? -2 : -1;
      std::array<float, 3> sum{};
      float weightSum = 0.f;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          const int cx = left + (flipCol ? 3 - col : col);
          const int cy = top + (flipRow ? 3 - row : row);
          const float dx = unjitteredX - outX + static_cast<float>(cx);
          const float dy = unjitteredY - outY + static_cast<float>(cy);
          const int sx = inX + cx;
          const int sy = inY + cy;
          const bool onScreen =
              sx >= 0 && sy >= 0 && sx < static_cast<int>(Render.width) && sy < static_cast<int>(Render.height);
          const float weight = onScreen ? std::exp(-2.3f * (dx * dx + dy * dy)) : 0.f;
          const std::array<float, 3> sample = load(sx, sy);
          for (size_t c = 0; c < 3; ++c) {
            sum[c] += sample[c] * weight;
          }
          weightSum += weight;
        }
      }
      const std::array<float, 3> rgb = ycocg_to_rgb({sum[0] / weightSum, sum[1] / weightSum, sum[2] / weightSum});
      const size_t i = (size_t{py} * Output.width + px) * 4;
      for (size_t c = 0; c < 3; ++c) {
        ASSERT_NEAR(upscaled[i + c], std::max(0.f, rgb[c]), 2e-3f) << px << "," << py << " " << c;
      }
    }
  }
}

TEST_F(FsrGpu, StorageTexturesPerStageAtMostFour) {
  for (const bool locksR8 : {false, true}) {
    for (size_t i = 0; i < fsr::StageCount; ++i) {
      const auto stage = static_cast<fsr::Stage>(i);
      const std::span<const fsr::testing::Binding> bindings = fsr::testing::layout(stage, locksR8);
      const auto storage = std::ranges::count_if(bindings, [](const fsr::testing::Binding& binding) {
        return binding.kind == fsr::testing::Kind::StorageWrite || binding.kind == fsr::testing::Kind::StorageReadWrite;
      });
      EXPECT_LE(storage, fsr::MaxStorageTextures) << fsr::shader_id(stage);
      EXPECT_FALSE(bindings.empty()) << fsr::shader_id(stage);
    }
  }
  ASSERT_TRUE(run(flat_inputs({64, 64}), 0, fsr::Stage::Accumulate));
  if (m_state.locksR8) {
    m_state = {};
    ASSERT_TRUE(run(flat_inputs({64, 64}), 0, fsr::Stage::Accumulate));
  }
}
TEST_F(SceneComposeGpu, RcasZeroIsPassthrough) {
  constexpr Extent Scene{48, 24};
  const Split split = split_targets(Scene);
  std::vector<Rgba> scene(size_t{Scene.width} * Scene.height);
  for (uint32_t y = 0; y < Scene.height; ++y) {
    for (uint32_t x = 0; x < Scene.width; ++x) {
      scene[size_t{y} * Scene.width + x] = {.r = 9, .g = 9, .b = 9, .a = static_cast<uint8_t>((x * 11 + y * 7) & 255)};
    }
  }
  write_color(split.sceneColor, scene);
  fill_depth(split.sceneDepth, std::vector<float>(scene.size(), 0.5f));
  const wgpu::Texture source = color_texture({OutputWidth, OutputHeight}, TextureFormat::RGBA8Unorm);
  std::vector<Rgba> resolved(size_t{OutputWidth} * OutputHeight);
  for (uint32_t y = 0; y < OutputHeight; ++y) {
    for (uint32_t x = 0; x < OutputWidth; ++x) {
      const uint32_t bits = mix_bits(x, y);
      resolved[size_t{y} * OutputWidth + x] = {.r = static_cast<uint8_t>(bits),
                                               .g = static_cast<uint8_t>(bits >> 8),
                                               .b = static_cast<uint8_t>(bits >> 16),
                                               .a = 255};
    }
  }
  write_color(source, resolved, false);
  constexpr float Jx = 0.25f;
  constexpr float Jy = -0.125f;
  for (const float strength : {0.f, 1.f}) {
    SCOPED_TRACE(strength);
    SceneComposeData data = split_data(Scene, Jx, Jy);
    data.rcasStrength = strength;
    ASSERT_TRUE(run(data, Scene, {OutputWidth, OutputHeight}, split.targets(),
                    compose::Source{.view = source.CreateView(), .format = TextureFormat::RGBA8Unorm}));
    const std::vector<Rgba> out = read_color(split.color);
    size_t changed = 0;
    for (uint32_t py = 0; py < OutputHeight; ++py) {
      for (uint32_t px = 0; px < OutputWidth; ++px) {
        const size_t i = size_t{py} * OutputWidth + px;
        changed += out[i].r != resolved[i].r || out[i].g != resolved[i].g || out[i].b != resolved[i].b ? 1 : 0;
        const uint32_t nx = nearest(px, scale_x(Scene), Jx, Scene.width);
        const uint32_t ny = nearest(py, scale_y(Scene), Jy, Scene.height);
        ASSERT_EQ(out[i].a, scene[size_t{ny} * Scene.width + nx].a) << px << "," << py;
      }
    }
    if (strength == 0.f) {
      EXPECT_EQ(changed, 0u);
    } else {
      EXPECT_GT(changed, resolved.size() / 2);
    }
  }
  EXPECT_EQ(compose::testing::counters().sourced, 2u);
}

class FsrGraphGpu {
public:
  using Scene = std::function<Rgba(float x, float y)>;

  FsrGraphGpu(test::TestDevice& device, const test::TestDeviceOptions& options)
  : m_device(device), rig(device, options) {}

  void clear_targets(float motionX, float motionY, float reactive) const {
    const wgpu::CommandEncoder encoder = m_device.device.CreateCommandEncoder();
    const auto clear = [&](const wgpu::TextureView& view, const wgpu::Color& value) {
      const wgpu::RenderPassColorAttachment attachment{
          .view = view,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = value,
      };
      const wgpu::RenderPassDescriptor descriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
      encoder.BeginRenderPass(&descriptor).End();
    };
    clear(webgpu::g_motionBuffer.view, {motionX, motionY, 0.0, 0.0});
    clear(webgpu::g_reactiveBuffer.view, {reactive, 0.0, 0.0, 0.0});
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(m_device, commands));
  }

  std::vector<Rgba> frame(const Scene& scene, std::array<float, 2> motion = {}, float reactive = 0.f) {
    const StageData post = rig.build(++m_frame);
    jitter = post.sceneHints != nullptr ? std::array{post.sceneHints->jitterPx[0], post.sceneHints->jitterPx[1]}
                                        : std::array{0.f, 0.f};
    const wgpu::Texture target = webgpu::scene_color().texture;
    const uint32_t width = target.GetWidth();
    const uint32_t height = target.GetHeight();
    input.assign(size_t{width} * height, Rgba{});
    std::vector<uint8_t> bgra(input.size() * 4);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const size_t i = size_t{y} * width + x;
        input[i] = scene(static_cast<float>(x) + 0.5f - jitter[0], static_cast<float>(y) + 0.5f - jitter[1]);
        bgra[i * 4 + 0] = input[i].b;
        bgra[i * 4 + 1] = input[i].g;
        bgra[i * 4 + 2] = input[i].r;
        bgra[i * 4 + 3] = input[i].a;
      }
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = target};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device.queue.WriteTexture(&destination, bgra.data(), bgra.size(), &layout, &size);
    clear_targets(motion[0], motion[1], reactive);
    EXPECT_EQ(rig.encode(post, true), std::nullopt);
    return read_output();
  }

  [[nodiscard]] std::vector<Rgba> read_output() const {
    const wgpu::Texture& texture = webgpu::g_frameBuffer.texture;
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    const uint32_t row = (width * 4 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device.device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device.device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(m_device, commands));
    const auto words = test::read_buffer(m_device, buffer, 0, uint64_t{row} * height);
    std::vector<Rgba> out(size_t{width} * height);
    if (!words) {
      ADD_FAILURE() << "colour readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint8_t* p = bytes + size_t{y} * row + size_t{x} * 4;
        out[size_t{y} * width + x] = {.r = p[2], .g = p[1], .b = p[0], .a = p[3]};
      }
    }
    return out;
  }

  test::TestDevice& m_device;
  fsr_rig::Rig rig;
  uint64_t m_frame = 0;
  std::array<float, 2> jitter{};
  std::vector<Rgba> input;
};

struct Errors {
  double mean = 0.0;
  int max = 0;
};

Errors rgb_errors(const std::vector<Rgba>& got, const std::vector<Rgba>& want, uint32_t width, uint32_t border,
                  uint32_t limitX = UINT32_MAX) {
  Errors out;
  size_t count = 0;
  const auto height = static_cast<uint32_t>(got.size() / width);
  for (uint32_t y = border; y + border < height; ++y) {
    for (uint32_t x = border; x + border < width && x < limitX; ++x) {
      const Rgba& a = got[size_t{y} * width + x];
      const Rgba& b = want[size_t{y} * width + x];
      for (const int d : {std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)}) {
        out.max = std::max(out.max, d);
        out.mean += d;
        ++count;
      }
    }
  }
  out.mean = count != 0 ? out.mean / static_cast<double>(count) : 0.0;
  return out;
}

Rgba noise(uint32_t x, uint32_t y, uint32_t seed) {
  const uint32_t bits = mix_bits(x + seed * 977u, y + seed * 131u);
  return {.r = static_cast<uint8_t>(bits),
          .g = static_cast<uint8_t>(bits >> 8),
          .b = static_cast<uint8_t>(bits >> 16),
          .a = static_cast<uint8_t>(bits >> 24)};
}

std::vector<Rgba> reset_reference(test::TestDevice& device, const test::TestDeviceOptions& options,
                                  const FsrGraphGpu::Scene& scene) {
  FsrGraphGpu gpu{device, options};
  EXPECT_TRUE(gpu.rig.ok());
  gpu.rig.install(fsr_rig::fsr_settings(UpscalerQuality::NativeAa));
  std::vector<Rgba> out = gpu.frame(scene);
  EXPECT_TRUE(fsr_pass::testing::last_frame().reset);
  return out;
}

TEST_F(FsrGpu, StaticSceneConvergesToJitterMean) {
  FsrGraphGpu gpu{*m_device, m_options};
  ASSERT_TRUE(gpu.rig.ok());
  scene::motion::testing::set_jitter_env(scene::motion::testing::JitterEnv::On);
  gpu.rig.install(fsr_rig::fsr_settings(UpscalerQuality::NativeAa));
  ASSERT_EQ(temporal_resolve::jitter_phases(), 8u);
  const FsrGraphGpu::Scene scene = [](float x, float y) {
    const auto value = [&](float phase) {
      return static_cast<uint8_t>(std::lround(128.0 + 90.0 * std::sin(x * 0.5 + phase) * std::cos(y * 0.2)));
    };
    return Rgba{.r = value(0.f), .g = value(1.f), .b = value(2.f), .a = 255};
  };
  constexpr uint32_t Size = 64;
  std::vector<std::array<double, 3>> sum(size_t{Size} * Size);
  std::vector<Rgba> output;
  std::vector<std::vector<Rgba>> cycle;
  bool jittered = false;
  for (uint32_t frame = 1; frame <= 24; ++frame) {
    output = gpu.frame(scene);
    jittered = jittered || gpu.jitter[0] != 0.f || gpu.jitter[1] != 0.f;
    if (frame > 16) {
      cycle.push_back(gpu.input);
      for (size_t i = 0; i < sum.size(); ++i) {
        sum[i][0] += gpu.input[i].r;
        sum[i][1] += gpu.input[i].g;
        sum[i][2] += gpu.input[i].b;
      }
    }
  }
  ASSERT_TRUE(jittered);
  std::vector<Rgba> mean(sum.size());
  for (size_t i = 0; i < sum.size(); ++i) {
    mean[i] = {.r = static_cast<uint8_t>(std::lround(sum[i][0] / 8.0)),
               .g = static_cast<uint8_t>(std::lround(sum[i][1] / 8.0)),
               .b = static_cast<uint8_t>(std::lround(sum[i][2] / 8.0)),
               .a = 255};
  }
  const Errors converged = rgb_errors(output, mean, Size, 4);
  Errors raw;
  for (const std::vector<Rgba>& input : cycle) {
    const Errors errors = rgb_errors(input, mean, Size, 4);
    raw.max = std::max(raw.max, errors.max);
    raw.mean += errors.mean / static_cast<double>(cycle.size());
  }
  RecordProperty("converged_max", converged.max);
  RecordProperty("converged_mean_x1000", static_cast<int>(converged.mean * 1000.0));
  RecordProperty("raw_max", raw.max);
  RecordProperty("raw_mean_x1000", static_cast<int>(raw.mean * 1000.0));
  EXPECT_LE(converged.max, 6);
  EXPECT_LT(converged.mean, 0.5 * raw.mean);
  EXPECT_EQ(fsr_pass::testing::counters().resets, 1u);
  EXPECT_EQ(fsr_pass::testing::counters().skipped, 0u);
}

TEST_F(FsrGpu, ReactiveOneTakesCurrent) {
  const FsrGraphGpu::Scene first = [](float, float) { return Rgba{.r = 200, .g = 40, .b = 90, .a = 255}; };
  const FsrGraphGpu::Scene second = [](float x, float y) {
    return Rgba{.r = static_cast<uint8_t>(20 + 3 * static_cast<int>(x)),
                .g = static_cast<uint8_t>(230 - 3 * static_cast<int>(y)),
                .b = 160,
                .a = 255};
  };
  const std::vector<Rgba> reference = reset_reference(*m_device, m_options, second);
  std::array<Errors, 2> errors{};
  for (const float reactive : {0.f, 1.f}) {
    FsrGraphGpu gpu{*m_device, m_options};
    ASSERT_TRUE(gpu.rig.ok());
    gpu.rig.install(fsr_rig::fsr_settings(UpscalerQuality::NativeAa));
    for (int i = 0; i < 8; ++i) {
      gpu.frame(first);
    }
    errors[reactive > 0.f ? 1 : 0] = rgb_errors(gpu.frame(second, {}, reactive), reference, 64, 0);
    EXPECT_FALSE(fsr_pass::testing::last_frame().reset);
  }
  RecordProperty("control_mean_x1000", static_cast<int>(errors[0].mean * 1000.0));
  RecordProperty("control_max", errors[0].max);
  RecordProperty("reactive_mean_x1000", static_cast<int>(errors[1].mean * 1000.0));
  RecordProperty("reactive_max", errors[1].max);
  EXPECT_LE(errors[1].max, 1);
  EXPECT_LT(errors[1].mean, errors[0].mean);
}

TEST_F(FsrGpu, OffscreenMotionRejectsHistory) {
  const FsrGraphGpu::Scene first = [](float x, float y) {
    return noise(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 3);
  };
  const FsrGraphGpu::Scene second = [](float x, float y) {
    return noise(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 4);
  };
  const std::vector<Rgba> reference = reset_reference(*m_device, m_options, second);
  FsrGraphGpu gpu{*m_device, m_options};
  ASSERT_TRUE(gpu.rig.ok());
  gpu.rig.install(fsr_rig::fsr_settings(UpscalerQuality::NativeAa));
  for (int i = 0; i < 6; ++i) {
    gpu.frame(first);
  }
  const Errors still = rgb_errors(gpu.frame(second), reference, 64, 0);
  for (int i = 0; i < 6; ++i) {
    gpu.frame(first);
  }
  const Errors invalid = rgb_errors(gpu.frame(second, {4.f, 4.f}), reference, 64, 0);
  for (int i = 0; i < 6; ++i) {
    gpu.frame(first);
  }
  const Errors offscreen = rgb_errors(gpu.frame(second, {1.5f, 0.f}), reference, 64, 0, 64 * 3 / 4 - 1);
  RecordProperty("still_mean_x1000", static_cast<int>(still.mean * 1000.0));
  RecordProperty("invalid_mean_x1000", static_cast<int>(invalid.mean * 1000.0));
  RecordProperty("invalid_max", invalid.max);
  RecordProperty("offscreen_mean_x1000", static_cast<int>(offscreen.mean * 1000.0));
  RecordProperty("offscreen_max", offscreen.max);
  EXPECT_LT(invalid.mean, 0.5 * still.mean);
  EXPECT_LT(offscreen.mean, 0.5 * still.mean);
  EXPECT_EQ(fsr_pass::testing::counters().resets, 1u);
  EXPECT_EQ(fsr_pass::testing::counters().skipped, 0u);
}

TEST_F(FsrGpu, LeavesEfbAlphaUntouched) {
  for (const UpscalerQuality quality : {UpscalerQuality::NativeAa, UpscalerQuality::Quality}) {
    for (const float sharpness : {0.f, 1.f}) {
      SCOPED_TRACE(std::to_string(static_cast<int>(quality)) + " sharpness " + std::to_string(sharpness));
      FsrGraphGpu gpu{*m_device, m_options};
      ASSERT_TRUE(gpu.rig.ok());
      scene::motion::testing::set_jitter_env(scene::motion::testing::JitterEnv::On);
      gpu.rig.install(fsr_rig::fsr_settings(quality, sharpness));
      for (uint32_t seed = 10; seed < 14; ++seed) {
        const std::vector<Rgba> output = gpu.frame([seed](float x, float y) {
          return noise(static_cast<uint32_t>(std::floor(x)), static_cast<uint32_t>(std::floor(y)), seed);
        });
        const wgpu::Texture scene = webgpu::scene_color().texture;
        const uint32_t sceneWidth = scene.GetWidth();
        const uint32_t sceneHeight = scene.GetHeight();
        const float scaleX = static_cast<float>(sceneWidth) / 64.f;
        const float scaleY = static_cast<float>(sceneHeight) / 64.f;
        size_t changed = 0;
        for (uint32_t py = 0; py < 64; ++py) {
          for (uint32_t px = 0; px < 64; ++px) {
            const bool split = sceneWidth != 64;
            const uint32_t sx = split ? nearest(px, scaleX, gpu.jitter[0], sceneWidth) : px;
            const uint32_t sy = split ? nearest(py, scaleY, gpu.jitter[1], sceneHeight) : py;
            changed += output[size_t{py} * 64 + px].a != gpu.input[size_t{sy} * sceneWidth + sx].a ? 1 : 0;
          }
        }
        EXPECT_EQ(changed, 0u) << "seed " << seed;
      }
      EXPECT_EQ(fsr_pass::testing::counters().skipped, 0u);
    }
  }
}

#if AURORA_ENABLE_NATIVE_D3D12
TEST(NativeD3D12Gpu, DeviceAndQueueComeFromDawn) {
  namespace d3d12 = native::d3d12;
  using Microsoft::WRL::ComPtr;
  const test::TestDeviceOptions options{
      .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
  };
  std::string error;
  const auto device = test::create_test_device(options, error);
  if (!device) {
    GTEST_SKIP() << "no adapter: " << error;
  }
  if (device->adapterInfo.backendType != wgpu::BackendType::D3D12) {
    EXPECT_FALSE(d3d12::initialize(device->device, device->adapter));
    EXPECT_FALSE(d3d12::available());
    GTEST_SKIP() << "needs the D3D12 backend";
  }
  ASSERT_TRUE(d3d12::initialize(device->device, device->adapter));
  ASSERT_TRUE(d3d12::available());
  ID3D12Device* native = d3d12::device();
  ASSERT_NE(native, nullptr);
  ASSERT_NE(d3d12::queue(), nullptr);
  ComPtr<ID3D12Device> queueDevice;
  ASSERT_TRUE(SUCCEEDED(d3d12::queue()->GetDevice(IID_PPV_ARGS(&queueDevice))));
  ComPtr<IUnknown> nativeIdentity;
  ComPtr<IUnknown> queueIdentity;
  ASSERT_TRUE(SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&nativeIdentity))));
  ASSERT_TRUE(SUCCEEDED(queueDevice->QueryInterface(IID_PPV_ARGS(&queueIdentity))));
  EXPECT_EQ(nativeIdentity.Get(), queueIdentity.Get());
  const DXGI_ADAPTER_DESC1& desc = d3d12::adapter_desc();
  EXPECT_EQ(desc.VendorId, device->adapterInfo.vendorID);
  EXPECT_EQ(desc.DeviceId, device->adapterInfo.deviceID);
  const LUID luid = native->GetAdapterLuid();
  EXPECT_EQ(luid.LowPart, desc.AdapterLuid.LowPart);
  EXPECT_EQ(luid.HighPart, desc.AdapterLuid.HighPart);
  EXPECT_NE(d3d12::driver_version(), 0u);
  d3d12::shutdown();
  EXPECT_FALSE(d3d12::available());
  EXPECT_EQ(d3d12::device(), nullptr);
  EXPECT_EQ(d3d12::queue(), nullptr);
}

constexpr uint32_t SharedWidth = 64;
constexpr uint32_t SharedHeight = 32;
constexpr uint32_t SharedRowBytes = SharedWidth * 4;

std::vector<uint32_t> texel_pattern(uint32_t seed) {
  std::vector<uint32_t> out(size_t{SharedWidth} * SharedHeight);
  for (uint32_t y = 0; y < SharedHeight; ++y) {
    for (uint32_t x = 0; x < SharedWidth; ++x) {
      out[size_t{y} * SharedWidth + x] = mix_bits(x + seed * 977u, y + seed * 131u);
    }
  }
  return out;
}

class NativeInteropGpu : public ::testing::Test {
protected:
  using SharedTexture = native::d3d12::SharedTexture;

  struct Commands {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
  };

  struct GateRelease {
    ID3D12Fence* gate = nullptr;
    ~GateRelease() {
      if (gate != nullptr) {
        gate->Signal(1);
      }
    }
  };

  void SetUp() override {
    m_options = {
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
        .efbSize = {SharedWidth, SharedHeight, 1},
        .createEfb = true,
        .backendValidation = true,
        .nativeInterop = true,
    };
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType != wgpu::BackendType::D3D12) {
      m_device.reset();
      GTEST_SKIP() << "needs the D3D12 backend";
    }
    if (!m_device->device.HasFeature(wgpu::FeatureName::SharedTextureMemoryD3D12Resource) ||
        !m_device->device.HasFeature(wgpu::FeatureName::SharedFenceDXGISharedHandle)) {
      m_device.reset();
      GTEST_SKIP() << "needs D3D12 shared textures and fences";
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    ASSERT_TRUE(native::d3d12::initialize(m_device->device, m_device->adapter));
    Microsoft::WRL::ComPtr<ID3D12DebugDevice> debugDevice;
    if (FAILED(native_device()->QueryInterface(IID_PPV_ARGS(&debugDevice)))) {
      std::fprintf(stderr, "note: the D3D12 debug layer is not available\n");
    }
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      drain(native::d3d12::queue());
      for (const auto& queue : m_queues) {
        drain(queue.Get());
      }
      m_commands.clear();
      m_queues.clear();
      native::d3d12::run_teardowns();
      EXPECT_EQ(native::d3d12::shared_bytes(), 0u);
      EXPECT_EQ(debug_layer_errors(), std::vector<std::string>{});
      native::d3d12::shutdown();
    }
    if (m_device) {
      EXPECT_EQ(test::take_uncaptured_errors(*m_device), std::vector<std::string>{});
    }
    m_globals.reset();
    m_device.reset();
  }

  static ID3D12Device* native_device() { return native::d3d12::device(); }

  static std::vector<std::string> debug_layer_errors() {
    std::vector<std::string> out;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> info;
    if (native_device() == nullptr || FAILED(native_device()->QueryInterface(IID_PPV_ARGS(&info)))) {
      return out;
    }
    const UINT64 count = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
      SIZE_T length = 0;
      info->GetMessage(i, nullptr, &length);
      std::vector<std::byte> storage(length);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(info->GetMessage(i, message, &length)) && message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
        out.emplace_back(message->pDescription);
      }
    }
    return out;
  }

  static void drain(ID3D12CommandQueue* queue) {
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    ASSERT_TRUE(SUCCEEDED(native_device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ASSERT_TRUE(SUCCEEDED(queue->Signal(fence.Get(), 1)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(event, nullptr);
    ASSERT_TRUE(SUCCEEDED(fence->SetEventOnCompletion(1, event)));
    EXPECT_EQ(WaitForSingleObject(event, 10000), WAIT_OBJECT_0);
    CloseHandle(event);
  }

  Commands open_commands() {
    Commands out;
    EXPECT_TRUE(SUCCEEDED(
        native_device()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&out.allocator))));
    EXPECT_TRUE(SUCCEEDED(native_device()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, out.allocator.Get(),
                                                             nullptr, IID_PPV_ARGS(&out.list))));
    return out;
  }

  void execute(ID3D12CommandQueue* queue, const Commands& commands) {
    ASSERT_TRUE(SUCCEEDED(commands.list->Close()));
    ID3D12CommandList* lists[] = {commands.list.Get()};
    queue->ExecuteCommandLists(1, lists);
    m_commands.push_back(commands);
  }

  ID3D12CommandQueue* side_queue() {
    const D3D12_COMMAND_QUEUE_DESC desc{.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    EXPECT_TRUE(SUCCEEDED(native_device()->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))));
    m_queues.push_back(queue);
    return queue.Get();
  }

  static Microsoft::WRL::ComPtr<ID3D12Fence> gate_fence() {
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    EXPECT_TRUE(SUCCEEDED(native_device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    return fence;
  }

  static Microsoft::WRL::ComPtr<ID3D12Resource> native_buffer(D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    const D3D12_HEAP_PROPERTIES heap{.Type = type};
    const D3D12_RESOURCE_DESC desc{
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Width = uint64_t{SharedRowBytes} * SharedHeight,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1},
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    };
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    EXPECT_TRUE(SUCCEEDED(native_device()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                                   IID_PPV_ARGS(&buffer))));
    return buffer;
  }

  static D3D12_TEXTURE_COPY_LOCATION buffer_location(ID3D12Resource* buffer, DXGI_FORMAT format) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint.Footprint = {
        .Format = format, .Width = SharedWidth, .Height = SharedHeight, .Depth = 1, .RowPitch = SharedRowBytes};
    return location;
  }

  static D3D12_TEXTURE_COPY_LOCATION texture_location(ID3D12Resource* texture) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = texture;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = 0;
    return location;
  }

  static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
  }

  static std::vector<uint32_t> map_words(ID3D12Resource* buffer) {
    std::vector<uint32_t> out(size_t{SharedWidth} * SharedHeight);
    void* data = nullptr;
    const D3D12_RANGE range{0, out.size() * sizeof(uint32_t)};
    if (FAILED(buffer->Map(0, &range, &data))) {
      return {};
    }
    std::memcpy(out.data(), data, out.size() * sizeof(uint32_t));
    const D3D12_RANGE written{0, 0};
    buffer->Unmap(0, &written);
    return out;
  }

  [[nodiscard]] static std::unique_ptr<SharedTexture> shared(TextureFormat format, std::string_view label,
                                                             bool renderTarget = false) {
    return SharedTexture::create(
        {.format = format, .width = SharedWidth, .height = SharedHeight, .renderTarget = renderTarget, .label = label});
  }

  void write_dawn(const wgpu::Texture& texture, const std::vector<uint32_t>& words) const {
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = SharedRowBytes, .rowsPerImage = SharedHeight};
    const wgpu::Extent3D extent{SharedWidth, SharedHeight, 1};
    m_device->queue.WriteTexture(&destination, words.data(), words.size() * sizeof(uint32_t), &layout, &extent);
  }

  [[nodiscard]] wgpu::Buffer copy_dawn(const wgpu::Texture& texture) const {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                                            .size = uint64_t{SharedRowBytes} * SharedHeight};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = SharedRowBytes, .rowsPerImage = SharedHeight},
                                                .buffer = buffer};
    const wgpu::Extent3D extent{SharedWidth, SharedHeight, 1};
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&source, &destination, &extent);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    return buffer;
  }

  [[nodiscard]] std::vector<uint32_t> map_dawn(const wgpu::Buffer& buffer) const {
    bool mapped = false;
    const wgpu::Future future = buffer.MapAsync(
        wgpu::MapMode::Read, 0, buffer.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) { mapped = status == wgpu::MapAsyncStatus::Success; });
    if (m_device->instance.WaitAny(future, 5'000'000'000) != wgpu::WaitStatus::Success || !mapped) {
      return {};
    }
    return mapped_words(buffer);
  }

  static std::vector<uint32_t> mapped_words(const wgpu::Buffer& buffer) {
    std::vector<uint32_t> words(buffer.GetSize() / sizeof(uint32_t));
    std::memcpy(words.data(), buffer.GetConstMappedRange(0, buffer.GetSize()), words.size() * sizeof(uint32_t));
    buffer.Unmap();
    return words;
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  std::vector<Commands> m_commands;
  std::vector<Microsoft::WRL::ComPtr<ID3D12CommandQueue>> m_queues;
};

class D3D12SharedTextureGpu : public NativeInteropGpu {};
class NativeFence : public NativeInteropGpu {};

TEST_F(D3D12SharedTextureGpu, DawnWriteNativeCopyDawnRead) {
  const auto fence = native::d3d12::NativeFence::create("test.fence");
  const auto source = shared(TextureFormat::RGBA8Unorm, "test.source");
  const auto target = shared(TextureFormat::RGBA8Unorm, "test.target");
  ASSERT_TRUE(fence && source && target);
  EXPECT_GT(source->bytes(), 0u);
  EXPECT_EQ(native::d3d12::shared_bytes(), source->bytes() + target->bytes());
  const auto pattern = texel_pattern(1);

  ASSERT_TRUE(source->begin(false));
  write_dawn(source->texture(), pattern);
  const auto written = source->end();
  ASSERT_TRUE(written.has_value());
  ASSERT_FALSE(written->fences.empty());

  ID3D12CommandQueue* queue = native::d3d12::queue();
  ASSERT_TRUE(native::d3d12::queue_wait(queue, *written));
  const Commands commands = open_commands();
  commands.list->CopyResource(target->resource(), source->resource());
  execute(queue, commands);
  const uint64_t copied = fence->signal(queue);
  source->mark_used(*fence, copied);
  target->mark_used(*fence, copied);

  const std::array fences{fence->dawn()};
  const std::array values{copied};
  ASSERT_TRUE(target->begin(true, fences, values));
  const wgpu::Buffer buffer = copy_dawn(target->texture());
  EXPECT_TRUE(target->end().has_value());
  EXPECT_EQ(map_dawn(buffer), pattern);
}

TEST_F(D3D12SharedTextureGpu, Bgra8CopyFromEfb) {
  const auto fence = native::d3d12::NativeFence::create("test.fence");
  const auto copy = shared(TextureFormat::BGRA8Unorm, "test.efb_copy");
  ASSERT_TRUE(fence && copy);
  const wgpu::Texture& efb = webgpu::g_frameBuffer.texture;
  ASSERT_EQ(efb.GetFormat(), TextureFormat::BGRA8Unorm);
  const auto pattern = texel_pattern(2);
  write_dawn(efb, pattern);

  ASSERT_TRUE(copy->begin(false));
  const wgpu::TexelCopyTextureInfo from{.texture = efb};
  const wgpu::TexelCopyTextureInfo to{.texture = copy->texture()};
  const wgpu::Extent3D extent{SharedWidth, SharedHeight, 1};
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  encoder.CopyTextureToTexture(&from, &to, &extent);
  const wgpu::CommandBuffer dawnCommands = encoder.Finish();
  m_device->queue.Submit(1, &dawnCommands);
  const auto copied = copy->end();
  ASSERT_TRUE(copied.has_value());

  ID3D12CommandQueue* queue = native::d3d12::queue();
  const auto readback = native_buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  ASSERT_TRUE(readback);
  ASSERT_TRUE(native::d3d12::queue_wait(queue, *copied));
  const Commands commands = open_commands();
  const auto destination = buffer_location(readback.Get(), DXGI_FORMAT_B8G8R8A8_UNORM);
  const auto sourceLocation = texture_location(copy->resource());
  commands.list->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
  execute(queue, commands);
  const uint64_t read = fence->signal(queue);
  copy->mark_used(*fence, read);
  ASSERT_TRUE(fence->wait_cpu(read));
  EXPECT_EQ(map_words(readback.Get()), pattern);
}

TEST_F(D3D12SharedTextureGpu, BeginAccessResetsToCommon) {
  const auto fence = native::d3d12::NativeFence::create("test.fence");
  const auto target = shared(TextureFormat::RGBA8Unorm, "test.target", true);
  ASSERT_TRUE(fence && target);
  ID3D12CommandQueue* queue = native::d3d12::queue();

  ASSERT_TRUE(target->begin(false));
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = target->view();
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0.2, 0.4, 0.6, 1.0};
  const wgpu::RenderPassDescriptor pass{.colorAttachmentCount = 1, .colorAttachments = &attachment};
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  encoder.BeginRenderPass(&pass).End();
  const wgpu::CommandBuffer dawnCommands = encoder.Finish();
  m_device->queue.Submit(1, &dawnCommands);
  const auto cleared = target->end();
  ASSERT_TRUE(cleared.has_value());

  const auto readback = native_buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  ASSERT_TRUE(readback);
  ASSERT_TRUE(native::d3d12::queue_wait(queue, *cleared));
  const Commands copyCommands = open_commands();
  transition(copyCommands.list.Get(), target->resource(), D3D12_RESOURCE_STATE_COMMON,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  const auto destination = buffer_location(readback.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
  const auto sourceLocation = texture_location(target->resource());
  copyCommands.list->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
  transition(copyCommands.list.Get(), target->resource(), D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_COMMON);
  execute(queue, copyCommands);
  const uint64_t read = fence->signal(queue);
  target->mark_used(*fence, read);
  ASSERT_TRUE(fence->wait_cpu(read));
  EXPECT_EQ(map_words(readback.Get()), std::vector<uint32_t>(size_t{SharedWidth} * SharedHeight, 0xFF996633u));

  const D3D12_DESCRIPTOR_HEAP_DESC heapDesc{.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
  ASSERT_TRUE(SUCCEEDED(native_device()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap))));
  const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap->GetCPUDescriptorHandleForHeapStart();
  native_device()->CreateRenderTargetView(target->resource(), nullptr, rtv);
  const Commands clearCommands = open_commands();
  transition(clearCommands.list.Get(), target->resource(), D3D12_RESOURCE_STATE_COMMON,
             D3D12_RESOURCE_STATE_RENDER_TARGET);
  constexpr std::array<float, 4> NativeClear{0.8f, 0.6f, 0.4f, 0.2f};
  clearCommands.list->ClearRenderTargetView(rtv, NativeClear.data(), 0, nullptr);
  transition(clearCommands.list.Get(), target->resource(), D3D12_RESOURCE_STATE_RENDER_TARGET,
             D3D12_RESOURCE_STATE_COMMON);
  execute(queue, clearCommands);
  const uint64_t nativeCleared = fence->signal(queue);
  target->mark_used(*fence, nativeCleared);

  const std::array fences{fence->dawn()};
  const std::array values{nativeCleared};
  ASSERT_TRUE(target->begin(true, fences, values));
  const wgpu::Buffer buffer = copy_dawn(target->texture());
  EXPECT_TRUE(target->end().has_value());
  EXPECT_EQ(map_dawn(buffer), std::vector<uint32_t>(size_t{SharedWidth} * SharedHeight, 0x336699CCu));
}

TEST_F(D3D12SharedTextureGpu, FenceOrdersNativeBeforeDawnRead) {
  const auto fence = native::d3d12::NativeFence::create("test.fence");
  const auto target = shared(TextureFormat::RGBA8Unorm, "test.target");
  ASSERT_TRUE(fence && target);
  const auto stale = texel_pattern(3);
  const auto fresh = texel_pattern(4);
  ASSERT_TRUE(target->begin(false));
  write_dawn(target->texture(), stale);
  const auto written = target->end();
  ASSERT_TRUE(written.has_value());

  const auto upload = native_buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  ASSERT_TRUE(upload);
  void* mapped = nullptr;
  ASSERT_TRUE(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
  std::memcpy(mapped, fresh.data(), fresh.size() * sizeof(uint32_t));
  upload->Unmap(0, nullptr);

  ID3D12CommandQueue* side = side_queue();
  const auto gate = gate_fence();
  ASSERT_TRUE(side && gate);
  const GateRelease release{gate.Get()};
  ASSERT_TRUE(SUCCEEDED(side->Wait(gate.Get(), 1)));
  ASSERT_TRUE(native::d3d12::queue_wait(side, *written));
  const Commands commands = open_commands();
  const auto destination = texture_location(target->resource());
  const auto sourceLocation = buffer_location(upload.Get(), DXGI_FORMAT_R8G8B8A8_UNORM);
  commands.list->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
  execute(side, commands);
  const uint64_t uploaded = fence->signal(side);
  target->mark_used(*fence, uploaded);

  const std::array fences{fence->dawn()};
  const std::array values{uploaded};
  ASSERT_TRUE(target->begin(true, fences, values));
  const wgpu::Buffer buffer = copy_dawn(target->texture());
  EXPECT_TRUE(target->end().has_value());
  bool done = false;
  const wgpu::Future future = buffer.MapAsync(
      wgpu::MapMode::Read, 0, buffer.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
      [&](wgpu::MapAsyncStatus status, wgpu::StringView) { done = status == wgpu::MapAsyncStatus::Success; });
  EXPECT_EQ(m_device->instance.WaitAny(future, 200'000'000), wgpu::WaitStatus::TimedOut);
  EXPECT_LT(fence->completed(), uploaded);
  ASSERT_TRUE(SUCCEEDED(gate->Signal(1)));
  ASSERT_EQ(m_device->instance.WaitAny(future, 5'000'000'000), wgpu::WaitStatus::Success);
  ASSERT_TRUE(done);
  EXPECT_EQ(mapped_words(buffer), fresh);
  EXPECT_TRUE(fence->wait_cpu(uploaded));
}

TEST_F(NativeFence, WaitCpuBlocksUntilSignal) {
  const auto fence = native::d3d12::NativeFence::create("test.fence");
  ASSERT_TRUE(fence);
  ASSERT_TRUE(fence->dawn());
  EXPECT_EQ(fence->last_signalled(), 0u);
  EXPECT_TRUE(fence->wait_cpu(0));
  ID3D12CommandQueue* side = side_queue();
  const auto gate = gate_fence();
  ASSERT_TRUE(side && gate);
  const GateRelease release{gate.Get()};
  ASSERT_TRUE(SUCCEEDED(side->Wait(gate.Get(), 1)));
  const uint64_t value = fence->signal(side);
  EXPECT_EQ(value, 1u);
  EXPECT_EQ(fence->last_signalled(), value);
  std::atomic<bool> done{false};
  std::atomic<bool> ok{false};
  std::thread waiter{[&] {
    ok = fence->wait_cpu(value);
    done = true;
  }};
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  EXPECT_FALSE(done);
  EXPECT_LT(fence->completed(), value);
  EXPECT_TRUE(SUCCEEDED(gate->Signal(1)));
  waiter.join();
  EXPECT_TRUE(done);
  EXPECT_TRUE(ok);
  EXPECT_GE(fence->completed(), value);
}

class NativeStepGpu : public NativeInteropGpu {
protected:
  static constexpr uint32_t SceneWidth = SharedWidth / 2;
  static constexpr uint32_t SceneHeight = SharedHeight / 2;
#ifdef TRACY_ENABLE
  static constexpr bool TracyCollects = true;
#else
  static constexpr bool TracyCollects = false;
#endif

  struct Evaluator {
    native::d3d12::SharedTexture* in = nullptr;
    native::d3d12::SharedTexture* out = nullptr;
    bool fails = false;
    uint32_t failures = 0;
  };

  static bool record_copy(void* user, ID3D12GraphicsCommandList* list) noexcept {
    const auto& evaluator = *static_cast<const Evaluator*>(user);
    if (evaluator.fails) {
      return false;
    }
    const auto destination = texture_location(evaluator.out->resource());
    const auto source = texture_location(evaluator.in->resource());
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    return true;
  }

  static void record_failure(void* user) noexcept { ++static_cast<Evaluator*>(user)->failures; }

  void SetUp() override {
    NativeInteropGpu::SetUp();
    if (!m_started) {
      return;
    }
    jobs::initialize(1);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    compose::testing::reset();
    m_composeStarted = true;
    const wgpu::TextureDescriptor depth{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {SharedWidth, SharedHeight, 1},
        .format = DepthFormat,
    };
    webgpu::g_depthBuffer.texture = m_device->device.CreateTexture(&depth);
    webgpu::g_depthBuffer.view = webgpu::g_depthBuffer.texture.CreateView();
    webgpu::g_depthBuffer.size = depth.size;
    webgpu::g_depthBuffer.format = DepthFormat;
    webgpu::set_scene_render_size(SceneWidth, SceneHeight);
    ASSERT_EQ(webgpu::scene_color().size.width, SceneWidth);
    ASSERT_TRUE(compose::prepare_pipelines(ComposeMode::Split, ColorFormat, DepthFormat));
    const compose::Params params = compose::make_params(m_data, {SceneWidth, SceneHeight}, {SharedWidth, SharedHeight});
    m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, m_data.uniformOffset, &params, sizeof(params));
    m_in = native::d3d12::SharedTexture::create(
        {.format = ColorFormat, .width = SceneWidth, .height = SceneHeight, .label = "test.step_in"});
    m_out = native::d3d12::SharedTexture::create(
        {.format = ColorFormat, .width = SceneWidth, .height = SceneHeight, .label = "test.step_out"});
    ASSERT_TRUE(m_in && m_out);
    m_evaluator = {.in = m_in.get(), .out = m_out.get()};
    m_runner = native::d3d12::StepRunner::create({
        .name = "test",
        .inputState = D3D12_RESOURCE_STATE_COPY_SOURCE,
        .outputState = D3D12_RESOURCE_STATE_COPY_DEST,
        .record = &record_copy,
        .failed = &record_failure,
        .user = &m_evaluator,
    });
    ASSERT_TRUE(m_runner);
  }

  void TearDown() override {
    m_runner.reset();
    m_in.reset();
    m_out.reset();
    if (m_timed) {
      EXPECT_TRUE(webgpu::gpu_prof::wait_idle(5'000'000'000));
      webgpu::gpu_prof::shutdown();
      webgpu::gpu_prof::set_always(TracyCollects);
    }
    if (m_composeStarted) {
      compose::shutdown();
      compose::testing::reset();
      webgpu::set_scene_render_size(0, 0);
      shaders::shutdown();
      jobs::shutdown();
      gfx::set_blocking_pipelines(false);
    }
    NativeInteropGpu::TearDown();
  }

  void write_scene(uint32_t seed) const {
    std::vector<uint32_t> words(size_t{SceneWidth} * SceneHeight);
    for (uint32_t y = 0; y < SceneHeight; ++y) {
      for (uint32_t x = 0; x < SceneWidth; ++x) {
        words[size_t{y} * SceneWidth + x] = mix_bits(x + seed * 977u, y + seed * 131u) | 0xFF000000u;
      }
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = webgpu::scene_color().texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = SceneWidth * 4, .rowsPerImage = SceneHeight};
    const wgpu::Extent3D extent{SceneWidth, SceneHeight, 1};
    m_device->queue.WriteTexture(&destination, words.data(), words.size() * sizeof(uint32_t), &layout, &extent);
  }

  // One frame as the worker and the submit callback run it, where the native node's input copy ends segment A
  void run_frame(bool native) {
    wgpu::CommandEncoder cmd = m_device->device.CreateCommandEncoder();
    if (m_timed) {
      webgpu::gpu_prof::frame_begin(cmd);
    }
    if (native) {
      const std::array inputs{m_in.get()};
      compose::request_native_step(m_runner->step(inputs, m_out.get()),
                                   {.view = m_out->view(), .format = m_out->format()});
      const wgpu::TexelCopyTextureInfo source{.texture = webgpu::scene_color().texture};
      const wgpu::TexelCopyTextureInfo destination{.texture = m_in->texture()};
      const wgpu::Extent3D extent{SceneWidth, SceneHeight, 1};
      cmd.CopyTextureToTexture(&source, &destination, &extent);
    }
    gfx::detail::FramePacket frame;
    const gfx::detail::FrameOp op{.type = gfx::detail::FrameOpType::SceneCompose, .sceneCompose = &m_data};
    gfx::detail::encode_op(cmd, frame, op);
    EXPECT_EQ(frame.segments.size(), native ? 1u : 0u);
    if (m_timed) {
      webgpu::gpu_prof::frame_end(cmd);
    }
    const wgpu::CommandBuffer last = cmd.Finish();
    submit_segments(frame.segments);
    m_device->queue.Submit(1, &last);
    if (m_timed) {
      webgpu::gpu_prof::after_submit();
    }
    end_segments(frame.segments);
  }

  [[nodiscard]] std::vector<uint32_t> read_output() const { return map_dawn(copy_dawn(webgpu::g_frameBuffer.texture)); }

  SceneComposeData m_data{
      .mode = ComposeMode::Split,
      .ratio = {static_cast<float>(SharedWidth) / SceneWidth, static_cast<float>(SharedHeight) / SceneHeight},
  };
  std::unique_ptr<native::d3d12::SharedTexture> m_in;
  std::unique_ptr<native::d3d12::SharedTexture> m_out;
  std::unique_ptr<native::d3d12::StepRunner> m_runner;
  Evaluator m_evaluator;
  bool m_composeStarted = false;
  bool m_timed = false;
};

TEST_F(NativeStepGpu, CopyStepOrdersSegments) {
  for (uint32_t frame = 0; frame < 3; ++frame) {
    SCOPED_TRACE(frame);
    write_scene(frame + 5);
    run_frame(false);
    const std::vector<uint32_t> bilinear = read_output();
    ASSERT_EQ(bilinear.size(), size_t{SharedWidth} * SharedHeight);
    run_frame(true);
    EXPECT_EQ(read_output(), bilinear);
  }
  EXPECT_FALSE(m_runner->failed());
  EXPECT_EQ(m_evaluator.failures, 0u);
  const auto counters = compose::testing::counters();
  EXPECT_EQ(counters.native, 3u);
  EXPECT_EQ(counters.statusWrites, 3u);
  EXPECT_EQ(counters.lastStatus, 1u);
}

TEST_F(NativeStepGpu, FailingStepComposesBilinear) {
  m_evaluator.fails = true;
  for (uint32_t frame = 0; frame < 2; ++frame) {
    SCOPED_TRACE(frame);
    write_scene(frame + 11);
    run_frame(false);
    const std::vector<uint32_t> bilinear = read_output();
    run_frame(true);
    EXPECT_EQ(read_output(), bilinear);
    EXPECT_EQ(compose::testing::counters().lastStatus, 0u);
  }
  EXPECT_TRUE(m_runner->failed());
  EXPECT_EQ(m_evaluator.failures, 1u);
}

TEST_F(NativeStepGpu, TimestampsReported) {
  webgpu::gpu_prof::shutdown();
  webgpu::gpu_prof::set_always(true);
  webgpu::gpu_prof::initialize();
  m_timed = true;
  if (!webgpu::gpu_prof::supported()) {
    GTEST_SKIP() << "the device has no timestamp queries";
  }
  for (uint32_t frame = 0; frame < 3; ++frame) {
    write_scene(frame);
    run_frame(true);
    ASSERT_TRUE(webgpu::gpu_prof::wait_idle(5'000'000'000));
  }
  webgpu::gpu_prof::FrameTimes times;
  ASSERT_TRUE(webgpu::gpu_prof::latest(times));
  const auto native = std::ranges::find_if(times.passes, [](const webgpu::gpu_prof::PassTime& pass) {
    return pass.name != nullptr && std::string_view{pass.name} == "native.test";
  });
  ASSERT_NE(native, times.passes.end());
  EXPECT_GT(native->ms, 0.f);
  EXPECT_LT(native->ms, 100.f);
}

namespace native_upscale = upscale::native_upscale;

class NativeUpscaleGpu : public NativeInteropGpu {
protected:
  using Passthrough = native::d3d12::upscale_bridge::PassthroughUpscaler;
  static constexpr float DepthValue = 0.375f;

  void SetUp() override {
    NativeInteropGpu::SetUp();
    if (!m_started) {
      return;
    }
    jobs::initialize(1);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    gfx::tex_copy_conv::initialize();
    compose::testing::reset();
    upscale::native_upscale::testing::reset();
    m_ready = true;
    m_rig = std::make_unique<fsr_rig::Rig>(*m_device, m_options);
    if (!m_rig->ok()) {
      GTEST_SKIP() << "the base G-buffer tier needs a core device";
    }
  }

  void TearDown() override {
    if (m_ready) {
      m_rig.reset();
      native::d3d12::upscale_bridge::testing::reset();
      gfx::tex_copy_conv::shutdown();
      shaders::shutdown();
      jobs::shutdown();
      gfx::set_blocking_pipelines(false);
    }
    NativeInteropGpu::TearDown();
  }

  static Settings settings(UpscalerQuality quality = UpscalerQuality::Quality, uint32_t debugView = 0) {
    Settings out;
    out.upscalerQuality = quality;
    out.taaSharpness = 0.f;
    out.debugView = debugView;
    return out;
  }

  void install(const Settings& wanted) {
    native_upscale::testing::set_stand_in(&m_copy);
    for (int i = 0; i < 3; ++i) {
      upscale::on_latch(wanted);
      native_worker_frame_begin();
    }
    ASSERT_TRUE(native_upscale::is_native(temporal_resolve::installed()));
    m_rig->use(m_rig->compile(wanted));
  }

  [[nodiscard]] const CompiledNode& node() const {
    const NodeId id = m_rig->node(native_upscale::Pass);
    EXPECT_NE(id, InvalidNode);
    return m_rig->m_graph->nodes[id];
  }

  void clear_targets() const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const auto clear = [&](const wgpu::TextureView& view, const wgpu::Color& value) {
      const wgpu::RenderPassColorAttachment attachment{
          .view = view,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = value,
      };
      const wgpu::RenderPassDescriptor descriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
      encoder.BeginRenderPass(&descriptor).End();
    };
    clear(webgpu::g_motionBuffer.view, {0.25, -0.5, 0.0, 0.0});
    clear(webgpu::g_reactiveBuffer.view, {0.6, 0.0, 0.0, 0.0});
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = webgpu::scene_depth().view,
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = DepthValue,
    };
    const wgpu::RenderPassDescriptor depthPass{.depthStencilAttachment = &depth};
    encoder.BeginRenderPass(&depthPass).End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  }

  void write_scene(uint32_t seed) {
    const wgpu::Texture target = webgpu::scene_color().texture;
    const uint32_t width = target.GetWidth();
    const uint32_t height = target.GetHeight();
    m_scene.assign(size_t{width} * height, Rgba{});
    std::vector<uint8_t> bgra(m_scene.size() * 4);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const size_t i = size_t{y} * width + x;
        m_scene[i] = noise(x, y, seed);
        bgra[i * 4 + 0] = m_scene[i].b;
        bgra[i * 4 + 1] = m_scene[i].g;
        bgra[i * 4 + 2] = m_scene[i].r;
        bgra[i * 4 + 3] = m_scene[i].a;
      }
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = target};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device->queue.WriteTexture(&destination, bgra.data(), bgra.size(), &layout, &size);
  }

  std::vector<Rgba> run_frame(uint32_t seed, fsr_rig::FrameOptions options = {}) {
    const webgpu::TextureWithSampler& depth = webgpu::scene_depth();
    options.sceneDepth = std::make_shared<const SceneDepth>(SceneDepth{
        .source = DepthSource::Direct,
        .sampledTexture = depth.texture,
        .sampled = depth.view,
        .sampledFormat = DepthFormat,
        .attachmentTexture = depth.texture,
        .attachment = depth.view,
        .attachmentFormat = DepthFormat,
        .size = depth.size,
    });
    const StageData post = m_rig->build(++m_frame, options);
    write_scene(seed);
    clear_targets();
    SceneComposeData data{
        .mode = upscale::split_live() ? ComposeMode::Split : ComposeMode::InPlace,
        .ratio = {upscale::ratio(), upscale::ratio()},
        .rcasStrength = upscale::rcas_strength(),
        .frameId = post.frameId,
    };
    if (post.sceneHints != nullptr && data.mode == ComposeMode::Split) {
      data.jitterPx[0] = post.sceneHints->jitterPx[0];
      data.jitterPx[1] = post.sceneHints->jitterPx[1];
    }
    const compose::Params params = compose::make_params(data, upscale::scene_size(), upscale::output_size());
    m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, &params, sizeof(params));
    wgpu::CommandEncoder cmd = m_device->device.CreateCommandEncoder();
    if (post.any_enabled()) {
      run_stage(cmd, post);
    }
    gfx::detail::FramePacket packet;
    const gfx::detail::FrameOp op{.type = gfx::detail::FrameOpType::SceneCompose, .sceneCompose = &data};
    gfx::detail::encode_op(cmd, packet, op);
    m_segments = packet.segments.size();
    const wgpu::CommandBuffer last = cmd.Finish();
    submit_segments(packet.segments);
    m_device->queue.Submit(1, &last);
    end_segments(packet.segments);
    const std::vector<uint8_t> bytes = read_texture(webgpu::g_frameBuffer.texture, 4);
    std::vector<Rgba> out(bytes.size() / 4);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = {.r = bytes[i * 4 + 2], .g = bytes[i * 4 + 1], .b = bytes[i * 4 + 0], .a = bytes[i * 4 + 3]};
    }
    return out;
  }

  [[nodiscard]] std::vector<uint8_t> read_texture(const wgpu::Texture& source, uint32_t bytesPerTexel) const {
    const uint32_t width = source.GetWidth();
    const uint32_t height = source.GetHeight();
    const uint32_t row = (width * bytesPerTexel + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo from{.texture = source};
    const wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = row, .rowsPerImage = height}, .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&from, &to, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<uint8_t> out(size_t{width} * height * bytesPerTexel);
    if (!words) {
      ADD_FAILURE() << "texture readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(out.data() + size_t{y} * width * bytesPerTexel, bytes + size_t{y} * row, width * bytesPerTexel);
    }
    return out;
  }

  [[nodiscard]] std::vector<uint8_t> read_shared(SharedTexture* texture, uint32_t bytesPerTexel) const {
    if (texture == nullptr) {
      ADD_FAILURE() << "shared texture missing";
      return {};
    }
    EXPECT_TRUE(texture->begin(true));
    std::vector<uint8_t> out = read_texture(texture->texture(), bytesPerTexel);
    EXPECT_TRUE(texture->end().has_value());
    return out;
  }

  static int max_rgb_error(const std::vector<Rgba>& got, const std::vector<Rgba>& want) {
    int out = got.size() == want.size() ? 0 : 255;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
      for (const int d :
           {std::abs(got[i].r - want[i].r), std::abs(got[i].g - want[i].g), std::abs(got[i].b - want[i].b)}) {
        out = std::max(out, d);
      }
    }
    return out;
  }

  bool m_ready = false;
  Passthrough m_copy;
  std::unique_ptr<fsr_rig::Rig> m_rig;
  std::vector<Rgba> m_scene;
  uint64_t m_frame = 0;
  size_t m_segments = 0;
};

TEST_F(NativeUpscaleGpu, InputsMatchSceneTargets) {
  m_copy = Passthrough{2.f, ColorEncoding::Stored, true};
  install(settings());
  ASSERT_TRUE(upscale::split_live());
  ASSERT_EQ(node().status, CullReason::None);
  run_frame(3);
  EXPECT_EQ(m_segments, 1u);
  EXPECT_EQ(m_copy.evaluations(), 1u);
  const auto shared = native::d3d12::upscale_bridge::testing::shared();
  EXPECT_EQ(read_shared(shared.color, 4), read_texture(webgpu::scene_color().texture, 4));
  EXPECT_EQ(read_shared(shared.motion, 4), read_texture(webgpu::g_motionBuffer.texture, 4));
  EXPECT_EQ(read_shared(shared.reactive, 1), read_texture(webgpu::g_reactiveBuffer.texture, 1));
  const std::vector<uint8_t> depth = read_shared(shared.depth, 4);
  ASSERT_EQ(depth.size(), size_t{SharedWidth / 2} * (SharedHeight / 2) * 4);
  for (size_t i = 0; i < depth.size(); i += 4) {
    float value = 0.f;
    std::memcpy(&value, depth.data() + i, sizeof(value));
    ASSERT_EQ(value, DepthValue) << i / 4;
  }
  EXPECT_EQ(compose::testing::counters().native, 1u);
}

TEST_F(NativeUpscaleGpu, LinearInputRoundTrips) {
  m_copy = Passthrough{1.f, ColorEncoding::Linear};
  install(settings(UpscalerQuality::NativeAa));
  ASSERT_FALSE(upscale::split_live());
  for (uint32_t seed = 0; seed < 3; ++seed) {
    const std::vector<Rgba> output = run_frame(seed + 20);
    EXPECT_EQ(m_segments, 1u);
    EXPECT_LE(max_rgb_error(output, m_scene), 1) << "seed " << seed;
  }
  const auto shared = native::d3d12::upscale_bridge::testing::shared();
  ASSERT_NE(shared.color, nullptr);
  EXPECT_EQ(shared.color->format(), TextureFormat::RGBA16Float);
  EXPECT_EQ(shared.out->format(), TextureFormat::RGBA16Float);
  EXPECT_EQ(compose::testing::counters().lastStatus, 1u);
}

TEST_F(NativeUpscaleGpu, ComposeIsRgbOnly) {
  m_copy = Passthrough{2.f};
  install(settings());
  ASSERT_TRUE(upscale::split_live());
  const std::vector<Rgba> bilinear = run_frame(41, {.camera = false});
  EXPECT_EQ(m_segments, 0u);
  const std::vector<Rgba> output = run_frame(41);
  EXPECT_EQ(m_segments, 1u);
  EXPECT_EQ(max_rgb_error(output, bilinear), 0);
  const uint32_t sceneWidth = webgpu::scene_color().size.width;
  const uint32_t sceneHeight = webgpu::scene_color().size.height;
  size_t changed = 0;
  for (uint32_t py = 0; py < SharedHeight; ++py) {
    for (uint32_t px = 0; px < SharedWidth; ++px) {
      const uint32_t sx = nearest(px, 0.5f, 0.f, sceneWidth);
      const uint32_t sy = nearest(py, 0.5f, 0.f, sceneHeight);
      changed += output[size_t{py} * SharedWidth + px].a != m_scene[size_t{sy} * sceneWidth + sx].a ? 1 : 0;
    }
  }
  EXPECT_EQ(changed, 0u);
  EXPECT_EQ(compose::testing::counters().native, 1u);
}

TEST_F(NativeUpscaleGpu, ResizeRecreatesShared) {
  m_copy = Passthrough{2.f};
  install(settings());
  run_frame(5);
  auto shared = native::d3d12::upscale_bridge::testing::shared();
  ASSERT_NE(shared.color, nullptr);
  EXPECT_EQ(shared.color->texture().GetWidth(), SharedWidth / 2);
  EXPECT_EQ(shared.out->texture().GetHeight(), SharedHeight / 2);
  m_copy.set_ratio(1.5f);
  install(settings(UpscalerQuality::Performance));
  EXPECT_EQ(upscale::scene_size(), (GraphExtent{42, 21}));
  const std::vector<Rgba> output = run_frame(6);
  EXPECT_EQ(m_segments, 1u);
  shared = native::d3d12::upscale_bridge::testing::shared();
  ASSERT_NE(shared.color, nullptr);
  EXPECT_EQ(shared.color->texture().GetWidth(), 42u);
  EXPECT_EQ(shared.color->texture().GetHeight(), 21u);
  EXPECT_EQ(shared.out->texture().GetWidth(), 42u);
  const uint64_t bytes = shared.color->bytes() + shared.depth->bytes() + shared.motion->bytes() + shared.out->bytes();
  EXPECT_EQ(native::d3d12::shared_bytes(), bytes);
  EXPECT_EQ(m_copy.prepares(), 2u);
  EXPECT_EQ(m_copy.evaluations(), 2u);
  EXPECT_EQ(compose::testing::counters().lastStatus, 1u);
  EXPECT_EQ(output.size(), size_t{SharedWidth} * SharedHeight);
}

TEST_F(NativeUpscaleGpu, NativeAaInplace) {
  m_copy = Passthrough{1.f};
  install(settings(UpscalerQuality::NativeAa));
  ASSERT_FALSE(upscale::split_live());
  ASSERT_TRUE(upscale::compose_wanted());
  for (uint32_t seed = 0; seed < 2; ++seed) {
    const std::vector<Rgba> output = run_frame(seed + 60);
    EXPECT_EQ(m_segments, 1u);
    ASSERT_EQ(output.size(), m_scene.size());
    EXPECT_EQ(max_rgb_error(output, m_scene), 0);
    size_t changed = 0;
    for (size_t i = 0; i < output.size(); ++i) {
      changed += output[i].a != m_scene[i].a ? 1 : 0;
    }
    EXPECT_EQ(changed, 0u);
  }
  EXPECT_EQ(compose::testing::counters().native, 2u);
  EXPECT_EQ(compose::testing::counters().skipped, 0u);
}

TEST_F(NativeUpscaleGpu, SkippedNodeFrameSubmitsClean) {
  m_copy = Passthrough{2.f};
  install(settings(UpscalerQuality::Quality, 16));
  EXPECT_EQ(node().status, CullReason::Skipped);
  run_frame(7);
  EXPECT_EQ(m_segments, 0u);

  install(settings());
  EXPECT_EQ(node().status, CullReason::None);
  run_frame(8, {.camera = false});
  EXPECT_EQ(m_segments, 0u);
  EXPECT_GE(native_upscale::testing::counters().skippedNoScene, 1u);
  const auto counters = compose::testing::counters();
  EXPECT_EQ(counters.encoded, 2u);
  EXPECT_EQ(counters.native, 0u);
  EXPECT_EQ(counters.skipped, 0u);
  EXPECT_EQ(m_copy.evaluations(), 0u);
}
#endif
} // namespace
} // namespace aurora::render::detail
