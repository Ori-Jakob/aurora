#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "render/jobs.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/post/upscale/fsr/fsr.hpp"
#include "render/shaders.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::detail {
namespace {
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
} // namespace
} // namespace aurora::render::detail
