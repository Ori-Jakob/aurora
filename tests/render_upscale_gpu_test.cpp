#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "render/jobs.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/shaders.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
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
} // namespace
} // namespace aurora::render::detail
