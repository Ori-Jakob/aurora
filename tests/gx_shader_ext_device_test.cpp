#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"
#include "render_test_device.hpp"

#include "gfx/frame_packet.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "gx/hex_wgsl.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "gx/shader_info.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aurora::gx {
namespace {
constexpr size_t MinHexConfigs = 20;
constexpr size_t MaxHexConfigs = 48;

struct HexCase {
  ShaderConfig config;
  corpus::Variant variant;
};

// Synthetic corpus configs with at least one eligible map, flagged on every eligible map
std::vector<HexCase> collect_hex_cases() {
  std::vector<HexCase> cases;
  for (const auto& entry : corpus::generate_synthetic()) {
    const corpus::Variant variant = corpus::decode_variant(entry.record);
    if (variant.dstAlphaMode == DstAlphaMode::DualSource) {
      continue;
    }
    ShaderConfig config = corpus::decode(entry.record);
    const ShaderInfo info = build_shader_info(config);
    if (info.stochasticEligible.none()) {
      continue;
    }
    config.ext.stochasticMask = static_cast<u8>(info.stochasticEligible.to_ulong());
    cases.push_back({config, variant});
    if (cases.size() == MaxHexConfigs) {
      break;
    }
  }
  return cases;
}

const std::vector<HexCase>& hex_cases() {
  static const std::vector<HexCase> cases = collect_hex_cases();
  return cases;
}

std::optional<std::string> module_error(test::TestDevice& device, const std::string& text) {
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = text.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
  device.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  const wgpu::ShaderModule module = device.device.CreateShaderModule(&descriptor);
  return test::pop_validation_error(device);
}

gfx::RenderTargetLayout scene_layout() {
  gfx::RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{gfx::ColorAttachmentSemantic::SceneColor, wgpu::TextureFormat::BGRA8Unorm}}},
      .depthStencilFormat = wgpu::TextureFormat::Depth32Float,
      .sampleCount = 1,
  };
  gfx::detail::finalize_render_target_layout(layout);
  return layout;
}

PipelineConfig pipeline_for(const ShaderConfig& shader) {
  PipelineConfig config{};
  config.shaderConfig = shader;
  config.depthFunc = GX_LEQUAL;
  config.cullMode = GX_CULL_NONE;
  config.blendMode = GX_BM_NONE;
  config.blendFacSrc = GX_BL_ONE;
  config.blendFacDst = GX_BL_ZERO;
  config.blendOp = GX_LO_CLEAR;
  config.dstAlpha = UINT32_MAX;
  config.depthCompare = true;
  config.depthUpdate = true;
  config.colorUpdate = true;
  config.alphaUpdate = true;
  return config;
}

std::optional<wgpu::BackendType> hardware_backend() {
  const auto backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND"));
  if (!backend || (*backend != wgpu::BackendType::D3D12 && *backend != wgpu::BackendType::D3D11 &&
                   *backend != wgpu::BackendType::Vulkan)) {
    return std::nullopt;
  }
  return backend;
}

class GxShaderExtDevice : public ::testing::Test {
protected:
  void SetUp() override {
    m_alignment = gfx::detail::resources().limits.minUniformBufferOffsetAlignment;
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
  }

  void TearDown() override {
    testing::set_hex_corrupt(false);
    gfx::set_blocking_pipelines(false);
    gfx::shutdown_pipeline_cache();
    m_globals.reset();
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_device.reset();
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = m_alignment;
  }

  bool start(const test::TestDeviceOptions& options, bool globals) {
    std::string error;
    m_device = test::create_test_device(options, error);
    if (!m_device) {
      return false;
    }
    if (globals) {
      m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, options);
      gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
    }
    return true;
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;

private:
  uint32_t m_alignment = 0;
};

TEST_F(GxShaderExtDevice, HexValidatesWhereLegacyValidates) {
  const std::vector<HexCase>& cases = hex_cases();
  ASSERT_GE(cases.size(), MinHexConfigs);
  for (const bool core : {true, false}) {
    ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = core}, false)) << "core " << core;
    size_t validated = 0;
    for (const HexCase& item : cases) {
      ShaderConfig legacy = item.config;
      legacy.ext = {};
      const auto legacyError = module_error(
          *m_device, build_shader_source(legacy, item.variant.dstAlphaMode, item.variant.normalAttachment));
      if (legacyError) {
        continue;
      }
      const std::string text =
          build_shader_source(item.config, item.variant.dstAlphaMode, item.variant.normalAttachment);
      ASSERT_NE(text.find("fn hex_tex"), std::string::npos);
      const auto error = module_error(*m_device, text);
      EXPECT_FALSE(error.has_value()) << "core " << core << ": " << error.value_or("") << "\n" << text;
      ++validated;
    }
    EXPECT_GE(validated, MinHexConfigs) << "core " << core;
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_device.reset();
  }
}

TEST_F(GxShaderExtDevice, InvalidExtPipelineIsCaptured) {
  const std::vector<HexCase>& cases = hex_cases();
  ASSERT_FALSE(cases.empty());
  for (const bool productionToggles : {false, true}) {
    const test::TestDeviceOptions options{
        .backend = wgpu::BackendType::Null, .core = true, .productionToggles = productionToggles};
    ASSERT_TRUE(start(options, true));
    gfx::set_blocking_pipelines(true);
    const gfx::RenderTargetLayout layout = scene_layout();
    const PipelineConfig config = pipeline_for(cases.front().config);

    // Dawn derives the auto layout only with validation on, so the compiled control runs without production toggles
    if (!productionToggles) {
      const gfx::PipelineRef good = gfx::find_debug_pipeline(config, layout);
      EXPECT_EQ(gfx::pipeline_state(good), gfx::PipelineState::Ready);
    }

    testing::set_hex_corrupt(true);
    PipelineConfig broken = config;
    broken.polygonOffsetBits = 1;
    const gfx::PipelineRef ref = gfx::find_debug_pipeline(broken, layout);
    testing::set_hex_corrupt(false);
    EXPECT_EQ(gfx::pipeline_state(ref), gfx::PipelineState::Failed) << "toggles " << productionToggles;
    gfx::CompiledPipeline compiled;
    EXPECT_FALSE(gfx::get_pipeline(ref, compiled));
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());

    gfx::set_blocking_pipelines(false);
    gfx::shutdown_pipeline_cache();
    m_globals.reset();
    m_device.reset();
  }
}

TEST_F(GxShaderExtDevice, HexPipelinesCompileOnBackend) {
  const auto backend = hardware_backend();
  if (!backend) {
    GTEST_SKIP() << "AURORA_TEST_BACKEND names no hardware backend";
  }
  if (!start({.backend = *backend, .core = true, .allowCpu = false}, true)) {
    GTEST_SKIP() << "no adapter";
  }
  const gfx::RenderTargetLayout layout = scene_layout();
  size_t compiled = 0;
  for (const HexCase& item : hex_cases()) {
    const gfx::CompiledPipeline pipeline = create_pipeline(pipeline_for(item.config), layout);
    EXPECT_TRUE(static_cast<bool>(pipeline.main));
    compiled += pipeline.main ? 1 : 0;
  }
  EXPECT_GE(compiled, MinHexConfigs);
  ::testing::Test::RecordProperty("compiled", static_cast<int>(compiled));
}

TEST_F(GxShaderExtDevice, HexTapsPreserveTexelLength) {
  const auto backend = hardware_backend();
  if (!backend) {
    GTEST_SKIP() << "AURORA_TEST_BACKEND names no hardware backend";
  }
  if (!start({.backend = *backend, .core = true, .allowCpu = false}, false)) {
    GTEST_SKIP() << "no adapter";
  }
  constexpr uint32_t Points = 256;
  constexpr uint32_t Fields = 6;
  std::string text;
  append_hex_common(text);
  text += R"(
@group(0) @binding(0) var<storage, read_write> out_words: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let uv = vec2f(f32(i % 16u) * 0.731 + 0.113, f32(i / 16u) * 0.587 - 3.29);
    let sz = select(vec2f(256.0, 64.0), vec2f(32.0, 512.0), (i & 1u) == 1u);
    let e = vec2f(0.37, -0.21) * 0.002;
    let a = hex_taps(uv, 0.8, 1.0, sz);
    let b = hex_taps(uv + e, 0.8, 1.0, sz);
    out_words[i * 6u + 0u] = length((b.uv0 - a.uv0) * sz);
    out_words[i * 6u + 1u] = length((b.uv1 - a.uv1) * sz);
    out_words[i * 6u + 2u] = length((b.uv2 - a.uv2) * sz);
    out_words[i * 6u + 3u] = length(e * sz);
    out_words[i * 6u + 4u] = length(a.cs0) + length(a.cs1) + length(a.cs2);
    out_words[i * 6u + 5u] = min(min(a.bary.x, a.bary.y), min(a.bary.z, min(min(b.bary.x, b.bary.y), b.bary.z)));
}
)";
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = text.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
  const wgpu::ShaderModule module = m_device->device.CreateShaderModule(&moduleDescriptor);
  const wgpu::ComputePipelineDescriptor pipelineDescriptor{.compute = {.module = module, .entryPoint = "main"}};
  const wgpu::ComputePipeline pipeline = m_device->device.CreateComputePipeline(&pipelineDescriptor);
  constexpr uint64_t Size = Points * Fields * sizeof(float);
  const wgpu::BufferDescriptor bufferDescriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                                .size = Size};
  const wgpu::Buffer output = m_device->device.CreateBuffer(&bufferDescriptor);
  const wgpu::BindGroupEntry entry{.binding = 0, .buffer = output, .size = Size};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .layout = pipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &entry};
  const wgpu::BindGroup group = m_device->device.CreateBindGroup(&groupDescriptor);
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group);
  pass.DispatchWorkgroups(Points / 64);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);
  const auto words = test::read_buffer(*m_device, output, 0, Size);
  ASSERT_TRUE(words.has_value());
  ASSERT_EQ(words->size(), Points * Fields);
  std::vector<float> values(words->size());
  std::memcpy(values.data(), words->data(), Size);
  size_t checked = 0;
  for (uint32_t i = 0; i < Points; ++i) {
    const float* v = &values[i * Fields];
    EXPECT_NEAR(v[4], 3.f, 1e-4f) << i;
    // A step that crosses a lattice edge changes the tap set, so only steps inside one triangle compare
    if (v[5] < 0.01f) {
      continue;
    }
    for (uint32_t tap = 0; tap < 3; ++tap) {
      EXPECT_NEAR(v[tap], v[3], v[3] * 0.02f) << "point " << i << " tap " << tap;
    }
    ++checked;
  }
  EXPECT_GE(checked, Points / 2);
}
} // namespace
} // namespace aurora::gx
