#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"
#include "render_test_device.hpp"

#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/hash.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "gfx/scene_targets.hpp"
#include "gx/hex_wgsl.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "gx/shader_info.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
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

bool in_allow_class(const corpus::Record& record) {
  const corpus::Variant variant = corpus::decode_variant(record);
  const ShaderConfig config = corpus::decode(record);
  return variant.normalAttachment != UINT32_MAX && config.attrs[GX_VA_NRM].attrType != GX_NONE &&
         build_shader_info(config).lightingEnabled;
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

gfx::RenderTargetLayout debug_id_layout(bool normal) {
  gfx::RenderTargetLayout layout = scene_layout();
  if (normal) {
    layout.colorAttachments[layout.colorAttachmentCount++] = {gfx::ColorAttachmentSemantic::Normal,
                                                              wgpu::TextureFormat::RGB10A2Unorm};
  }
  layout.colorAttachments[layout.colorAttachmentCount++] = {gfx::ColorAttachmentSemantic::DebugId,
                                                            wgpu::TextureFormat::R32Uint};
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

std::optional<std::vector<float>> run_float_compute(test::TestDevice& device, const std::string& text,
                                                    uint32_t invocations, uint32_t fields) {
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = text.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
  const wgpu::ShaderModule module = device.device.CreateShaderModule(&moduleDescriptor);
  const wgpu::ComputePipelineDescriptor pipelineDescriptor{.compute = {.module = module, .entryPoint = "main"}};
  const wgpu::ComputePipeline pipeline = device.device.CreateComputePipeline(&pipelineDescriptor);
  const uint64_t size = uint64_t{invocations} * fields * sizeof(float);
  const wgpu::BufferDescriptor bufferDescriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                                .size = size};
  const wgpu::Buffer output = device.device.CreateBuffer(&bufferDescriptor);
  const wgpu::BindGroupEntry entry{.binding = 0, .buffer = output, .size = size};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .layout = pipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &entry};
  const wgpu::BindGroup group = device.device.CreateBindGroup(&groupDescriptor);
  const wgpu::CommandEncoder encoder = device.device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group);
  pass.DispatchWorkgroups(invocations / 64);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  device.queue.Submit(1, &commands);
  const auto words = test::read_buffer(device, output, 0, size);
  if (!words || words->size() != uint64_t{invocations} * fields) {
    return std::nullopt;
  }
  std::vector<float> values(words->size());
  std::memcpy(values.data(), words->data(), size);
  return values;
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
      EXPECT_FALSE(legacyError.has_value()) << "core " << core << ": " << legacyError.value_or("");
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

TEST_F(GxShaderExtDevice, LitNormalTargetValidates) {
  std::vector<corpus::Entry> local;
  if (const char* path = std::getenv("AURORA_GX_SHADER_CORPUS"); path != nullptr && *path != '\0') {
    std::string error;
    auto entries = corpus::read_corpus(path, &error);
    ASSERT_TRUE(entries.has_value()) << error;
    local = std::move(*entries);
  }
  const std::vector<corpus::Entry> synthetic = corpus::generate_synthetic();
  for (const bool core : {true, false}) {
    ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = core}, false)) << "core " << core;
    const auto validate_class = [&](const std::vector<corpus::Entry>& entries) {
      std::pair<size_t, size_t> counts{};
      for (const corpus::Entry& entry : entries) {
        if (corpus::validate(entry.record) || !in_allow_class(entry.record)) {
          continue;
        }
        const ShaderConfig config = corpus::decode(entry.record);
        if (module_error(*m_device, build_shader_source(config, corpus::decode_variant(entry.record).dstAlphaMode))) {
          ++counts.second;
          continue;
        }
        const auto error = module_error(*m_device, corpus::generate_wgsl(entry.record));
        EXPECT_FALSE(error.has_value()) << "core " << core << " " << entry.label << ": " << error.value_or("");
        ++counts.first;
      }
      return counts;
    };
    const auto [count, untargetedInvalid] = validate_class(synthetic);
    EXPECT_GE(count, 1u) << "core " << core;
    std::printf("core %d: synthetic allow class %zu validated, %zu with invalid untargeted text\n", core ? 1 : 0, count,
                untargetedInvalid);
    RecordProperty("allowClass", static_cast<int>(count));
    if (!local.empty()) {
      const auto [localCount, localUntargetedInvalid] = validate_class(local);
      std::printf("core %d: local allow class %zu validated, %zu with invalid untargeted text\n", core ? 1 : 0,
                  localCount, localUntargetedInvalid);
      RecordProperty("localAllowClass", static_cast<int>(localCount));
    }
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

  ASSERT_FALSE(hex_cases().empty());
  const PipelineConfig hex = pipeline_for(hex_cases().front().config);
  PipelineConfig legacy = hex;
  legacy.shaderConfig.ext = {};
  PipelineConfig replace = legacy;
  replace.dstAlpha = 0x80;
  PipelineConfig prepass = replace;
  prepass.blendMode = GX_BM_BLEND;
  prepass.blendFacSrc = GX_BL_SRCALPHA;
  prepass.blendFacDst = GX_BL_INVSRCALPHA;
  m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
  for (const bool normal : {false, true}) {
    const gfx::RenderTargetLayout ids = debug_id_layout(normal);
    for (const PipelineConfig& config : {hex, legacy, replace, prepass}) {
      EXPECT_TRUE(static_cast<bool>(create_pipeline(config, ids).main)) << normal;
    }
    EXPECT_TRUE(static_cast<bool>(create_pipeline(prepass, ids).prepass)) << normal;
    for (uint32_t bits = 0; bits < 8; ++bits) {
      const gfx::clear::PipelineConfig clear{
          .clearColor = (bits & 1) != 0, .clearAlpha = (bits & 2) != 0, .clearDepth = (bits & 4) != 0};
      EXPECT_TRUE(static_cast<bool>(gfx::clear::create_pipeline(clear, ids))) << normal << " " << bits;
    }
  }
  const auto error = test::pop_validation_error(*m_device);
  EXPECT_FALSE(error.has_value()) << error.value_or("");
}

TEST_F(GxShaderExtDevice, GBufferLayoutPipelinesValidate) {
  const gfx::SceneLayoutBase base{
      .color = wgpu::TextureFormat::BGRA8Unorm,
      .depth = wgpu::TextureFormat::Depth32Float,
      .size = {64, 64, 1},
      .samples = 1,
  };
  const auto created = [&](const auto& create) {
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const bool built = create();
    const auto error = test::pop_validation_error(*m_device);
    EXPECT_FALSE(error.has_value()) << error.value_or("");
    return built && !error;
  };
  const auto build_clears = [&](const gfx::RenderTargetLayout& layout) {
    for (uint32_t bits = 0; bits < 8; ++bits) {
      const gfx::clear::PipelineConfig clear{
          .clearColor = (bits & 1) != 0, .clearAlpha = (bits & 2) != 0, .clearDepth = (bits & 4) != 0};
      EXPECT_TRUE(created([&] { return static_cast<bool>(gfx::clear::create_pipeline(clear, layout)); })) << bits;
    }
  };

  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true}, true));
  ASSERT_TRUE(m_device->core);
  const gfx::RenderTargetLayout gbuffer = gfx::compose_scene_layout({.gbuffer = true}, base);
  const gfx::RenderTargetLayout ids = gfx::compose_scene_layout({.gbuffer = true, .debugId = true}, base);
  EXPECT_EQ(gfx::compose_scene_layout({.normal = true, .gbuffer = true}, base).key, gbuffer.key);

  std::vector<ShaderConfig> configs;
  std::unordered_set<uint64_t> seen;
  size_t invalidLegacy = 0;
  for (const corpus::Entry& entry : corpus::generate_synthetic()) {
    ShaderConfig config = corpus::decode(entry.record);
    config.ext = {};
    if (!seen.insert(xxh3_hash(config)).second) {
      continue;
    }
    if (module_error(*m_device, build_shader_source(config, DstAlphaMode::None))) {
      ++invalidLegacy;
      continue;
    }
    configs.push_back(config);
  }
  ASSERT_FALSE(configs.empty());
  size_t litNormal = 0;
  for (const ShaderConfig& config : configs) {
    const PipelineConfig plain = pipeline_for(config);
    PipelineConfig prepass = plain;
    prepass.dstAlpha = 0x80;
    prepass.blendMode = GX_BM_BLEND;
    prepass.blendFacSrc = GX_BL_SRCALPHA;
    prepass.blendFacDst = GX_BL_INVSRCALPHA;
    PipelineConfig colorOff = plain;
    colorOff.colorUpdate = false;
    EXPECT_TRUE(created([&] { return static_cast<bool>(create_pipeline(plain, gbuffer).main); }));
    EXPECT_TRUE(created([&] { return static_cast<bool>(create_pipeline(plain, ids).main); }));
    EXPECT_TRUE(created([&] {
      const gfx::CompiledPipeline pipeline = create_pipeline(prepass, gbuffer);
      return pipeline.main && pipeline.prepass;
    }));
    EXPECT_TRUE(created([&] { return static_cast<bool>(create_pipeline(colorOff, gbuffer).main); }));
    litNormal += config.attrs[GX_VA_NRM].attrType != GX_NONE && build_shader_info(config).lightingEnabled ? 1 : 0;
  }
  EXPECT_GE(litNormal, 1u);
  build_clears(gbuffer);
  build_clears(ids);
  std::printf("gbuffer layouts: %zu configs built, %zu lit with normals, %zu with invalid legacy text\n",
              configs.size(), litNormal, invalidLegacy);
  RecordProperty("configs", static_cast<int>(configs.size()));
  m_globals.reset();
  EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
  m_device.reset();

  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true, .profile = FEATURE_PROFILE_METAL32}, true));
  wgpu::Limits limits{};
  m_device->device.GetLimits(&limits);
  if (!m_device->core || limits.maxColorAttachmentBytesPerSample != 32) {
    GTEST_SKIP() << "no 32-byte core device";
  }
  const PipelineConfig plain = pipeline_for(configs.front());
  for (const gfx::RenderTargetLayout& layout : {gbuffer, ids}) {
    EXPECT_TRUE(created([&] { return static_cast<bool>(create_pipeline(plain, layout).main); }));
    build_clears(layout);
  }
}

// Unique ext-0 synthetic configs; those whose legacy text already fails fail for older reasons
std::vector<ShaderConfig> legacy_valid_configs(test::TestDevice& device, size_t limit = SIZE_MAX) {
  std::vector<ShaderConfig> configs;
  std::unordered_set<uint64_t> seen;
  for (const corpus::Entry& entry : corpus::generate_synthetic()) {
    ShaderConfig config = corpus::decode(entry.record);
    config.ext = {};
    if (!seen.insert(xxh3_hash(config)).second ||
        module_error(device, build_shader_source(config, DstAlphaMode::None))) {
      continue;
    }
    configs.push_back(config);
    if (configs.size() == limit) {
      break;
    }
  }
  return configs;
}

gfx::RenderTargetLayout gbuffer_layout() {
  return gfx::compose_scene_layout({.gbuffer = true}, {.color = wgpu::TextureFormat::BGRA8Unorm,
                                                       .depth = wgpu::TextureFormat::Depth32Float,
                                                       .size = {64, 64, 1},
                                                       .samples = 1});
}

PipelineConfig blended(PipelineConfig config) {
  config.blendMode = GX_BM_BLEND;
  config.blendFacSrc = GX_BL_SRCALPHA;
  config.blendFacDst = GX_BL_INVSRCALPHA;
  config.depthUpdate = false;
  return config;
}

TEST_F(GxShaderExtDevice, MotionPipelinesValidate) {
  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true}, true));
  gfx::set_blocking_pipelines(true);
  const gfx::RenderTargetLayout layout = gbuffer_layout();
  const ShaderTargets targets = shader_targets(layout);
  ASSERT_NE(targets.motion, UINT32_MAX);
  ASSERT_NE(targets.reactive, UINT32_MAX);
  const std::vector<ShaderConfig> configs = legacy_valid_configs(*m_device);
  ASSERT_FALSE(configs.empty());
  const uint64_t failures = gbuffer_pipeline_failures();
  size_t built = 0;
  for (const ShaderConfig& config : configs) {
    const PipelineConfig opaque = pipeline_for(config);
    PipelineConfig noDepth = opaque;
    noDepth.depthUpdate = false;
    PipelineConfig prepass = blended(opaque);
    prepass.dstAlpha = 0x80;
    for (const PipelineConfig& variant : {opaque, noDepth, blended(opaque)}) {
      EXPECT_TRUE(static_cast<bool>(create_pipeline(variant, layout).main));
      ++built;
    }
    const gfx::CompiledPipeline split = create_pipeline(prepass, layout);
    EXPECT_TRUE(split.main && split.prepass);
    ++built;
  }
  EXPECT_EQ(gbuffer_pipeline_failures(), failures);
  EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
  std::printf("motion pipelines: %zu configs, %zu pipelines\n", configs.size(), built);
  RecordProperty("configs", static_cast<int>(configs.size()));

  const char* path = std::getenv("AURORA_GX_SHADER_CORPUS");
  if (path == nullptr || *path == '\0') {
    return;
  }
  std::string error;
  const auto entries = corpus::read_corpus(path, &error);
  ASSERT_TRUE(entries.has_value()) << error;
  size_t validated = 0;
  size_t skipped = 0;
  for (const corpus::Entry& entry : *entries) {
    const ShaderConfig config = corpus::decode(entry.record);
    const DstAlphaMode mode = corpus::decode_variant(entry.record).dstAlphaMode;
    if (mode == DstAlphaMode::DualSource || module_error(*m_device, build_shader_source(config, mode))) {
      ++skipped;
      continue;
    }
    const auto failure = module_error(*m_device, build_shader_source(config, mode, targets));
    EXPECT_FALSE(failure.has_value()) << entry.label << ": " << failure.value_or("");
    ++validated;
  }
  EXPECT_GT(validated, 0u);
  std::printf("TP corpus: %zu tier-B texts validated, %zu skipped\n", validated, skipped);
  RecordProperty("corpusValidated", static_cast<int>(validated));
}

TEST_F(GxShaderExtDevice, GbufferPipelineFailureIsCaptured) {
  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true}, true));
  gfx::set_blocking_pipelines(true);
  const gfx::RenderTargetLayout control = gbuffer_layout();
  gfx::RenderTargetLayout broken = control;
  const uint32_t motion = shader_targets(control).motion;
  ASSERT_NE(motion, UINT32_MAX);
  broken.colorAttachments[motion].format = wgpu::TextureFormat::R32Uint;
  gfx::detail::finalize_render_target_layout(broken);
  ASSERT_NE(broken.key, control.key);
  const std::vector<ShaderConfig> configs = legacy_valid_configs(*m_device, 1);
  ASSERT_FALSE(configs.empty());
  const PipelineConfig config = pipeline_for(configs.front());

  const uint64_t failures = gbuffer_pipeline_failures();
  EXPECT_EQ(gfx::pipeline_state(gfx::find_debug_pipeline(config, control)), gfx::PipelineState::Ready);
  EXPECT_EQ(gbuffer_pipeline_failures(), failures);
  const gfx::PipelineRef ref = gfx::find_debug_pipeline(config, broken);
  EXPECT_EQ(gfx::pipeline_state(ref), gfx::PipelineState::Failed);
  gfx::CompiledPipeline compiled;
  EXPECT_FALSE(gfx::get_pipeline(ref, compiled));
  EXPECT_EQ(gbuffer_pipeline_failures(), failures + 1);
  EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
}

gfx::RenderTargetLayout gbuffer_id_layout() {
  return gfx::compose_scene_layout({.gbuffer = true, .debugId = true}, {.color = wgpu::TextureFormat::BGRA8Unorm,
                                                                        .depth = wgpu::TextureFormat::Depth32Float,
                                                                        .size = {64, 64, 1},
                                                                        .samples = 1});
}

ShaderConfig with_output(ShaderConfig config, u8 output) {
  config.ext.output = output;
  return config;
}

bool has_normal(const ShaderConfig& config) { return config.attrs[GX_VA_NRM].attrType != GX_NONE; }

TEST_F(GxShaderExtDevice, GbufferMarkerValidates) {
  constexpr size_t PerClass = 8;
  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true}, true));
  gfx::set_blocking_pipelines(true);
  std::array<std::vector<ShaderConfig>, 3> classes;
  for (const ShaderConfig& config : legacy_valid_configs(*m_device)) {
    const size_t index = !has_normal(config) ? 2 : build_shader_info(config).lightingEnabled ? 0 : 1;
    if (classes[index].size() < PerClass) {
      classes[index].push_back(config);
    }
  }
  const uint64_t failures = gbuffer_pipeline_failures();
  size_t checked = 0;
  for (size_t index = 0; index < classes.size(); ++index) {
    ASSERT_FALSE(classes[index].empty()) << "class " << index;
    for (const gfx::RenderTargetLayout& layout : {gbuffer_layout(), gbuffer_id_layout()}) {
      const ShaderTargets targets = shader_targets(layout);
      ASSERT_NE(targets.gbufferNormal, UINT32_MAX);
      for (const ShaderConfig& base : classes[index]) {
        for (const u8 output : {u8{0}, u8{1}}) {
          const ShaderConfig config = with_output(base, output);
          for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
            const std::string text = build_shader_source(config, mode, targets);
            EXPECT_EQ(text.find("gbuffer_rx_alpha(imm._pad") != std::string::npos, output == 1 && index != 2);
            const auto error = module_error(*m_device, text);
            EXPECT_FALSE(error.has_value())
                << "class " << index << " output " << int{output} << ": " << error.value_or("");
            PipelineConfig pipeline = pipeline_for(config);
            pipeline.dstAlpha = mode == DstAlphaMode::Replace ? 0x80 : UINT32_MAX;
            EXPECT_EQ(gfx::pipeline_state(gfx::find_debug_pipeline(pipeline, layout)), gfx::PipelineState::Ready)
                << "class " << index << " output " << int{output};
            ++checked;
          }
        }
      }
    }
  }
  EXPECT_EQ(gbuffer_pipeline_failures(), failures);
  EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
  RecordProperty("checked", static_cast<int>(checked));
}

TEST_F(GxShaderExtDevice, WarmSkipsOutputRowsWithoutTargets) {
  ASSERT_TRUE(start({.backend = wgpu::BackendType::Null, .core = true}, true));
  gfx::set_blocking_pipelines(true);
  const std::vector<ShaderConfig> configs = legacy_valid_configs(*m_device, 1);
  ASSERT_FALSE(configs.empty());
  const gfx::RenderTargetLayout gbuffer = gbuffer_layout();
  const std::array without{scene_layout(), debug_id_layout(true)};
  const auto known = [](const gfx::RenderTargetLayout& layout) {
    return gfx::warm_known_pipelines(layout, gfx::ShaderType::GX).total;
  };
  const uint32_t before = known(gbuffer);
  const std::array<uint32_t, 2> beforeWithout{known(without[0]), known(without[1])};
  const PipelineConfig plain = pipeline_for(configs.front());
  const PipelineConfig marker = pipeline_for(with_output(configs.front(), 1));
  EXPECT_EQ(gfx::pipeline_state(gfx::find_pipeline(plain, gbuffer)), gfx::PipelineState::Ready);
  EXPECT_EQ(gfx::pipeline_state(gfx::find_pipeline(marker, gbuffer)), gfx::PipelineState::Ready);
  EXPECT_EQ(known(gbuffer), before + 2);
  for (size_t i = 0; i < without.size(); ++i) {
    EXPECT_EQ(known(without[i]), beforeWithout[i] + 1) << i;
  }
}

TEST_F(GxShaderExtDevice, MotionPipelinesCompileOnBackend) {
  const auto backend = hardware_backend();
  if (!backend) {
    GTEST_SKIP() << "AURORA_TEST_BACKEND names no hardware backend";
  }
  if (!start({.backend = *backend, .core = true, .allowCpu = false}, true)) {
    GTEST_SKIP() << "no adapter";
  }
  const gfx::RenderTargetLayout layout = gbuffer_layout();
  const std::vector<ShaderConfig> configs = legacy_valid_configs(*m_device, 64);
  ASSERT_EQ(configs.size(), 64u);
  const uint64_t failures = gbuffer_pipeline_failures();
  size_t compiled = 0;
  for (const ShaderConfig& config : configs) {
    for (const PipelineConfig& variant : {pipeline_for(config), blended(pipeline_for(config))}) {
      const bool built = static_cast<bool>(create_pipeline(variant, layout).main);
      EXPECT_TRUE(built);
      compiled += built ? 1 : 0;
    }
  }
  EXPECT_EQ(gbuffer_pipeline_failures(), failures);
  RecordProperty("compiled", static_cast<int>(compiled));
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
  const auto values = run_float_compute(*m_device, text, Points, Fields);
  ASSERT_TRUE(values.has_value());
  size_t checked = 0;
  for (uint32_t i = 0; i < Points; ++i) {
    const float* v = &(*values)[i * Fields];
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

TEST_F(GxShaderExtDevice, HexWeightsStayFiniteAtHighSharpness) {
  const auto backend =
      test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
  if (!start({.backend = backend, .core = true}, false)) {
    GTEST_SKIP() << "no adapter";
  }
  if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
    GTEST_SKIP() << "the null backend runs no shaders";
  }
  constexpr uint32_t Points = 1024;
  constexpr uint32_t Fields = 4;
  constexpr std::array<float, 3> Exponents{7.f, 16.f, 32.f};
  std::string text;
  append_hex_common(text);
  text += R"(
@group(0) @binding(0) var<storage, read_write> out_words: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x % 1024u;
    let k = id.x / 1024u;
    let e = select(select(7.0, 16.0, k == 1u), 32.0, k == 2u);
    let uv = vec2f(f32(i % 32u), f32(i / 32u)) * 0.0731 + vec2f(0.113, -1.29);
    let bary = select(hex_taps(uv, 1.0, 1.0, vec2f(256.0)).bary, vec3f(1.0 / 3.0), i == 0u);
    let grey = vec4f(0.5, 0.5, 0.5, 1.0);
    let g = hex_weights(bary, grey, grey, grey, 0.6, e);
    let shade = g.x * grey + g.y * grey + g.z * grey;
    let c0 = vec4f(0.9, 0.2, 0.1, 1.0);
    let c1 = vec4f(0.1, 0.6, 0.3, 1.0);
    let c2 = vec4f(0.3, 0.3, 0.8, 1.0);
    let w = hex_weights(bary, c0, c1, c2, 0.6, e);
    let lw = vec3f(0.299, 0.587, 0.114);
    let lum = vec3f(dot(c0.rgb, lw), dot(c1.rgb, lw), dot(c2.rgb, lw));
    let prior = mix(vec3f(1.0), lum, 0.6) * pow(max(bary, vec3f(0.0)), vec3f(e));
    let priorSum = prior.x + prior.y + prior.z;
    let d = abs(w - prior / max(priorSum, 1e-6));
    out_words[id.x * 4u + 0u] = shade.r;
    out_words[id.x * 4u + 1u] = g.x + g.y + g.z;
    out_words[id.x * 4u + 2u] = priorSum;
    out_words[id.x * 4u + 3u] = max(d.x, max(d.y, d.z));
}
)";
  const auto values = run_float_compute(*m_device, text, Points * static_cast<uint32_t>(Exponents.size()), Fields);
  ASSERT_TRUE(values.has_value());
  for (size_t k = 0; k < Exponents.size(); ++k) {
    size_t underflowed = 0;
    size_t compared = 0;
    for (uint32_t i = 0; i < Points; ++i) {
      const float* v = &(*values)[(k * Points + i) * Fields];
      EXPECT_NEAR(v[0], 0.5f, 1e-3f) << "exponent " << Exponents[k] << " point " << i;
      EXPECT_NEAR(v[1], 1.f, 1e-4f) << "exponent " << Exponents[k] << " point " << i;
      if (v[2] < 1e-6f) {
        ++underflowed;
      } else if (v[2] > 1e-4f) {
        // Where the unscaled weights never neared the epsilon, the scaled ones must give the same blend
        EXPECT_LT(v[3], 2e-4f) << "exponent " << Exponents[k] << " point " << i;
        ++compared;
      }
    }
    EXPECT_GE(compared, Points / 8) << "exponent " << Exponents[k];
    if (Exponents[k] >= 16.f) {
      EXPECT_GE(underflowed, 1u) << "exponent " << Exponents[k];
    }
    if (Exponents[k] >= 32.f) {
      EXPECT_GE(underflowed, Points / 4) << "exponent " << Exponents[k];
    }
  }
}

TEST_F(GxShaderExtDevice, HexWeightsKeepAlphaOnBlackTaps) {
  const auto backend =
      test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
  if (!start({.backend = backend, .core = true}, false)) {
    GTEST_SKIP() << "no adapter";
  }
  if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
    GTEST_SKIP() << "the null backend runs no shaders";
  }
  constexpr uint32_t Points = 256;
  constexpr uint32_t Fields = 4;
  std::string text;
  append_hex_common(text);
  text += R"(
@group(0) @binding(0) var<storage, read_write> out_words: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let uv = vec2f(f32(i % 16u), f32(i / 16u)) * 0.0731 + vec2f(0.113, -1.29);
    let bary = select(hex_taps(uv, 1.0, 1.0, vec2f(256.0)).bary, vec3f(1.0 / 3.0), i == 0u);
    let black = vec4f(0.0, 0.0, 0.0, 0.75);
    let w = hex_weights(bary, black, black, black, 1.0, 32.0);
    let c = w.x * black + w.y * black + w.z * black;
    out_words[i * 4u + 0u] = w.x + w.y + w.z;
    out_words[i * 4u + 1u] = c.a;
    out_words[i * 4u + 2u] = max(c.r, max(c.g, c.b));
    out_words[i * 4u + 3u] = min(w.x, min(w.y, w.z));
}
)";
  const auto values = run_float_compute(*m_device, text, Points, Fields);
  ASSERT_TRUE(values.has_value());
  for (uint32_t i = 0; i < Points; ++i) {
    const float* v = &(*values)[i * Fields];
    EXPECT_NEAR(v[0], 1.f, 1e-4f) << "point " << i;
    EXPECT_NEAR(v[1], 0.75f, 1e-4f) << "point " << i;
    EXPECT_EQ(v[2], 0.f) << "point " << i;
    EXPECT_GE(v[3], 0.f) << "point " << i;
  }
  EXPECT_NEAR((*values)[3], 1.f / 3.f, 1e-4f) << "centroid";
}
TEST_F(GxShaderExtDevice, ReceiverAlphaRoundTripsThroughRgb10a2) {
  const auto backend = hardware_backend();
  if (!backend) {
    GTEST_SKIP() << "AURORA_TEST_BACKEND names no hardware backend";
  }
  if (!start({.backend = *backend, .core = true, .allowCpu = false}, false)) {
    GTEST_SKIP() << "no adapter";
  }
  constexpr uint32_t Width = 6;
  const std::string text = "@group(0) @binding(0) var<storage, read> abuf: array<u32>;\n" + receiver_alpha_wgsl() + R"(

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let x = u32(pos.x);
    let rec = select(select(4u + x * 8u, 0u, x == 4u), 28u, x == 5u);
    let nrm = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0), x == 5u);
    let nrm_len_sq = dot(nrm, nrm);
    return vec4f(nrm * 0.5 + 0.5, gbuffer_rx_alpha(rec, nrm_len_sq));
}
)";
  std::array<uint32_t, 40> words{};
  words.fill(0xFFFFFFFFu);
  for (uint32_t code = 0; code < 4; ++code) {
    words[4 + code * 8 + render_ext::DrawRecordFlagsWord] = (code << render_ext::DrawRecordReceiverShift) | 0xFF3Fu;
  }
  const wgpu::BufferDescriptor bufferDescriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                .size = sizeof(words)};
  const wgpu::Buffer records = m_device->device.CreateBuffer(&bufferDescriptor);
  m_device->queue.WriteBuffer(records, 0, words.data(), sizeof(words));
  const wgpu::TextureDescriptor textureDescriptor{
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .size = {Width, 1, 1},
      .format = wgpu::TextureFormat::RGB10A2Unorm,
  };
  const wgpu::Texture target = m_device->device.CreateTexture(&textureDescriptor);

  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = text.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
  const wgpu::ShaderModule module = m_device->device.CreateShaderModule(&moduleDescriptor);
  const wgpu::ColorTargetState colorTarget{.format = wgpu::TextureFormat::RGB10A2Unorm};
  const wgpu::FragmentState fragment{
      .module = module, .entryPoint = "fs_main", .targetCount = 1, .targets = &colorTarget};
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .fragment = &fragment,
  };
  const wgpu::RenderPipeline pipeline = m_device->device.CreateRenderPipeline(&pipelineDescriptor);
  const wgpu::BindGroupEntry entry{.binding = 0, .buffer = records, .size = sizeof(words)};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .layout = pipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &entry};
  const wgpu::BindGroup group = m_device->device.CreateBindGroup(&groupDescriptor);

  const wgpu::RenderPassColorAttachment attachment{
      .view = target.CreateView(),
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group);
  pass.Draw(3);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);

  const std::vector<float> texels = test::read_texture_mip(*m_device, target, 0);
  ASSERT_EQ(texels.size(), Width);
  constexpr std::array<uint32_t, Width> Expected{0, 1, 2, 3, 0, 0};
  for (uint32_t x = 0; x < Width; ++x) {
    EXPECT_EQ(std::bit_cast<uint32_t>(texels[x]) >> 30, Expected[x]) << "x " << x;
  }
}

TEST_F(GxShaderExtDevice, GbufferMarkerPipelinesCompileOnBackend) {
  const auto backend = hardware_backend();
  if (!backend) {
    GTEST_SKIP() << "AURORA_TEST_BACKEND names no hardware backend";
  }
  if (!start({.backend = *backend, .core = true, .allowCpu = false}, true)) {
    GTEST_SKIP() << "no adapter";
  }
  const gfx::RenderTargetLayout layout = gbuffer_layout();
  std::vector<ShaderConfig> configs = legacy_valid_configs(*m_device);
  // Normals first, so the receiver path is what compiles
  std::ranges::stable_partition(configs, has_normal);
  ASSERT_GE(configs.size(), 64u);
  ASSERT_TRUE(has_normal(configs.front()));
  configs.resize(64);
  const uint64_t failures = gbuffer_pipeline_failures();
  size_t compiled = 0;
  for (const ShaderConfig& config : configs) {
    const PipelineConfig marker = pipeline_for(with_output(config, 1));
    for (const PipelineConfig& variant : {marker, blended(marker)}) {
      const bool built = static_cast<bool>(create_pipeline(variant, layout).main);
      EXPECT_TRUE(built);
      compiled += built ? 1 : 0;
    }
  }
  EXPECT_EQ(gbuffer_pipeline_failures(), failures);
  RecordProperty("compiled", static_cast<int>(compiled));
}
} // namespace
} // namespace aurora::gx
