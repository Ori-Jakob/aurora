// The render sizes, jitter phase count, depth reconstruction factors and SPD setup are ported from AMD FidelityFX
// Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0, ffx_fsr3upscaler.cpp and spd/ffx_spd.h),
// which carry this notice:
//
// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "fsr.hpp"

#include "../../../graph_layout.hpp"

#include "../../../../webgpu/gpu.hpp"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace aurora::render::detail::upscale::fsr {
namespace {
using testing::Binding;
using testing::Kind;
using wgpu::TextureFormat;

constexpr float DefaultUnitsPerMeter = 100.f;
constexpr uint64_t LayoutSalt = 0x4653'5233'5550'5343ull;
constexpr std::array<std::string_view, 1> LocksR8Defines{"FSR_LOCKS_R8"};

constexpr std::array<std::string_view, StageCount> ShaderIds{
    "post/upscale/fsr/fsr_prepare_inputs",   "post/upscale/fsr/fsr_luma_pyramid_a",
    "post/upscale/fsr/fsr_luma_pyramid_b",   "post/upscale/fsr/fsr_shading_change_pyramid",
    "post/upscale/fsr/fsr_shading_change",   "post/upscale/fsr/fsr_prepare_reactivity",
    "post/upscale/fsr/fsr_luma_instability", "post/upscale/fsr/fsr_accumulate",
};

constexpr std::array<Binding, 8> PrepareInputsLayout{{
    {Kind::Texture},
    {Kind::UnfilterableTexture},
    {Kind::Texture},
    {Kind::StorageWrite, TextureFormat::RG16Float},
    {Kind::StorageWrite, TextureFormat::R32Float},
    {Kind::Buffer},
    {Kind::StorageWrite, TextureFormat::R16Float},
    {Kind::StorageWrite, TextureFormat::R16Float},
}};
constexpr std::array<Binding, 6> LumaPyramidALayout{{
    {Kind::Texture},
    {Kind::Texture},
    {Kind::UnfilterableTexture},
    {Kind::StorageWrite, TextureFormat::R16Float},
    {Kind::StorageWrite, TextureFormat::RG16Float},
    {Kind::StorageWrite, TextureFormat::RGBA32Float},
}};
constexpr std::array<Binding, 3> LumaPyramidBLayout{{
    {Kind::Texture},
    {Kind::UnfilterableTexture},
    {Kind::StorageWrite, TextureFormat::RGBA32Float},
}};
constexpr std::array<Binding, 6> ShadingChangePyramidLayout{{
    {Kind::Texture},
    {Kind::Texture},
    {Kind::Texture},
    {Kind::StorageWrite, TextureFormat::RG16Float},
    {Kind::StorageWrite, TextureFormat::RG16Float},
    {Kind::StorageWrite, TextureFormat::RG16Float},
}};
constexpr std::array<Binding, 2> ShadingChangeLayout{{
    {Kind::Texture},
    {Kind::StorageWrite, TextureFormat::R8Unorm},
}};
constexpr std::array<Binding, 10> prepare_reactivity_layout(TextureFormat locks) {
  return {{
      {Kind::ReadOnlyBuffer},
      {Kind::Texture},
      {Kind::UnfilterableTexture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::StorageWrite, TextureFormat::RGBA8Unorm},
      {Kind::StorageWrite, locks},
      {Kind::StorageWrite, TextureFormat::R8Unorm},
  }};
}
constexpr std::array<Binding, 10> PrepareReactivityLayout = prepare_reactivity_layout(TextureFormat::R32Float);
constexpr std::array<Binding, 10> PrepareReactivityR8Layout = prepare_reactivity_layout(TextureFormat::R8Unorm);
constexpr std::array<Binding, 6> LumaInstabilityLayout{{
    {Kind::Texture},
    {Kind::Texture},
    {Kind::Texture},
    {Kind::Texture},
    {Kind::StorageWrite, TextureFormat::RGBA16Float},
    {Kind::StorageWrite, TextureFormat::R16Float},
}};
constexpr std::array<Binding, 8> accumulate_layout(TextureFormat locks) {
  return {{
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::Texture},
      {Kind::StorageWrite, TextureFormat::RGBA16Float},
      {Kind::StorageReadWrite, locks},
  }};
}
constexpr std::array<Binding, 8> AccumulateLayout = accumulate_layout(TextureFormat::R32Float);
constexpr std::array<Binding, 8> AccumulateR8Layout = accumulate_layout(TextureFormat::R8Unorm);

struct PipelineLayoutEntry {
  wgpu::BindGroupLayout frameLayout;
  bool locksR8 = false;
  std::array<wgpu::PipelineLayout, StageCount> layouts;
};

struct Shared {
  std::array<std::array<wgpu::BindGroupLayout, StageCount>, 2> groupLayouts;
  std::vector<PipelineLayoutEntry> pipelineLayouts;
};

struct TestShared {
  wgpu::BindGroupLayout frameLayout;
  std::array<Pipelines, 2> pipelines;
  std::array<bool, 2> requested{};
  wgpu::Buffer constants;
  wgpu::Buffer params;
  std::array<wgpu::Sampler, 4> samplers;
};

std::mutex g_mutex;
Shared g_shared;
TestShared g_test;

constexpr size_t index_of(Stage stage) noexcept { return static_cast<size_t>(stage); }

constexpr uint32_t groups(uint32_t size, uint32_t group) noexcept { return (size + group - 1) / group; }

constexpr bool uses_locks(Stage stage) noexcept {
  return stage == Stage::PrepareReactivity || stage == Stage::Accumulate;
}

wgpu::BindGroupLayoutEntry layout_entry(uint32_t binding, const Binding& info) {
  wgpu::BindGroupLayoutEntry entry{.binding = binding, .visibility = wgpu::ShaderStage::Compute};
  switch (info.kind) {
  case Kind::Texture:
    entry.texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D};
    break;
  case Kind::UnfilterableTexture:
    entry.texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                     .viewDimension = wgpu::TextureViewDimension::e2D};
    break;
  case Kind::StorageWrite:
    entry.storageTexture = {.access = wgpu::StorageTextureAccess::WriteOnly,
                            .format = info.format,
                            .viewDimension = wgpu::TextureViewDimension::e2D};
    break;
  case Kind::StorageReadWrite:
    entry.storageTexture = {.access = wgpu::StorageTextureAccess::ReadWrite,
                            .format = info.format,
                            .viewDimension = wgpu::TextureViewDimension::e2D};
    break;
  case Kind::Buffer:
    entry.buffer = {.type = wgpu::BufferBindingType::Storage};
    break;
  case Kind::ReadOnlyBuffer:
    entry.buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage};
    break;
  }
  return entry;
}

const wgpu::BindGroupLayout& group_layout_locked(Stage stage, bool locksR8) {
  wgpu::BindGroupLayout& layout = g_shared.groupLayouts[locksR8 ? 1 : 0][index_of(stage)];
  if (!layout) {
    const std::span<const Binding> bindings = testing::layout(stage, locksR8);
    std::vector<wgpu::BindGroupLayoutEntry> entries;
    entries.reserve(bindings.size());
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      entries.push_back(layout_entry(i, bindings[i]));
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "aurora.fsr",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    layout = webgpu::g_device.CreateBindGroupLayout(&descriptor);
  }
  return layout;
}

const PipelineLayoutEntry& pipeline_layouts_locked(const wgpu::BindGroupLayout& frameLayout, bool locksR8) {
  for (const PipelineLayoutEntry& entry : g_shared.pipelineLayouts) {
    if (entry.frameLayout.Get() == frameLayout.Get() && entry.locksR8 == locksR8) {
      return entry;
    }
  }
  PipelineLayoutEntry& entry = g_shared.pipelineLayouts.emplace_back();
  entry.frameLayout = frameLayout;
  entry.locksR8 = locksR8;
  for (size_t i = 0; i < StageCount; ++i) {
    const std::array groupLayouts{frameLayout, group_layout_locked(static_cast<Stage>(i), locksR8)};
    const wgpu::PipelineLayoutDescriptor descriptor{
        .label = "aurora.fsr",
        .bindGroupLayoutCount = groupLayouts.size(),
        .bindGroupLayouts = groupLayouts.data(),
    };
    entry.layouts[i] = webgpu::g_device.CreatePipelineLayout(&descriptor);
  }
  return entry;
}

wgpu::BindGroupEntry texture(uint32_t binding, const wgpu::TextureView& view) {
  return {.binding = binding, .textureView = view};
}

wgpu::BindGroupEntry buffer(uint32_t binding, const BufferView& view) {
  return {.binding = binding, .buffer = view.buffer, .offset = view.offset, .size = view.size};
}

std::vector<wgpu::BindGroupEntry> group_entries(Stage stage, const Bindings& b) {
  switch (stage) {
  case Stage::PrepareInputs:
    return {texture(0, b.motion),       texture(1, b.depth), texture(2, b.color),         texture(3, b.dilatedMotion),
            texture(4, b.dilatedDepth), buffer(5, b.recon),  texture(6, b.farthestDepth), texture(7, b.luma)};
  case Stage::LumaPyramidA:
    return {texture(0, b.luma),          texture(1, b.farthestDepth),
            texture(2, b.prevFrameInfo), texture(3, b.farthestDepthMip1),
            texture(4, b.lumaMip5),      texture(5, b.frameInfo)};
  case Stage::LumaPyramidB:
    return {texture(0, b.lumaMip5), texture(1, b.prevFrameInfo), texture(2, b.frameInfo)};
  case Stage::ShadingChangePyramid:
    return {texture(0, b.luma),       texture(1, b.prevLuma),   texture(2, b.dilatedMotion),
            texture(3, b.spdMips[0]), texture(4, b.spdMips[1]), texture(5, b.spdMips[2])};
  case Stage::ShadingChange:
    return {texture(0, b.spd), texture(1, b.shadingChange)};
  case Stage::PrepareReactivity:
    return {buffer(0, b.recon),        texture(1, b.dilatedMotion),    texture(2, b.dilatedDepth),
            texture(3, b.reactive),    texture(4, b.prevAccumulation), texture(5, b.shadingChange),
            texture(6, b.luma),        texture(7, b.dilatedReactive),  texture(8, b.locks),
            texture(9, b.accumulation)};
  case Stage::LumaInstability:
    return {texture(0, b.dilatedReactive), texture(1, b.dilatedMotion),
            texture(2, b.prevLumaHistory), texture(3, b.luma),
            texture(4, b.lumaHistory),     texture(5, b.lumaInstability)};
  case Stage::Accumulate:
    return {texture(0, b.dilatedReactive), texture(1, b.dilatedMotion),
            texture(2, b.prevUpscaled),    texture(3, b.farthestDepthMip1),
            texture(4, b.lumaInstability), texture(5, b.color),
            texture(6, b.upscaled),        texture(7, b.locks)};
  }
  return {};
}

std::array<uint32_t, 2> dispatch_size(Stage stage, GraphExtent render, GraphExtent output) noexcept {
  const GraphExtent tiles = spd_tiles(render);
  switch (stage) {
  case Stage::PrepareInputs:
  case Stage::PrepareReactivity:
  case Stage::LumaInstability:
    return {groups(render.width, GroupSize), groups(render.height, GroupSize)};
  case Stage::LumaPyramidA:
  case Stage::ShadingChangePyramid:
    return {tiles.width, tiles.height};
  case Stage::LumaPyramidB:
    return spd_mips(render) > LumaTailMip ? std::array<uint32_t, 2>{1, 1} : std::array<uint32_t, 2>{0, 0};
  case Stage::ShadingChange:
    return {groups(render.width / 2, GroupSize), groups(render.height / 2, GroupSize)};
  case Stage::Accumulate:
    return {groups(output.width, GroupSize), groups(output.height, GroupSize)};
  }
  return {0, 0};
}

wgpu::Texture create_texture(const wgpu::Device& device, const char* label, GraphExtent size, TextureFormat format,
                             uint32_t mips = 1) {
  const wgpu::TextureDescriptor descriptor{
      .label = label,
      .usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
               wgpu::TextureUsage::CopyDst,
      .size = {std::max(1u, size.width), std::max(1u, size.height), 1},
      .format = format,
      .mipLevelCount = mips,
  };
  return device.CreateTexture(&descriptor);
}

GraphExtent scaled(GraphExtent size, uint32_t divisor) noexcept {
  return {std::max(1u, groups(size.width, divisor)), std::max(1u, groups(size.height, divisor))};
}

wgpu::TextureView mip_view(const wgpu::Texture& texture, uint32_t mip) {
  const wgpu::TextureViewDescriptor descriptor{.baseMipLevel = mip, .mipLevelCount = 1};
  return texture.CreateView(&descriptor);
}

void ensure_test_shared_locked(const wgpu::Device& device) {
  if (g_test.frameLayout) {
    return;
  }
  const std::span<const wgpu::BindGroupLayoutEntry> entries = frame_layout_entries();
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "aurora.fsr test frame",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_test.frameLayout = device.CreateBindGroupLayout(&descriptor);
  const wgpu::BufferDescriptor constants{
      .label = "aurora.fsr test constants",
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = FrameUniformsSize,
  };
  g_test.constants = device.CreateBuffer(&constants);
  const wgpu::BufferDescriptor params{
      .label = "aurora.fsr test params",
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = MaxPassParamsSize,
  };
  g_test.params = device.CreateBuffer(&params);
  const wgpu::SamplerDescriptor linear{
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
  };
  const wgpu::SamplerDescriptor repeat{
      .addressModeU = wgpu::AddressMode::Repeat,
      .addressModeV = wgpu::AddressMode::Repeat,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  const wgpu::SamplerDescriptor compare{.compare = wgpu::CompareFunction::GreaterEqual};
  g_test.samplers = {device.CreateSampler(&linear), device.CreateSampler(), device.CreateSampler(&repeat),
                     device.CreateSampler(&compare)};
}

void create_state_resources(const wgpu::Device& device, testing::State& state, GraphExtent render, GraphExtent output,
                            bool locksR8) {
  state = testing::State{.render = render, .output = output, .locksR8 = locksR8};
  const GraphExtent half = scaled(render, 2);
  state.dilatedDepth = create_texture(device, "fsr.dilated_depth", render, TextureFormat::R32Float);
  state.dilatedMotion = create_texture(device, "fsr.dilated_motion", render, TextureFormat::RG16Float);
  state.farthestDepth = create_texture(device, "fsr.farthest_depth", render, TextureFormat::R16Float);
  state.farthestDepthMip1 = create_texture(device, "fsr.farthest_depth_mip1", half, TextureFormat::R16Float);
  state.lumaMip5 = create_texture(device, "fsr.luma_mip5", scaled(render, TileSize), TextureFormat::RG16Float);
  state.spd = create_texture(device, "fsr.spd", half, TextureFormat::RG16Float, 3);
  state.shadingChange = create_texture(device, "fsr.shading_change", half, TextureFormat::R8Unorm);
  state.dilatedReactive = create_texture(device, "fsr.dilated_reactive", render, TextureFormat::RGBA8Unorm);
  state.lumaInstability = create_texture(device, "fsr.luma_instability", render, TextureFormat::R16Float);
  for (size_t i = 0; i < 2; ++i) {
    state.luma[i] = create_texture(device, "fsr.luma", render, TextureFormat::R16Float);
    state.accumulation[i] = create_texture(device, "fsr.accumulation", render, TextureFormat::R8Unorm);
    state.frameInfo[i] = create_texture(device, "fsr.frame_info", {1, 1}, TextureFormat::RGBA32Float);
    state.lumaHistory[i] = create_texture(device, "fsr.luma_history", render, TextureFormat::RGBA16Float);
    state.upscaled[i] = create_texture(device, "fsr.upscaled", output, TextureFormat::RGBA16Float);
  }
  state.locks = create_texture(device, "fsr.locks", output, locksR8 ? TextureFormat::R8Unorm : TextureFormat::R32Float);
  const wgpu::BufferDescriptor recon{
      .label = "fsr.recon",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
      .size = uint64_t{render.width} * render.height * sizeof(uint32_t),
  };
  state.recon = device.CreateBuffer(&recon);
}
} // namespace

float mode_ratio(UpscalerQuality quality) noexcept {
  switch (quality) {
  case UpscalerQuality::NativeAa:
    return 1.f;
  case UpscalerQuality::Quality:
    return 1.5f;
  case UpscalerQuality::Balanced:
    return 1.7f;
  case UpscalerQuality::Performance:
    return 2.f;
  case UpscalerQuality::UltraPerformance:
    return 3.f;
  }
  return 1.f;
}

GraphExtent render_size(GraphExtent output, UpscalerQuality quality) noexcept {
  const float ratio = mode_ratio(quality);
  const auto axis = [ratio](uint32_t size) {
    return std::max(MinRenderSize, static_cast<uint32_t>(static_cast<float>(size) / ratio));
  };
  return {axis(output.width), axis(output.height)};
}

uint32_t jitter_phases(float ratio) noexcept {
  return std::max(BasePhases, static_cast<uint32_t>(static_cast<float>(BasePhases) * ratio * ratio));
}

uint32_t spd_mips(GraphExtent render) noexcept {
  const uint32_t largest = std::max(render.width, render.height);
  return largest == 0 ? 0 : std::min(MaxSpdMips, static_cast<uint32_t>(std::bit_width(largest)) - 1);
}

GraphExtent spd_tiles(GraphExtent render) noexcept {
  return {groups(render.width, TileSize), groups(render.height, TileSize)};
}

std::optional<Camera> camera_from_projection(const Mat4x4<float>& gxProj, float unitsPerMeter) noexcept {
  const double p22 = gxProj.m2[2];
  const double p23 = gxProj.m2[3];
  const double p11 = gxProj.m1[1];
  const double nearZ = p23 / (p22 - 1.0);
  const double farZ = p22 != 0.0 ? p23 / p22 : std::numeric_limits<double>::infinity();
  if (!(nearZ > 0.0) || !(farZ > nearZ) || !(p11 > 0.0)) {
    return std::nullopt;
  }
  return Camera{
      .nearZ = static_cast<float>(nearZ),
      .farZ = static_cast<float>(farZ),
      .fovY = static_cast<float>(2.0 * std::atan(1.0 / p11)),
      .viewToMeters = 1.f / (unitsPerMeter > 0.f ? unitsPerMeter : DefaultUnitsPerMeter),
  };
}

float frame_delta_ms(std::optional<std::chrono::steady_clock::time_point>& last,
                     std::chrono::steady_clock::time_point now, bool blocking) noexcept {
  const std::optional<std::chrono::steady_clock::time_point> previous = std::exchange(last, now);
  if (blocking || !previous) {
    return NominalFrameMs;
  }
  return std::chrono::duration<float, std::milli>(now - *previous).count();
}

Params make_params(const FrameInputs& inputs) noexcept {
  Params params;
  std::copy_n(inputs.jitter, 2, params.jitter);
  std::copy_n(inputs.prevJitter, 2, params.prevJitter);
  std::copy_n(inputs.motionScale, 2, params.motionScale);
  const float farthest = std::max(inputs.camera.nearZ, inputs.camera.farZ);
  const float nearest = std::min(inputs.camera.nearZ, inputs.camera.farZ);
  if (std::isfinite(farthest)) {
    const float q = nearest / (farthest - nearest);
    params.deviceToView[0] = -q;
    params.deviceToView[1] = q * farthest;
  } else {
    params.deviceToView[0] = -FLT_EPSILON;
    params.deviceToView[1] = nearest;
  }
  params.jitterSequenceLength = static_cast<float>(std::max(1u, inputs.jitterPhases));
  params.deltaTime = std::clamp(inputs.frameTimeMs / 1000.f, 0.f, 1.f);
  params.viewToMeters = inputs.camera.viewToMeters > 0.f ? inputs.camera.viewToMeters : 1.f;
  params.frameIndex = static_cast<float>(inputs.framesSinceReset);
  params.spdMips = spd_mips(inputs.render);
  return params;
}

std::string_view shader_id(Stage stage) noexcept { return ShaderIds[index_of(stage)]; }

Refs declare(PassBuilder& builder, const Selection& selection) {
  const auto transient = [&builder](std::string_view name, TextureFormat format, uint16_t divisor = 1,
                                    uint32_t mips = 1) {
    return builder.create(name,
                          TexDesc{
                              .format = format,
                              .sizeClass = SizeClass::Efb,
                              .divisor = divisor,
                              .mips = mips,
                              .extraUsage = wgpu::TextureUsage::TextureBinding,
                          },
                          Access::StorageWrite);
  };
  const auto history = [&builder](std::string_view name, TextureFormat format, SizeClass sizeClass,
                                  wgpu::TextureUsage extraUsage, Access access) {
    return builder.history(name, TexDesc{.format = format, .sizeClass = sizeClass, .extraUsage = extraUsage}, access);
  };
  constexpr wgpu::TextureUsage Sampled = wgpu::TextureUsage::TextureBinding;
  Refs refs;
  refs.dilatedDepth = transient("fsr.dilated_depth", TextureFormat::R32Float);
  refs.dilatedMotion = transient("fsr.dilated_motion", TextureFormat::RG16Float);
  refs.farthestDepth = transient("fsr.farthest_depth", TextureFormat::R16Float);
  refs.farthestDepthMip1 = transient("fsr.farthest_depth_mip1", TextureFormat::R16Float, 2);
  refs.lumaMip5 = transient("fsr.luma_mip5", TextureFormat::RG16Float, TileSize);
  refs.spd = transient("fsr.spd", TextureFormat::RG16Float, 2, 3);
  refs.shadingChange = transient("fsr.shading_change", TextureFormat::R8Unorm, 2);
  refs.dilatedReactive = transient("fsr.dilated_reactive", TextureFormat::RGBA8Unorm);
  refs.lumaInstability = transient("fsr.luma_instability", TextureFormat::R16Float);
  refs.luma = history("fsr.luma", TextureFormat::R16Float, SizeClass::Efb, Sampled, Access::StorageWrite);
  refs.accumulation =
      history("fsr.accumulation", TextureFormat::R8Unorm, SizeClass::Efb, Sampled, Access::StorageWrite);
  refs.frameInfo = builder.history("fsr.frame_info",
                                   TexDesc{
                                       .format = TextureFormat::RGBA32Float,
                                       .sizeClass = SizeClass::Absolute,
                                       .width = 1,
                                       .height = 1,
                                       .extraUsage = Sampled,
                                   },
                                   Access::StorageWrite);
  refs.lumaHistory =
      history("fsr.luma_history", TextureFormat::RGBA16Float, SizeClass::Efb, Sampled, Access::StorageWrite);
  refs.upscaled =
      history("fsr.upscaled", TextureFormat::RGBA16Float, SizeClass::OutputEfb, Sampled, Access::StorageWrite);
  refs.locks = history("fsr.locks", selection.locksR8 ? TextureFormat::R8Unorm : TextureFormat::R32Float,
                       SizeClass::OutputEfb, wgpu::TextureUsage::None, Access::StorageRW);
  refs.recon = builder.create_buffer("fsr.recon",
                                     BufDesc{
                                         .sizeClass = SizeClass::Efb,
                                         .bytesPerTexel = sizeof(uint32_t),
                                         .extraUsage = wgpu::BufferUsage::CopyDst,
                                     },
                                     Access::StorageRW);
  return refs;
}

bool request_pipelines(const wgpu::BindGroupLayout& frameLayout, uint64_t layoutKey, bool locksR8, Pipelines& out) {
  if (!webgpu::g_device || !frameLayout) {
    return false;
  }
  std::array<wgpu::PipelineLayout, StageCount> layouts;
  {
    std::lock_guard lock{g_mutex};
    layouts = pipeline_layouts_locked(frameLayout, locksR8).layouts;
  }
  out.locksR8 = locksR8;
  for (size_t i = 0; i < StageCount; ++i) {
    const auto stage = static_cast<Stage>(i);
    const bool r8 = locksR8 && uses_locks(stage);
    out.ids[i] = shaders::request_compute({
        .label = "aurora.fsr",
        .shader = shader_id(stage),
        .entryPoint = "main",
        .defines = r8 ? std::span<const std::string_view>{LocksR8Defines} : std::span<const std::string_view>{},
        .layout = layouts[i],
        .layoutKey = layoutKey ^ (LayoutSalt + (i << 1) + (r8 ? 1 : 0)),
    });
  }
  return std::ranges::all_of(out.ids, [](shaders::PipelineId id) { return static_cast<bool>(id); });
}

bool pipelines_ready(const Pipelines& pipelines) noexcept {
  return std::ranges::all_of(pipelines.ids, [](shaders::PipelineId id) { return shaders::ready(id); });
}

bool encode(const wgpu::CommandEncoder& cmd, const Pipelines& pipelines, const FrameGroup& frame,
            const Bindings& bindings, GraphExtent render, GraphExtent output, Stage last) {
  const wgpu::Device& device = webgpu::g_device;
  if (!device || !cmd || !frame.group || !bindings.recon.buffer) {
    return false;
  }
  const size_t count = index_of(last) + 1;
  std::array<wgpu::ComputePipeline, StageCount> compiled;
  std::array<wgpu::BindGroupLayout, StageCount> layouts;
  for (size_t i = 0; i < count; ++i) {
    compiled[i] = shaders::compute_pipeline(pipelines.ids[i]);
    if (!compiled[i]) {
      return false;
    }
  }
  {
    std::lock_guard lock{g_mutex};
    for (size_t i = 0; i < count; ++i) {
      layouts[i] = group_layout_locked(static_cast<Stage>(i), pipelines.locksR8);
    }
  }
  cmd.ClearBuffer(bindings.recon.buffer, bindings.recon.offset, bindings.recon.size);
  const wgpu::ComputePassDescriptor descriptor{.label = "aurora.fsr"};
  const wgpu::ComputePassEncoder pass = cmd.BeginComputePass(&descriptor);
  for (size_t i = 0; i < count; ++i) {
    const auto stage = static_cast<Stage>(i);
    const std::array<uint32_t, 2> size = dispatch_size(stage, render, output);
    if (size[0] == 0 || size[1] == 0) {
      continue;
    }
    const std::vector<wgpu::BindGroupEntry> entries = group_entries(stage, bindings);
    const wgpu::BindGroupDescriptor groupDescriptor{
        .label = "aurora.fsr",
        .layout = layouts[i],
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const wgpu::BindGroup group = device.CreateBindGroup(&groupDescriptor);
    pass.SetPipeline(compiled[i]);
    pass.SetBindGroup(FrameGroupIndex, frame.group, frame.offsetCount, frame.offsets.data());
    pass.SetBindGroup(PassGroupIndex, group);
    pass.DispatchWorkgroups(size[0], size[1]);
  }
  pass.End();
  return true;
}

void shutdown() noexcept {
  std::lock_guard lock{g_mutex};
  g_shared = {};
  g_test = {};
}

namespace testing {
std::span<const Binding> layout(Stage stage, bool locksR8) noexcept {
  switch (stage) {
  case Stage::PrepareInputs:
    return PrepareInputsLayout;
  case Stage::LumaPyramidA:
    return LumaPyramidALayout;
  case Stage::LumaPyramidB:
    return LumaPyramidBLayout;
  case Stage::ShadingChangePyramid:
    return ShadingChangePyramidLayout;
  case Stage::ShadingChange:
    return ShadingChangeLayout;
  case Stage::PrepareReactivity:
    return locksR8 ? std::span<const Binding>{PrepareReactivityR8Layout} : PrepareReactivityLayout;
  case Stage::LumaInstability:
    return LumaInstabilityLayout;
  case Stage::Accumulate:
    return locksR8 ? std::span<const Binding>{AccumulateR8Layout} : AccumulateLayout;
  }
  return {};
}

bool run(State& state, const Inputs& inputs, Stage last) {
  const wgpu::Device& device = webgpu::g_device;
  if (!device || !inputs.color || !inputs.depth || !inputs.motion || !inputs.reactive) {
    return false;
  }
  const GraphExtent render{inputs.color.GetWidth(), inputs.color.GetHeight()};
  const GraphExtent output = state.output.width != 0 ? state.output : render;
  const bool locksR8 = state.locksR8;
  if (!state.recon || state.render != render || state.output != output) {
    create_state_resources(device, state, render, output, locksR8);
  }
  {
    std::lock_guard lock{g_mutex};
    ensure_test_shared_locked(device);
  }
  const size_t variant = locksR8 ? 1 : 0;
  if (!g_test.requested[variant]) {
    if (!request_pipelines(g_test.frameLayout, 0x7465'7374ull, locksR8, g_test.pipelines[variant])) {
      return false;
    }
    g_test.requested[variant] = true;
  }
  const Pipelines pipelines = g_test.pipelines[variant];
  for (const shaders::PipelineId id : pipelines.ids) {
    if (shaders::wait(id) != shaders::PipelineStatus::Ready) {
      return false;
    }
  }
  FrameConstants constants{};
  constants.efbSize[0] = render.width;
  constants.efbSize[1] = render.height;
  constants.outputSize[0] = output.width;
  constants.outputSize[1] = output.height;
  constants.invEfbSize[0] = 1.f / static_cast<float>(render.width);
  constants.invEfbSize[1] = 1.f / static_cast<float>(render.height);
  constants.invOutputSize[0] = 1.f / static_cast<float>(output.width);
  constants.invOutputSize[1] = 1.f / static_cast<float>(output.height);
  constants.frameIndex = state.frames;
  constants.flags = FrameFlagReversedZ;
  constants.msaaSamples = 1;
  FrameInputs frameInputs = inputs.frame;
  frameInputs.render = render;
  const Params params = make_params(frameInputs);
  webgpu::g_queue.WriteBuffer(g_test.constants, 0, &constants, sizeof(constants));
  webgpu::g_queue.WriteBuffer(g_test.params, 0, &params, sizeof(params));
  const std::array frameEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_test.constants, .size = FrameUniformsSize},
      wgpu::BindGroupEntry{.binding = 1, .buffer = g_test.params, .size = MaxPassParamsSize},
      wgpu::BindGroupEntry{.binding = 2, .sampler = g_test.samplers[0]},
      wgpu::BindGroupEntry{.binding = 3, .sampler = g_test.samplers[1]},
      wgpu::BindGroupEntry{.binding = 4, .sampler = g_test.samplers[2]},
      wgpu::BindGroupEntry{.binding = 5, .sampler = g_test.samplers[3]},
  };
  const wgpu::BindGroupDescriptor frameDescriptor{
      .label = "aurora.fsr test frame",
      .layout = g_test.frameLayout,
      .entryCount = frameEntries.size(),
      .entries = frameEntries.data(),
  };
  const wgpu::BindGroup frameGroup = device.CreateBindGroup(&frameDescriptor);
  const size_t cur = state.frames % 2;
  const size_t prev = 1 - cur;
  const wgpu::TextureViewDescriptor depthView{.aspect = wgpu::TextureAspect::DepthOnly};
  Bindings b{
      .color = inputs.color.CreateView(),
      .depth = inputs.depth.CreateView(&depthView),
      .motion = inputs.motion.CreateView(),
      .reactive = inputs.reactive.CreateView(),
      .dilatedDepth = state.dilatedDepth.CreateView(),
      .dilatedMotion = state.dilatedMotion.CreateView(),
      .farthestDepth = state.farthestDepth.CreateView(),
      .farthestDepthMip1 = state.farthestDepthMip1.CreateView(),
      .lumaMip5 = state.lumaMip5.CreateView(),
      .spd = state.spd.CreateView(),
      .spdMips = {mip_view(state.spd, 0), mip_view(state.spd, 1), mip_view(state.spd, 2)},
      .shadingChange = state.shadingChange.CreateView(),
      .dilatedReactive = state.dilatedReactive.CreateView(),
      .lumaInstability = state.lumaInstability.CreateView(),
      .luma = state.luma[cur].CreateView(),
      .prevLuma = state.luma[prev].CreateView(),
      .accumulation = state.accumulation[cur].CreateView(),
      .prevAccumulation = state.accumulation[prev].CreateView(),
      .frameInfo = state.frameInfo[cur].CreateView(),
      .prevFrameInfo = state.frameInfo[prev].CreateView(),
      .lumaHistory = state.lumaHistory[cur].CreateView(),
      .prevLumaHistory = state.lumaHistory[prev].CreateView(),
      .upscaled = state.upscaled[cur].CreateView(),
      .prevUpscaled = state.upscaled[prev].CreateView(),
      .locks = state.locks.CreateView(),
      .recon = {.buffer = state.recon, .offset = 0, .size = state.recon.GetSize()},
  };
  const wgpu::CommandEncoder cmd = device.CreateCommandEncoder();
  const FrameGroup frame{.group = frameGroup, .offsets = {0, 0}, .offsetCount = 2};
  if (!encode(cmd, pipelines, frame, b, render, output, last)) {
    return false;
  }
  const wgpu::CommandBuffer commands = cmd.Finish();
  webgpu::g_queue.Submit(1, &commands);
  ++state.frames;
  return true;
}
} // namespace testing
} // namespace aurora::render::detail::upscale::fsr
