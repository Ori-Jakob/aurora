#include "indirect_bench.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace render_demo {
namespace {
using Clock = std::chrono::steady_clock;

constexpr uint32_t VerticesPerQuad = 4;
constexpr uint32_t IndicesPerQuad = 6;
constexpr uint32_t QuadPixels = 4;
constexpr uint32_t QuadSpacing = 16;
constexpr uint32_t QuadColumns = 50;
constexpr uint32_t ArgsWords = 5;
constexpr uint64_t ArgsBytes = BenchMaxDraws * ArgsWords * sizeof(uint32_t);
constexpr uint32_t ImmediateBytes = 64;
constexpr uint32_t ArgsImmediateBytes = 16;
constexpr uint32_t UniformBytes = 16;
constexpr uint32_t ArgsWorkgroup = 64;
constexpr uint32_t QueryCount = 2 * BenchMaxPasses + 2;
constexpr uint64_t QueryBytes = QueryCount * sizeof(uint64_t);
constexpr uint32_t MaxWarmupBlocks = 10;
constexpr auto WaitTimeout = std::chrono::seconds{5};
constexpr size_t ErrorChars = 120;

constexpr std::string_view RenderWgsl = R"(
struct Immediates {
  offset: vec4f,
  tint: vec4f,
  pad0: vec4f,
  pad1: vec4f,
};
var<immediate> imm: Immediates;
struct DrawUniform {
  tint: vec4f,
};
@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(1) @binding(0) var<uniform> draw: DrawUniform;
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) tint: vec4f,
};

fn make_vertex(p: vec2f) -> VertexOut {
  var out: VertexOut;
  out.position = vec4f(p + imm.offset.xy, 0.5, 1.0);
  out.uv = p * 0.5 + vec2f(0.5);
  out.tint = draw.tint + imm.tint;
  return out;
}

@vertex fn vs_vi(@builtin(vertex_index) vi: u32) -> VertexOut {
  return make_vertex(positions[vi]);
}

@vertex fn vs_plain(@location(0) p: vec2f) -> VertexOut {
  return make_vertex(p);
}

@vertex fn vs_vi_subst(@builtin(vertex_index) vi: u32) -> VertexOut {
  var out = make_vertex(positions[vi]);
  out.tint = out.tint * textureSampleLevel(tex, samp, vec2f(0.5), 0.0);
  return out;
}

@fragment fn fs_main(input: VertexOut) -> @location(0) vec4f {
  return textureSample(tex, samp, input.uv) * input.tint;
}
)";

constexpr std::string_view ComputeWgsl = R"(
override ZERO_ODD: bool = false;
struct ArgsImmediates {
  count: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
var<immediate> imm: ArgsImmediates;
@group(0) @binding(0) var<storage, read_write> args: array<u32>;

@compute @workgroup_size(64)
fn cs_args(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= imm.count) {
    return;
  }
  let base = i * 5u;
  args[base] = select(6u, 0u, ZERO_ODD && (i & 1u) == 1u);
  args[base + 1u] = 1u;
  args[base + 2u] = 6u * i;
  args[base + 3u] = 0u;
  args[base + 4u] = 0u;
}

@compute @workgroup_size(1)
fn cs_noop() {
  args[arrayLength(&args) - 1u] = 1u;
}
)";

enum class RenderKind : uint8_t { VertexIndex, Plain, Subst };
enum class ComputeKind : uint8_t { Args, ArgsZero, Noop };

struct Iteration {
  double encodeUs = 0, finishUs = 0, submitUs = 0, totalUs = 0;
  double gpuTotalUs = std::numeric_limits<double>::quiet_NaN();
  double gpuInPassUs = std::numeric_limits<double>::quiet_NaN();
  double gpuArgsUs = std::numeric_limits<double>::quiet_NaN();
  bool gpuValid = false;
};

double micros(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

RenderKind render_kind(const BenchArm& arm) noexcept {
  if (arm.state == BenchState::Subst) {
    return RenderKind::Subst;
  }
  return arm.pipeline == BenchPipeline::VertexIndex ? RenderKind::VertexIndex : RenderKind::Plain;
}

ComputeKind compute_kind(const BenchArm& arm) noexcept {
  if (arm.args == BenchArgs::Cpu) {
    return ComputeKind::Noop;
  }
  return arm.mode == BenchMode::Zero ? ComputeKind::ArgsZero : ComputeKind::Args;
}

class Bench {
public:
  Bench(const BenchDevice& device, const BenchOptions& options, BenchRun& run)
  : m_dev(device), m_options(options), m_run(run) {}

  bool setup(std::string& error);
  bool run_pairs(BenchPump pump, void* userdata, std::string& error);

private:
  wgpu::Buffer buffer(const char* label, wgpu::BufferUsage usage, uint64_t size, const void* data = nullptr) const;
  wgpu::BindGroupLayout layout(std::initializer_list<wgpu::BindGroupLayoutEntry> entries) const;
  wgpu::BindGroup group(const wgpu::BindGroupLayout& layout, std::initializer_list<wgpu::BindGroupEntry> entries) const;
  wgpu::Texture texture(const std::array<uint8_t, 4>& color) const;
  const wgpu::RenderPipeline& render_pipeline(RenderKind kind);
  const wgpu::ComputePipeline& compute_pipeline(ComputeKind kind);
  void prepare(const BenchArm& arm, const BenchPair& pair, std::vector<wgpu::RenderBundle>& bundles);
  template <typename Encoder>
  void record_draws(const Encoder& encoder, const BenchArm& arm, uint32_t first, uint32_t count) const;
  bool iterate(const BenchArm& arm, const BenchPair& pair, const std::vector<wgpu::RenderBundle>& bundles,
               Iteration& out, std::string& error);
  bool wait(std::span<wgpu::FutureWaitInfo> futures, std::string& error) const;
  std::optional<std::string> pop_scopes() const;
  bool probe_timestamps();

  const BenchDevice& m_dev;
  const BenchOptions& m_options;
  BenchRun& m_run;
  bool m_timestamps = false;
  wgpu::TextureView m_color;
  wgpu::TextureView m_depth;
  wgpu::Buffer m_vertices;
  wgpu::Buffer m_indices16;
  wgpu::Buffer m_indices32;
  wgpu::Buffer m_uniforms;
  wgpu::Buffer m_args;
  wgpu::Buffer m_cpuArgs;
  wgpu::QuerySet m_queries;
  wgpu::Buffer m_resolve;
  wgpu::Buffer m_readback;
  std::array<wgpu::Texture, 2> m_textures;
  wgpu::Sampler m_sampler;
  wgpu::BindGroupLayout m_positionsLayout;
  wgpu::BindGroupLayout m_uniformLayout;
  wgpu::BindGroupLayout m_textureLayout;
  wgpu::BindGroupLayout m_substTextureLayout;
  wgpu::BindGroupLayout m_argsLayout;
  wgpu::PipelineLayout m_renderLayout;
  wgpu::PipelineLayout m_substLayout;
  wgpu::PipelineLayout m_computeLayout;
  std::array<wgpu::BindGroup, 2> m_positionGroups;
  wgpu::BindGroup m_uniformGroup;
  std::array<wgpu::BindGroup, 2> m_textureGroups;
  std::array<wgpu::BindGroup, 2> m_substTextureGroups;
  wgpu::BindGroup m_argsGroup;
  wgpu::ShaderModule m_renderModule;
  wgpu::ShaderModule m_computeModule;
  std::array<wgpu::RenderPipeline, 3> m_renderPipelines;
  std::array<wgpu::ComputePipeline, 3> m_computePipelines;
  std::vector<std::array<uint32_t, ImmediateBytes / sizeof(uint32_t)>> m_immediates;
};

wgpu::Buffer Bench::buffer(const char* label, wgpu::BufferUsage usage, uint64_t size, const void* data) const {
  const wgpu::BufferDescriptor descriptor{
      .label = label, .usage = usage, .size = size, .mappedAtCreation = data != nullptr};
  wgpu::Buffer out = m_dev.device.CreateBuffer(&descriptor);
  if (data != nullptr) {
    if (void* mapped = out.GetMappedRange(0, size)) {
      std::memcpy(mapped, data, size);
    }
    out.Unmap();
  }
  return out;
}

wgpu::BindGroupLayout Bench::layout(std::initializer_list<wgpu::BindGroupLayoutEntry> entries) const {
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "render_demo bench",
      .entryCount = entries.size(),
      .entries = entries.begin(),
  };
  return m_dev.device.CreateBindGroupLayout(&descriptor);
}

wgpu::BindGroup Bench::group(const wgpu::BindGroupLayout& groupLayout,
                             std::initializer_list<wgpu::BindGroupEntry> entries) const {
  const wgpu::BindGroupDescriptor descriptor{
      .label = "render_demo bench",
      .layout = groupLayout,
      .entryCount = entries.size(),
      .entries = entries.begin(),
  };
  return m_dev.device.CreateBindGroup(&descriptor);
}

wgpu::Texture Bench::texture(const std::array<uint8_t, 4>& color) const {
  const wgpu::TextureDescriptor descriptor{
      .label = "render_demo bench texture",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {1, 1, 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
  };
  wgpu::Texture out = m_dev.device.CreateTexture(&descriptor);
  const wgpu::TexelCopyTextureInfo destination{.texture = out};
  const wgpu::TexelCopyBufferLayout dataLayout{.bytesPerRow = 4, .rowsPerImage = 1};
  const wgpu::Extent3D size{1, 1, 1};
  m_dev.queue.WriteTexture(&destination, color.data(), color.size(), &dataLayout, &size);
  return out;
}

bool Bench::probe_timestamps() {
  m_dev.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  const wgpu::QuerySetDescriptor queryDescriptor{
      .label = "render_demo bench timestamps",
      .type = wgpu::QueryType::Timestamp,
      .count = QueryCount,
  };
  m_queries = m_dev.device.CreateQuerySet(&queryDescriptor);
  m_resolve =
      buffer("render_demo bench resolve", wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, QueryBytes);
  m_readback =
      buffer("render_demo bench readback", wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, QueryBytes);
  const wgpu::CommandEncoder encoder = m_dev.device.CreateCommandEncoder();
  const wgpu::PassTimestampWrites writes{
      .querySet = m_queries, .beginningOfPassWriteIndex = 0, .endOfPassWriteIndex = 1};
  const wgpu::ComputePassDescriptor passDescriptor{.timestampWrites = &writes};
  encoder.BeginComputePass(&passDescriptor).End();
  encoder.ResolveQuerySet(m_queries, 0, 2, m_resolve, 0);
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_dev.queue.Submit(1, &commands);
  std::optional<std::string> failure;
  const wgpu::Future future = m_dev.device.PopErrorScope(
      wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView) {
        if (status != wgpu::PopErrorScopeStatus::Success || type != wgpu::ErrorType::NoError) {
          failure = "error";
        }
      });
  std::array<wgpu::FutureWaitInfo, 1> futures{{{.future = future}}};
  std::string error;
  return wait(futures, error) && !failure;
}

bool Bench::setup(std::string& error) {
  m_run.msaa = m_dev.sampleCount;
  m_run.width = m_options.width;
  m_run.height = m_options.height;
  m_run.blocks = m_options.blocks;
  m_run.warmupBlocks = std::min(MaxWarmupBlocks, m_options.blocks);
  m_run.uniformPermille = m_options.uniformPermille;
  m_run.texturePermille = m_options.texturePermille;
  m_run.mdi = m_dev.multiDrawIndirect;
#ifdef NDEBUG
  m_run.build = "release";
  m_run.validation = m_run.profile == "full" ? "skipped" : "on";
#else
  m_run.build = "debug";
  m_run.validation = "on";
#endif
  m_timestamps = m_dev.timestampQuery && probe_timestamps();
  m_run.timestamps = m_timestamps;

  m_dev.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  m_dev.device.PushErrorScope(wgpu::ErrorFilter::Internal);
  const wgpu::TextureFormat colorFormat =
      m_dev.colorFormat != wgpu::TextureFormat::Undefined ? m_dev.colorFormat : wgpu::TextureFormat::BGRA8Unorm;
  const wgpu::TextureDescriptor colorDescriptor{
      .label = "render_demo bench color",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .size = {m_options.width, m_options.height, 1},
      .format = colorFormat,
      .sampleCount = m_dev.sampleCount,
  };
  m_color = m_dev.device.CreateTexture(&colorDescriptor).CreateView();
  wgpu::TextureDescriptor depthDescriptor = colorDescriptor;
  depthDescriptor.label = "render_demo bench depth";
  depthDescriptor.format = wgpu::TextureFormat::Depth32Float;
  m_depth = m_dev.device.CreateTexture(&depthDescriptor).CreateView();

  std::vector<float> positions;
  positions.reserve(BenchMaxDraws * VerticesPerQuad * 2);
  std::vector<uint16_t> indices16;
  std::vector<uint32_t> indices32;
  const auto ndc = [](uint32_t pixel, uint32_t size) {
    return 2.f * static_cast<float>(pixel) / static_cast<float>(size) - 1.f;
  };
  for (uint32_t quad = 0; quad < BenchMaxDraws; ++quad) {
    const uint32_t x = QuadSpacing * (quad % QuadColumns) + QuadSpacing / 2;
    const uint32_t y = QuadSpacing * (quad / QuadColumns) + QuadSpacing / 2;
    for (const auto& [dx, dy] :
         {std::pair{0u, 0u}, std::pair{QuadPixels, 0u}, std::pair{0u, QuadPixels}, std::pair{QuadPixels, QuadPixels}}) {
      positions.push_back(ndc(x + dx, m_options.width));
      positions.push_back(-ndc(y + dy, m_options.height));
    }
    for (const uint32_t corner : {0u, 1u, 2u, 2u, 1u, 3u}) {
      indices16.push_back(static_cast<uint16_t>(quad * VerticesPerQuad + corner));
      indices32.push_back(quad * VerticesPerQuad + corner);
    }
  }
  m_vertices = buffer("render_demo bench vertices",
                      wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      positions.size() * sizeof(float), positions.data());
  m_indices16 = buffer("render_demo bench u16 indices", wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst,
                       indices16.size() * sizeof(uint16_t), indices16.data());
  m_indices32 = buffer("render_demo bench u32 indices", wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst,
                       indices32.size() * sizeof(uint32_t), indices32.data());
  const uint32_t alignment = std::max<uint32_t>(m_dev.minUniformBufferOffsetAlignment, UniformBytes);
  std::vector<float> uniforms(static_cast<size_t>(BenchMaxDraws) * alignment / sizeof(float), 0.f);
  m_immediates.resize(BenchMaxDraws);
  for (uint32_t draw = 0; draw < BenchMaxDraws; ++draw) {
    float* tint = uniforms.data() + static_cast<size_t>(draw) * alignment / sizeof(float);
    tint[0] = static_cast<float>(draw % 7) / 7.f;
    tint[1] = static_cast<float>(draw % 11) / 11.f;
    tint[2] = static_cast<float>(draw % 13) / 13.f;
    tint[3] = 1.f;
    m_immediates[draw] = {};
    const float shade = static_cast<float>(draw % 5) * 0.01f;
    std::memcpy(&m_immediates[draw][4], &shade, sizeof(shade));
  }
  m_uniforms = buffer("render_demo bench uniforms", wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      uniforms.size() * sizeof(float), uniforms.data());
  std::vector<uint32_t> cpuArgs;
  cpuArgs.reserve(BenchMaxDraws * ArgsWords);
  for (uint32_t draw = 0; draw < BenchMaxDraws; ++draw) {
    cpuArgs.insert(cpuArgs.end(), {IndicesPerQuad, 1u, IndicesPerQuad * draw, 0u, 0u});
  }
  m_args =
      buffer("render_demo bench args",
             wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, ArgsBytes + 16);
  m_cpuArgs = buffer("render_demo bench cpu args", wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst, ArgsBytes,
                     cpuArgs.data());
  m_textures = {texture({255, 255, 255, 255}), texture({200, 220, 255, 255})};
  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "render_demo bench",
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  m_sampler = m_dev.device.CreateSampler(&samplerDescriptor);

  m_positionsLayout = layout({{.binding = 0,
                               .visibility = wgpu::ShaderStage::Vertex,
                               .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}}});
  m_uniformLayout = layout(
      {{.binding = 0,
        .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
        .buffer = {
            .type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true, .minBindingSize = UniformBytes}}});
  const auto textureEntries = [](wgpu::ShaderStage visibility) {
    return std::array<wgpu::BindGroupLayoutEntry, 2>{{
        {.binding = 0, .visibility = visibility, .texture = {.sampleType = wgpu::TextureSampleType::Float}},
        {.binding = 1, .visibility = visibility, .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
    }};
  };
  const auto fragmentTexture = textureEntries(wgpu::ShaderStage::Fragment);
  m_textureLayout = layout({fragmentTexture[0], fragmentTexture[1]});
  const auto bothTexture = textureEntries(wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment);
  m_substTextureLayout = layout({bothTexture[0], bothTexture[1]});
  m_argsLayout = layout(
      {{.binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage}}});
  const auto pipelineLayout = [&](std::initializer_list<wgpu::BindGroupLayout> groups, uint32_t immediateSize) {
    const wgpu::PipelineLayoutDescriptor descriptor{
        .label = "render_demo bench",
        .bindGroupLayoutCount = groups.size(),
        .bindGroupLayouts = groups.begin(),
        .immediateSize = immediateSize,
    };
    return m_dev.device.CreatePipelineLayout(&descriptor);
  };
  m_renderLayout = pipelineLayout({m_positionsLayout, m_uniformLayout, m_textureLayout}, ImmediateBytes);
  m_substLayout = pipelineLayout({m_positionsLayout, m_uniformLayout, m_substTextureLayout}, ImmediateBytes);
  m_computeLayout = pipelineLayout({m_argsLayout}, ArgsImmediateBytes);
  for (auto& positionGroup : m_positionGroups) {
    positionGroup = group(m_positionsLayout, {{.binding = 0, .buffer = m_vertices}});
  }
  m_uniformGroup = group(m_uniformLayout, {{.binding = 0, .buffer = m_uniforms, .size = UniformBytes}});
  for (size_t i = 0; i < m_textures.size(); ++i) {
    const wgpu::TextureView view = m_textures[i].CreateView();
    m_textureGroups[i] =
        group(m_textureLayout, {{.binding = 0, .textureView = view}, {.binding = 1, .sampler = m_sampler}});
    m_substTextureGroups[i] =
        group(m_substTextureLayout, {{.binding = 0, .textureView = view}, {.binding = 1, .sampler = m_sampler}});
  }
  m_argsGroup = group(m_argsLayout, {{.binding = 0, .buffer = m_args}});
  const auto module = [&](std::string_view code, const char* label) {
    const std::string text{code};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = text.c_str()}};
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl, .label = label};
    return m_dev.device.CreateShaderModule(&descriptor);
  };
  m_renderModule = module(RenderWgsl, "render_demo bench render");
  m_computeModule = module(ComputeWgsl, "render_demo bench compute");
  if (const auto failure = pop_scopes()) {
    error = "setup failed: " + *failure;
    return false;
  }
  return true;
}

const wgpu::RenderPipeline& Bench::render_pipeline(RenderKind kind) {
  auto& pipeline = m_renderPipelines[static_cast<size_t>(kind)];
  if (pipeline) {
    return pipeline;
  }
  constexpr wgpu::VertexAttribute Attribute{.format = wgpu::VertexFormat::Float32x2, .offset = 0, .shaderLocation = 0};
  const wgpu::VertexBufferLayout plainBuffer{
      .stepMode = wgpu::VertexStepMode::Vertex,
      .arrayStride = 2 * sizeof(float),
      .attributeCount = 1,
      .attributes = &Attribute,
  };
  const wgpu::TextureFormat colorFormat =
      m_dev.colorFormat != wgpu::TextureFormat::Undefined ? m_dev.colorFormat : wgpu::TextureFormat::BGRA8Unorm;
  const wgpu::ColorTargetState target{.format = colorFormat};
  const wgpu::FragmentState fragment{
      .module = m_renderModule,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::DepthStencilState depth{
      .format = wgpu::TextureFormat::Depth32Float,
      .depthWriteEnabled = wgpu::OptionalBool::True,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  const bool plain = kind == RenderKind::Plain;
  const char* entry = kind == RenderKind::Subst ? "vs_vi_subst" : plain ? "vs_plain" : "vs_vi";
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = entry,
      .layout = kind == RenderKind::Subst ? m_substLayout : m_renderLayout,
      .vertex = {.module = m_renderModule,
                 .entryPoint = entry,
                 .bufferCount = plain ? 1u : 0u,
                 .buffers = plain ? &plainBuffer : nullptr},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depth,
      .multisample = {.count = m_dev.sampleCount},
      .fragment = &fragment,
  };
  pipeline = m_dev.device.CreateRenderPipeline(&descriptor);
  return pipeline;
}

const wgpu::ComputePipeline& Bench::compute_pipeline(ComputeKind kind) {
  auto& pipeline = m_computePipelines[static_cast<size_t>(kind)];
  if (pipeline) {
    return pipeline;
  }
  const wgpu::ConstantEntry zeroOdd{.key = "ZERO_ODD", .value = 1.0};
  const bool noop = kind == ComputeKind::Noop;
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = noop ? "cs_noop" : "cs_args",
      .layout = m_computeLayout,
      .compute = {.module = m_computeModule,
                  .entryPoint = noop ? "cs_noop" : "cs_args",
                  .constantCount = kind == ComputeKind::ArgsZero ? 1u : 0u,
                  .constants = kind == ComputeKind::ArgsZero ? &zeroOdd : nullptr},
  };
  pipeline = m_dev.device.CreateComputePipeline(&descriptor);
  return pipeline;
}

template <typename Encoder>
void Bench::record_draws(const Encoder& encoder, const BenchArm& arm, uint32_t first, uint32_t count) const {
  const bool subst = arm.state == BenchState::Subst;
  const bool u16 = arm.index == BenchIndex::U16;
  const wgpu::Buffer& args = arm.args == BenchArgs::Gpu ? m_args : m_cpuArgs;
  const uint32_t alignment = std::max<uint32_t>(m_dev.minUniformBufferOffsetAlignment, UniformBytes);
  encoder.SetPipeline(m_renderPipelines[static_cast<size_t>(render_kind(arm))]);
  if (!subst) {
    encoder.SetBindGroup(0, m_positionGroups[0]);
    encoder.SetIndexBuffer(u16 ? m_indices16 : m_indices32,
                           u16 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32);
  }
  if (arm.pipeline == BenchPipeline::Plain) {
    encoder.SetVertexBuffer(0, m_vertices);
  }
  if (arm.state == BenchState::None) {
    const uint32_t offset = 0;
    encoder.SetBindGroup(1, m_uniformGroup, 1, &offset);
    encoder.SetBindGroup(2, m_textureGroups[0]);
    encoder.SetImmediates(0, m_immediates[first].data(), ImmediateBytes);
  }
  uint32_t texture = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t draw = first + i;
    if (subst) {
      encoder.SetBindGroup(0, m_positionGroups[draw & 1]);
      encoder.SetBindGroup(2, m_substTextureGroups[draw & 1]);
      encoder.SetIndexBuffer(m_indices32, wgpu::IndexFormat::Uint32);
    }
    if (arm.state != BenchState::None) {
      encoder.SetImmediates(0, m_immediates[draw].data(), ImmediateBytes);
      if (rebinds(i, draw, m_options.uniformPermille)) {
        const uint32_t offset = draw * alignment;
        encoder.SetBindGroup(1, m_uniformGroup, 1, &offset);
      }
      if (!subst && rebinds(i, draw, m_options.texturePermille)) {
        texture ^= 1;
        encoder.SetBindGroup(2, m_textureGroups[texture]);
      }
    }
    if (arm.mode == BenchMode::Direct) {
      encoder.DrawIndexed(IndicesPerQuad, 1, IndicesPerQuad * draw, 0, 0);
    } else {
      encoder.DrawIndexedIndirect(args, static_cast<uint64_t>(draw) * ArgsWords * sizeof(uint32_t));
    }
  }
}

void Bench::prepare(const BenchArm& arm, const BenchPair& pair, std::vector<wgpu::RenderBundle>& bundles) {
  render_pipeline(render_kind(arm));
  compute_pipeline(compute_kind(arm));
  if (arm.mode != BenchMode::Bundle || !bundles.empty()) {
    return;
  }
  const wgpu::TextureFormat colorFormat =
      m_dev.colorFormat != wgpu::TextureFormat::Undefined ? m_dev.colorFormat : wgpu::TextureFormat::BGRA8Unorm;
  const wgpu::RenderBundleEncoderDescriptor descriptor{
      .label = "render_demo bench bundle",
      .colorFormatCount = 1,
      .colorFormats = &colorFormat,
      .depthStencilFormat = wgpu::TextureFormat::Depth32Float,
      .sampleCount = m_dev.sampleCount,
  };
  for (uint32_t pass = 0; pass < pair.passes; ++pass) {
    const uint32_t first = pass * pair.draws / pair.passes;
    const uint32_t last = (pass + 1) * pair.draws / pair.passes;
    const wgpu::RenderBundleEncoder encoder = m_dev.device.CreateRenderBundleEncoder(&descriptor);
    record_draws(encoder, arm, first, last - first);
    bundles.push_back(encoder.Finish());
  }
}

bool Bench::wait(std::span<wgpu::FutureWaitInfo> futures, std::string& error) const {
  const auto deadline = Clock::now() + WaitTimeout;
  while (true) {
    m_dev.instance.WaitAny(futures.size(), futures.data(), 0);
    if (std::ranges::all_of(futures, [](const wgpu::FutureWaitInfo& info) { return info.completed; })) {
      return true;
    }
    if (Clock::now() >= deadline) {
      error = "timed out waiting for the GPU";
      return false;
    }
  }
}

std::optional<std::string> Bench::pop_scopes() const {
  std::optional<std::string> failure;
  const auto callback = [&failure](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message) {
    if (failure) {
      return;
    }
    if (status != wgpu::PopErrorScopeStatus::Success) {
      failure = "error scope was not popped";
    } else if (type != wgpu::ErrorType::NoError) {
      failure = std::string{std::string_view{message}.substr(0, ErrorChars)};
    }
  };
  std::array<wgpu::FutureWaitInfo, 2> futures{{
      {.future = m_dev.device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly, callback)},
      {.future = m_dev.device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly, callback)},
  }};
  if (std::string error; !wait(futures, error)) {
    return error;
  }
  return failure;
}

bool Bench::iterate(const BenchArm& arm, const BenchPair& pair, const std::vector<wgpu::RenderBundle>& bundles,
                    Iteration& out, std::string& error) {
  const auto start = Clock::now();
  const wgpu::CommandEncoder argsEncoder = m_dev.device.CreateCommandEncoder();
  const wgpu::PassTimestampWrites argsWrites{
      .querySet = m_queries, .beginningOfPassWriteIndex = 0, .endOfPassWriteIndex = 1};
  const wgpu::ComputePassDescriptor argsDescriptor{.timestampWrites = m_timestamps ? &argsWrites : nullptr};
  const wgpu::ComputePassEncoder compute = argsEncoder.BeginComputePass(&argsDescriptor);
  const ComputeKind computeKind = compute_kind(arm);
  compute.SetPipeline(m_computePipelines[static_cast<size_t>(computeKind)]);
  compute.SetBindGroup(0, m_argsGroup);
  if (computeKind == ComputeKind::Noop) {
    compute.DispatchWorkgroups(1);
  } else {
    const std::array<uint32_t, 4> immediates{pair.draws, 0, 0, 0};
    compute.SetImmediates(0, immediates.data(), ArgsImmediateBytes);
    compute.DispatchWorkgroups((pair.draws + ArgsWorkgroup - 1) / ArgsWorkgroup);
  }
  compute.End();
  const wgpu::CommandEncoder encoder = m_dev.device.CreateCommandEncoder();
  for (uint32_t pass = 0; pass < pair.passes; ++pass) {
    const wgpu::RenderPassColorAttachment color{
        .view = m_color,
        .loadOp = pass == 0 ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.0, 0.0, 0.0, 1.0},
    };
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = m_depth,
        .depthLoadOp = pass == 0 ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.f,
    };
    const wgpu::PassTimestampWrites writes{
        .querySet = m_queries, .beginningOfPassWriteIndex = 2 + 2 * pass, .endOfPassWriteIndex = 3 + 2 * pass};
    const wgpu::RenderPassDescriptor descriptor{
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
        .depthStencilAttachment = &depth,
        .timestampWrites = m_timestamps ? &writes : nullptr,
    };
    const wgpu::RenderPassEncoder render = encoder.BeginRenderPass(&descriptor);
    const uint32_t first = pass * pair.draws / pair.passes;
    const uint32_t last = (pass + 1) * pair.draws / pair.passes;
    if (arm.mode == BenchMode::Bundle) {
      render.ExecuteBundles(1, &bundles[pass]);
    } else if (arm.mode == BenchMode::Mdi) {
      record_draws(render, arm, first, 0);
      render.MultiDrawIndexedIndirect(arm.args == BenchArgs::Gpu ? m_args : m_cpuArgs,
                                      static_cast<uint64_t>(first) * ArgsWords * sizeof(uint32_t), last - first);
    } else {
      record_draws(render, arm, first, last - first);
    }
    render.End();
  }
  const uint32_t queries = 2 + 2 * pair.passes;
  if (m_timestamps) {
    encoder.ResolveQuerySet(m_queries, 0, queries, m_resolve, 0);
    encoder.CopyBufferToBuffer(m_resolve, 0, m_readback, 0, queries * sizeof(uint64_t));
  }
  const auto encoded = Clock::now();
  const std::array<wgpu::CommandBuffer, 2> commands{argsEncoder.Finish(), encoder.Finish()};
  const auto finished = Clock::now();
  m_dev.queue.Submit(commands.size(), commands.data());
  const auto submitted = Clock::now();
  out.encodeUs = micros(encoded - start);
  out.finishUs = micros(finished - encoded);
  out.submitUs = micros(submitted - finished);
  out.totalUs = micros(submitted - start);

  bool mapped = false;
  std::array<wgpu::FutureWaitInfo, 2> futures{{
      {.future = m_dev.queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                                 [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {})},
  }};
  size_t futureCount = 1;
  if (m_timestamps) {
    futures[1].future = m_readback.MapAsync(
        wgpu::MapMode::Read, 0, queries * sizeof(uint64_t), wgpu::CallbackMode::WaitAnyOnly,
        [&mapped](wgpu::MapAsyncStatus status, wgpu::StringView) { mapped = status == wgpu::MapAsyncStatus::Success; });
    futureCount = 2;
  }
  if (!wait(std::span{futures}.first(futureCount), error)) {
    return false;
  }
  if (!m_timestamps) {
    return true;
  }
  if (!mapped) {
    error = "timestamp readback failed to map";
    return false;
  }
  std::array<uint64_t, QueryCount> ticks{};
  std::memcpy(ticks.data(), m_readback.GetConstMappedRange(0, queries * sizeof(uint64_t)), queries * sizeof(uint64_t));
  m_readback.Unmap();
  bool valid = true;
  for (uint32_t i = 0; i < queries; ++i) {
    valid = valid && ticks[i] != 0 && (i == 0 || ticks[i] >= ticks[i - 1]);
  }
  if (!valid) {
    return true;
  }
  double inPass = 0;
  for (uint32_t pass = 0; pass < pair.passes; ++pass) {
    inPass += static_cast<double>(ticks[3 + 2 * pass] - ticks[2 + 2 * pass]);
  }
  out.gpuValid = true;
  out.gpuArgsUs = static_cast<double>(ticks[1] - ticks[0]) / 1000.0;
  out.gpuTotalUs = static_cast<double>(ticks[queries - 1] - ticks[1]) / 1000.0;
  out.gpuInPassUs = inPass / 1000.0;
  return true;
}

void record(const Iteration& it, ArmSamples& samples, bool gpuColumns) {
  samples.encodeUs.push_back(it.encodeUs);
  samples.finishUs.push_back(it.finishUs);
  samples.submitUs.push_back(it.submitUs);
  samples.totalUs.push_back(it.totalUs);
  if (!gpuColumns) {
    return;
  }
  samples.gpuInvalid += it.gpuValid ? 0 : 1;
  samples.gpuTotalUs.push_back(it.gpuTotalUs);
  samples.gpuInPassUs.push_back(it.gpuInPassUs);
  samples.gpuArgsUs.push_back(it.gpuArgsUs);
}

bool Bench::run_pairs(BenchPump pump, void* userdata, std::string& error) {
  const bool gpuColumns = m_timestamps && !m_dev.nullBackend;
  // Dawn's D3D12 backend crashes replaying a bundle's indirect draws that read vertex_index when validation is skipped
  const bool bundleIndexCrash = m_run.backend == "d3d12" && m_run.validation == "skipped";
  uint32_t runnable = 0;
  for (const BenchPair& pair : bench_pairs()) {
    PairResult result{.pair = pair};
    const bool usesMdi = pair.a.mode == BenchMode::Mdi || pair.b.mode == BenchMode::Mdi;
    const auto viBundle = [](const BenchArm& arm) {
      return arm.mode == BenchMode::Bundle && arm.pipeline == BenchPipeline::VertexIndex;
    };
    if (usesMdi && !m_dev.multiDrawIndirect) {
      result.clamp = "no-mdi";
    } else if (bundleIndexCrash && (viBundle(pair.a) || viBundle(pair.b))) {
      result.clamp = "no-vi-bundle";
    }
    if (!result.clamp.empty()) {
      m_run.pairs.push_back(std::move(result));
      continue;
    }
    std::array<std::vector<wgpu::RenderBundle>, 2> bundles;
    const std::array<const BenchArm*, 2> arms{&pair.a, &pair.b};
    const auto block = [&](ArmSamples* a, ArmSamples* b) {
      for (const size_t arm : {0, 1, 1, 0}) {
        Iteration it;
        if (!iterate(*arms[arm], pair, bundles[arm], it, error)) {
          return false;
        }
        if (ArmSamples* samples = arm == 0 ? a : b) {
          record(it, *samples, gpuColumns);
        }
      }
      return true;
    };
    m_dev.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    m_dev.device.PushErrorScope(wgpu::ErrorFilter::Internal);
    prepare(pair.a, pair, bundles[0]);
    prepare(pair.b, pair, bundles[1]);
    const bool warmed = block(nullptr, nullptr);
    const std::optional<std::string> failure = pop_scopes();
    if (!warmed) {
      return false;
    }
    if (failure) {
      result.clamp = "error: " + *failure;
      std::printf("render_demo: bench: pair %s errored: %s\n", pair_name(pair).c_str(), failure->c_str());
      ++m_run.erroredPairs;
      m_run.pairs.push_back(std::move(result));
      continue;
    }
    for (uint32_t i = 1; i < m_run.warmupBlocks; ++i) {
      if (!block(nullptr, nullptr)) {
        return false;
      }
    }
    ArmSamples a;
    ArmSamples b;
    for (uint32_t i = 0; i < m_options.blocks; ++i) {
      if (!block(&a, &b)) {
        return false;
      }
    }
    m_run.pairs.push_back(reduce_pair(pair, a, b));
    ++runnable;
    if (pump != nullptr && !pump(userdata)) {
      error = "aborted by an exit request";
      return false;
    }
  }
  if (runnable == 0 && !m_dev.nullBackend) {
    error = "no pair could run";
    return false;
  }
  return true;
}
} // namespace

bool run_indirect_bench(const BenchDevice& device, const BenchOptions& options, BenchPump pump, void* userdata,
                        BenchRun& run, std::string& error) {
  if (options.blocks == 0 || options.width == 0 || options.height == 0) {
    error = "invalid bench options";
    return false;
  }
  Bench bench{device, options, run};
  return bench.setup(error) && bench.run_pairs(pump, userdata, error);
}
} // namespace render_demo
