#include "graph_layout.hpp"

#include "graph_slots.hpp"

#include "../gfx/hash.hpp"

#include <algorithm>
#include <optional>
#include <string_view>

namespace aurora::render::detail {
namespace {
constexpr auto FrameVisibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment | wgpu::ShaderStage::Compute;

constexpr bool is_attachment(Access access) noexcept {
  return access == Access::ColorWrite || access == Access::ColorRMW || access == Access::DepthRead ||
         access == Access::DepthWrite;
}

wgpu::TextureViewDimension natural_view_dimension(const CompiledResource& resource) noexcept {
  if (resource.kind == ResourceKind::Gx || !resource.described) {
    return wgpu::TextureViewDimension::e2D;
  }
  if (resource.tex.dimension == wgpu::TextureDimension::e3D) {
    return wgpu::TextureViewDimension::e3D;
  }
  return resource.tex.depthOrLayers > 1 ? wgpu::TextureViewDimension::e2DArray : wgpu::TextureViewDimension::e2D;
}

// Matches dummy_key, so a GX resource the environment lacks still binds its fixed format's dummy
wgpu::TextureFormat resource_format(const CompiledResource& resource) noexcept {
  if (resource.kind == ResourceKind::Gx && resource.tex.format == wgpu::TextureFormat::Undefined) {
    return gx_fixed_format(resource.gx);
  }
  return resource.tex.format;
}

wgpu::TextureSampleType sample_type(const CompiledUse& use, const CompiledResource& resource,
                                    const GraphFeatures& features) noexcept {
  if (use.options.sampleType != wgpu::TextureSampleType::Undefined) {
    return use.options.sampleType;
  }
  const auto info = texture_format_info(resource_format(resource));
  if (!info) {
    return wgpu::TextureSampleType::Float;
  }
  if (info->sampleType == wgpu::TextureSampleType::UnfilterableFloat && features.float32Filterable) {
    return wgpu::TextureSampleType::Float;
  }
  return info->sampleType;
}

bool efb_attachment(const CompiledResource& resource) noexcept {
  return resource.kind == ResourceKind::Gx && (resource.gx == GxResource::Color || resource.gx == GxResource::Depth);
}

bool mixed_samples(const BindingPlan& plan) noexcept {
  std::optional<uint32_t> samples;
  const auto differs = [&](const AttachmentSlot& attachment) {
    if (samples && *samples != attachment.samples) {
      return true;
    }
    samples = attachment.samples;
    return false;
  };
  return std::ranges::any_of(plan.colors, differs) || (plan.depth && differs(*plan.depth));
}
} // namespace

std::span<const wgpu::BindGroupLayoutEntry> frame_layout_entries() noexcept {
  static const std::array<wgpu::BindGroupLayoutEntry, 6> Entries{{
      {
          .binding = static_cast<uint32_t>(FrameBinding::Constants),
          .visibility = FrameVisibility,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true},
      },
      {
          .binding = static_cast<uint32_t>(FrameBinding::Params),
          .visibility = FrameVisibility,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true},
      },
      {
          .binding = static_cast<uint32_t>(FrameBinding::LinearClamp),
          .visibility = FrameVisibility,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      },
      {
          .binding = static_cast<uint32_t>(FrameBinding::NearestClamp),
          .visibility = FrameVisibility,
          .sampler = {.type = wgpu::SamplerBindingType::NonFiltering},
      },
      {
          .binding = static_cast<uint32_t>(FrameBinding::LinearRepeat),
          .visibility = FrameVisibility,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      },
      {
          .binding = static_cast<uint32_t>(FrameBinding::DepthCompare),
          .visibility = FrameVisibility,
          .sampler = {.type = wgpu::SamplerBindingType::Comparison},
      },
  }};
  return Entries;
}

BindingType binding_type(Access access, bool buffer) noexcept {
  switch (access) {
  case Access::Sampled:
    return buffer ? BindingType::None : BindingType::SampledTexture;
  case Access::Uniform:
    return buffer ? BindingType::UniformBuffer : BindingType::None;
  case Access::StorageRead:
    return buffer ? BindingType::ReadOnlyStorageBuffer : BindingType::ReadOnlyStorageTexture;
  case Access::StorageWrite:
    return buffer ? BindingType::StorageBuffer : BindingType::WriteOnlyStorageTexture;
  case Access::StorageRW:
    return buffer ? BindingType::StorageBuffer : BindingType::ReadWriteStorageTexture;
  default:
    return BindingType::None;
  }
}

wgpu::TextureViewDimension use_view_dimension(const CompiledUse& use, const CompiledResource& resource) noexcept {
  if (use.options.dimension != wgpu::TextureViewDimension::Undefined) {
    return use.options.dimension;
  }
  return natural_view_dimension(resource);
}

CullReason derive_binding_plan(const BindingInputs& inputs, BindingPlan& out, ResourceId* mismatch) {
  out = BindingPlan{.paramsSize = inputs.paramsSize, .manual = inputs.manual};
  const GraphFeatures features = inputs.env != nullptr ? inputs.env->limits.features : GraphFeatures{};
  const uint32_t gxSamples = inputs.env != nullptr ? std::max(inputs.env->gxSampleCount, 1u) : 1;
  uint32_t binding = 0;
  for (uint32_t i = 0; i < inputs.uses.size(); ++i) {
    const CompiledUse& use = inputs.uses[i];
    const CompiledResource& resource = inputs.resources[use.resource];
    if (is_attachment(use.access)) {
      const AttachmentSlot attachment{
          .use = i,
          .resource = use.resource,
          .access = use.access,
          .creatorWrite = use.role == UseRole::Create || use.role == UseRole::History,
          .samples = efb_attachment(resource) ? gxSamples : 1,
      };
      if (use.access == Access::DepthRead || use.access == Access::DepthWrite) {
        if (!out.depth) {
          out.depth = attachment;
        }
      } else {
        out.colors.push_back(attachment);
      }
      continue;
    }
    const BindingType type = binding_type(use.access, use.buffer);
    if (type == BindingType::None || inputs.manual) {
      continue;
    }
    BindingSlot slot{
        .use = i,
        .resource = use.resource,
        .binding = binding++,
        .type = type,
        .visibility = use.visibility,
        .prev = use.role == UseRole::Prev,
    };
    if (!use.buffer) {
      slot.viewDimension = use_view_dimension(use, resource);
      if (type == BindingType::SampledTexture) {
        slot.sampleType = sample_type(use, resource, features);
      } else {
        const wgpu::TextureFormat format = resource_format(resource);
        slot.storageFormat = format != wgpu::TextureFormat::Undefined ? format : use.options.format;
      }
    }
    out.slots.push_back(slot);
  }
  out.layoutHash = layout_hash(out);

  if (mixed_samples(out)) {
    return CullReason::SampleCountMismatch;
  }
  if (features.core || inputs.bindingViews.empty()) {
    return CullReason::None;
  }
  for (const BindingSlot& slot : out.slots) {
    const CompiledResource& resource = inputs.resources[slot.resource];
    const bool pooled = resource.kind == ResourceKind::Transient || resource.kind == ResourceKind::History;
    if (slot.type != BindingType::SampledTexture || !pooled || !resource.described ||
        inputs.uses[slot.use].staticDummy) {
      continue;
    }
    // Compat textures sample through the one view dimension they were created with
    auto view = natural_view_dimension(resource);
    if (slot.resource < inputs.bindingViews.size() &&
        inputs.bindingViews[slot.resource] != wgpu::TextureViewDimension::Undefined) {
      view = inputs.bindingViews[slot.resource];
    }
    if (slot.viewDimension != view) {
      if (mismatch != nullptr) {
        *mismatch = slot.resource;
      }
      return CullReason::BindingViewMismatch;
    }
  }
  return CullReason::None;
}

uint64_t layout_hash(const BindingPlan& plan) noexcept {
  Hasher hasher{0};
  constexpr std::string_view Tag = "aurora.render.layout/1";
  hasher.update(Tag.data(), Tag.size());
  hasher.update(static_cast<uint8_t>(plan.manual));
  hasher.update(static_cast<uint32_t>(plan.slots.size()));
  for (const BindingSlot& slot : plan.slots) {
    hasher.update(slot.binding);
    hasher.update(static_cast<uint8_t>(slot.type));
    hasher.update(static_cast<uint64_t>(slot.visibility));
    hasher.update(static_cast<uint32_t>(slot.viewDimension));
    hasher.update(static_cast<uint32_t>(slot.sampleType));
    hasher.update(static_cast<uint32_t>(slot.storageFormat));
    hasher.update(static_cast<uint8_t>(slot.multisampled));
  }
  return hasher.digest();
}
} // namespace aurora::render::detail
