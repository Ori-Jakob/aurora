#include "views.hpp"

#include "capture.hpp"
#include "dual_source.hpp"
#include "gbuf_view.hpp"

#include "../graph.hpp"
#include "../../gx/pipeline.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/gfx.hpp>
#include <aurora/render.hpp>
#include <imgui.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug {
namespace {
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
constexpr std::array<std::string_view, 5> CaptureResources{
    resource::GxColor, resource::GxNormal, resource::Motion, resource::Reactive, resource::Fog,
};

struct TabState {
  int resource = 0;
  int anchor = 0;
  std::string lastRequest;
};

TabState g_state;

void text(std::string_view value) { ImGui::TextUnformatted(value.data(), value.data() + value.size()); }

std::string_view semantic_name(gfx::ColorAttachmentSemantic semantic) noexcept {
  switch (semantic) {
  case gfx::ColorAttachmentSemantic::SceneColor:
    return "Scene colour";
  case gfx::ColorAttachmentSemantic::Normal:
    return "Normal";
  case gfx::ColorAttachmentSemantic::Auxiliary:
    return "Auxiliary";
  case gfx::ColorAttachmentSemantic::DebugId:
    return "Debug id";
  case gfx::ColorAttachmentSemantic::Motion:
    return "Motion";
  case gfx::ColorAttachmentSemantic::Reactive:
    return "Reactive";
  case gfx::ColorAttachmentSemantic::Fog:
    return "Fog record";
  }
  return "Unknown";
}

bool input_present(std::string_view name) noexcept {
  const std::optional<detail::GxResource> gx = detail::find_gx_resource(name);
  if (!gx) {
    return false;
  }
  switch (*gx) {
  case detail::GxResource::Color:
    return true;
  case detail::GxResource::Normal:
    return webgpu::normal_present();
  case detail::GxResource::Motion:
  case detail::GxResource::Reactive:
  case detail::GxResource::Fog:
    return webgpu::g_graphicsConfig.gbuffer;
  default:
    return false;
  }
}

std::vector<std::string_view> capture_anchors() {
  std::vector<std::string_view> anchors;
  for (const detail::ChainNode& node : detail::chain_nodes()) {
    if (node.stage != Stage::AfterFrame) {
      anchors.push_back(node.name);
    }
  }
  return anchors;
}

void draw_layout() {
  ImGui::SeparatorText("Scene layout");
  const gfx::RenderTargetLayout layout = gfx::scene_render_target_layout();
  std::array<wgpu::TextureFormat, gfx::MaxColorAttachments> formats{};
  const uint32_t count = std::min<uint32_t>(layout.colorAttachmentCount, gfx::MaxColorAttachments);
  if (ImGui::BeginTable("##layout", 3, TableFlags)) {
    ImGui::TableSetupColumn("Attachment");
    ImGui::TableSetupColumn("Format");
    ImGui::TableSetupColumn("Size");
    ImGui::TableHeadersRow();
    for (uint32_t i = 0; i < count; ++i) {
      const gfx::ColorAttachmentLayout& attachment = layout.colorAttachments[i];
      formats[i] = attachment.format;
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(semantic_name(attachment.semantic));
      ImGui::TableNextColumn();
      text(magic_enum::enum_name(attachment.format));
      ImGui::TableNextColumn();
      ImGui::Text("%ux%u", attachment.width, attachment.height);
    }
    ImGui::EndTable();
  }
  ImGui::Text("Colour bytes per sample: %u", detail::color_bytes_per_sample(std::span{formats}.first(count)));
  const FeatureSupport support = feature_support(Feature::GBuffer);
  if (support.supported) {
    ImGui::Text("G-buffer tier: %s", webgpu::g_graphicsConfig.gbuffer ? "live" : "supported, off");
  } else {
    const std::string_view reason = reason_name(support.reason);
    ImGui::TextColored(WarningColor, "G-buffer tier unavailable: %.*s", static_cast<int>(reason.size()), reason.data());
  }
}

void draw_view_buttons() {
  ImGui::SeparatorText("G-buffer view");
  const uint16_t current = view_override();
  if (ImGui::RadioButton("Off", current == 0 || find_gbuf_view(current) == nullptr)) {
    set_view_override(0);
  }
  for (const GbufView& view : GbufViews) {
    const bool present = input_present(view.resource);
    ImGui::BeginDisabled(!present);
    const std::string label{view.label};
    if (ImGui::RadioButton(label.c_str(), current == view.id)) {
      set_view_override(view.id);
    }
    ImGui::EndDisabled();
    if (!present && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("%.*s is not in the scene layout", static_cast<int>(view.resource.size()),
                        view.resource.data());
    }
  }
  if (const GbufView* active = gbuf_view::active_view()) {
    ImGui::Text("Snapshot of %.*s after %.*s", static_cast<int>(active->resource.size()), active->resource.data(),
                static_cast<int>(active->anchor.size()), active->anchor.data());
  }
}

void draw_capture() {
  ImGui::SeparatorText("Capture");
  const std::vector<std::string_view> anchors = capture_anchors();
  g_state.resource = std::clamp(g_state.resource, 0, static_cast<int>(CaptureResources.size()) - 1);
  g_state.anchor = std::clamp(g_state.anchor, 0, static_cast<int>(anchors.size()) - 1);
  const std::string resourcePreview{CaptureResources[g_state.resource]};
  if (ImGui::BeginCombo("Resource", resourcePreview.c_str())) {
    for (int i = 0; i < static_cast<int>(CaptureResources.size()); ++i) {
      const std::string name{CaptureResources[i]};
      if (ImGui::Selectable(name.c_str(), g_state.resource == i)) {
        g_state.resource = i;
      }
    }
    ImGui::EndCombo();
  }
  const std::string anchorPreview{anchors.empty() ? std::string_view{} : anchors[g_state.anchor]};
  if (ImGui::BeginCombo("After", anchorPreview.c_str())) {
    for (int i = 0; i < static_cast<int>(anchors.size()); ++i) {
      const std::string name{anchors[i]};
      if (ImGui::Selectable(name.c_str(), g_state.anchor == i)) {
        g_state.anchor = i;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::BeginDisabled(!input_present(CaptureResources[g_state.resource]) || anchors.empty());
  if (ImGui::Button("Capture once")) {
    const capture::Request request{
        .resource = std::string{CaptureResources[g_state.resource]},
        .anchor = std::string{anchors[g_state.anchor]},
        .dir = capture::default_dir(),
    };
    g_state.lastRequest = capture::request(request) ? request.resource + "@" + request.anchor + " requested"
                                                    : std::string{"Every capture slot is busy"};
  }
  ImGui::EndDisabled();
  if (!g_state.lastRequest.empty()) {
    ImGui::SameLine();
    text(g_state.lastRequest);
  }
  const std::vector<capture::EntryInfo> entries = capture::entries();
  if (entries.empty()) {
    ImGui::TextUnformatted("No captures");
    return;
  }
  if (ImGui::BeginTable("##captures", 4, TableFlags)) {
    ImGui::TableSetupColumn("Target");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Dropped");
    ImGui::TableSetupColumn("Detail", ImGuiTableColumnFlags_WidthStretch, 3.f);
    ImGui::TableHeadersRow();
    for (const capture::EntryInfo& entry : entries) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(entry.request.resource + "@" + entry.request.anchor);
      ImGui::TableNextColumn();
      text(capture::status_name(entry.status));
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(entry.dropped));
      ImGui::TableNextColumn();
      ImGui::TextWrapped("%s", entry.message.c_str());
    }
    ImGui::EndTable();
  }
}

void draw_dual_source() {
  ImGui::SeparatorText("Dual-source blending");
  const gx::DstAlphaCounts counts = gx::dst_alpha_counts();
  ImGui::Text("Device support: %s%s", webgpu::g_dualSourceBlendingSupported ? "yes" : "no",
              dual_source::active() ? " (report on)" : "");
  ImGui::Text("Pipelines: %llu dual-source, %llu alpha prepass",
              static_cast<unsigned long long>(counts.dualSourcePipelines),
              static_cast<unsigned long long>(counts.prepassPipelines));
  ImGui::Text("Draws: %llu destination alpha, %llu with a prepass",
              static_cast<unsigned long long>(counts.dstAlphaDraws),
              static_cast<unsigned long long>(counts.prepassDraws));
}
} // namespace

void draw_views_tab() {
  draw_layout();
  draw_view_buttons();
  draw_capture();
  draw_dual_source();
}
} // namespace aurora::render::debug
