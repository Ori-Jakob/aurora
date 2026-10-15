#include "debug.hpp"

#include "../jobs.hpp"
#include "../shaders.hpp"
#include "../../webgpu/feature_profile.hpp"
#include "../../webgpu/gpu.hpp"
#include "../../webgpu/gpu_prof.hpp"

#include <imgui.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <string_view>
#include <vector>

namespace aurora::render::debug {
namespace {
constexpr double MiB = 1024.0 * 1024.0;
constexpr size_t TopPassCount = 10;
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;

// Adapter strings and enum names can be empty views with no storage behind them
std::string_view printable(std::string_view value) noexcept { return value.empty() ? std::string_view{""} : value; }

void text(std::string_view value) {
  value = printable(value);
  ImGui::TextUnformatted(value.data(), value.data() + value.size());
}

void labelled(std::string_view label, std::string_view value) {
  value = printable(value);
  ImGui::Text("%.*s: %.*s", static_cast<int>(label.size()), label.data(), static_cast<int>(value.size()), value.data());
}

void draw_device() {
  ImGui::SeparatorText("Device");
  const auto& info = webgpu::g_adapterInfo;
  labelled("Backend", magic_enum::enum_name(webgpu::g_backendType));
  labelled("Adapter", std::string_view{info.device});
  labelled("Vendor", std::string_view{info.vendor});
  labelled("Description", std::string_view{info.description});
  labelled("Feature profile", webgpu::feature_profile_name(webgpu::g_featureProfile));
  labelled("Core features", webgpu::g_hasCoreFeatures ? "yes" : "no");
  labelled("GPU timestamps",
           webgpu::gpu_prof::supported() ? "available, collecting while this window is open" : "unavailable");
}

void draw_settings() {
  ImGui::SeparatorText("Settings");
  labelled("Preset", magic_enum::enum_name(effective_settings().preset));
  if (!ImGui::BeginTable("##features", 3, TableFlags)) {
    return;
  }
  ImGui::TableSetupColumn("Feature");
  ImGui::TableSetupColumn("Active");
  ImGui::TableSetupColumn("Support");
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < FeatureCount; ++i) {
    const auto feature = static_cast<Feature>(i);
    const FeatureSupport support = feature_support(feature);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    text(feature_name(feature));
    ImGui::TableNextColumn();
    text(feature_active(feature) ? "yes" : "no");
    ImGui::TableNextColumn();
    if (support.reason != Reason::None) {
      text(reason_name(support.reason));
    } else {
      text(support.supported ? "Supported" : "Unsupported");
    }
  }
  ImGui::EndTable();
}

void draw_gpu() {
  ImGui::SeparatorText("GPU");
  if (!webgpu::gpu_prof::supported()) {
    ImGui::TextUnformatted("GPU timings unavailable without timestamp queries");
    return;
  }
  const TimingAverages& averages = timing_averages();
  if (averages.sequence == 0) {
    ImGui::TextUnformatted("Waiting for the first timed frame");
    return;
  }
  ImGui::Text("GPU frame: %.3f ms", averages.frameMs);
  if (averages.dropped > 0) {
    ImGui::TextColored(WarningColor, "Dropped zones: %u", averages.dropped);
  } else {
    ImGui::TextUnformatted("Dropped zones: 0");
  }
  std::vector<const AveragedPass*> top;
  for (const AveragedPass& pass : averages.passes) {
    if (pass.inLatest) {
      top.push_back(&pass);
    }
  }
  const size_t count = std::min(top.size(), TopPassCount);
  std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(count), top.end(),
                    [](const AveragedPass* a, const AveragedPass* b) { return a->ms > b->ms; });
  if (count == 0 || !ImGui::BeginTable("##top_passes", 2, TableFlags)) {
    return;
  }
  ImGui::TableSetupColumn("Pass");
  ImGui::TableSetupColumn("GPU ms");
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < count; ++i) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(top[i]->name);
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", top[i]->ms);
  }
  ImGui::EndTable();
}

void draw_pipelines() {
  ImGui::SeparatorText("Pipelines");
  const shaders::PipelineStats stats = shaders::stats();
  ImGui::Text("Pending %u, ready %u, failed %u, modules %u", stats.pending, stats.ready, stats.failed, stats.modules);
  const std::vector<shaders::PipelineFailure> failures = shaders::failures();
  if (failures.empty()) {
    return;
  }
  if (ImGui::TreeNode("##pipeline_failures", "Failures (%zu)", failures.size())) {
    for (const shaders::PipelineFailure& failure : failures) {
      ImGui::TextWrapped("%s: %s", failure.label.c_str(), failure.message.c_str());
    }
    ImGui::TreePop();
  }
}

void draw_jobs() {
  ImGui::SeparatorText("Jobs");
  const jobs::JobStats stats = jobs::stats();
  ImGui::Text("Workers %u, running %u", stats.workers, stats.running);
  ImGui::Text("Queued: high %u, normal %u, low %u", stats.queued[0], stats.queued[1], stats.queued[2]);
  ImGui::Text("Completed %llu, cancelled %llu", static_cast<unsigned long long>(stats.completed),
              static_cast<unsigned long long>(stats.cancelled));
}

void draw_vram() {
  ImGui::SeparatorText("VRAM");
  for (const VramCounter& source : vram_sources()) {
    const uint64_t bytes = source.bytes != nullptr ? source.bytes() : 0;
    ImGui::Text("%s: %.1f MiB", source.name.c_str(), static_cast<double>(bytes) / MiB);
  }
  const Caps& deviceCaps = caps();
  if (deviceCaps.vramBytes != 0) {
    const std::string_view source = vram_source_name(deviceCaps.vramSource);
    ImGui::Text("Device: %.0f MiB (%.*s)", static_cast<double>(deviceCaps.vramBytes) / MiB,
                static_cast<int>(source.size()), source.data());
  }
}
} // namespace

void draw_overview_tab() {
  draw_device();
  draw_settings();
  draw_gpu();
  draw_pipelines();
  draw_jobs();
  draw_vram();
}
} // namespace aurora::render::debug
