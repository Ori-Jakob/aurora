#include "dual_source.hpp"

#include "../json.hpp"
#include "../../gx/pipeline.hpp"
#include "../../io.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/gfx.hpp>
#include <SDL3/SDL_stdinc.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <optional>

namespace aurora::render::debug::dual_source {
namespace {
constexpr Module Log{"aurora::render::dual_source"};

std::optional<std::string> g_path;
Report g_report;
Counts g_last{};

Counts live_counts() noexcept {
  const gx::DstAlphaCounts counts = gx::dst_alpha_counts();
  return {counts.dstAlphaDraws, counts.prepassDraws, counts.dualSourcePipelines, counts.prepassPipelines};
}

Counts difference(const Counts& now, const Counts& before) noexcept {
  return {
      .dstAlphaDraws = now.dstAlphaDraws - before.dstAlphaDraws,
      .prepassDraws = now.prepassDraws - before.prepassDraws,
      .dualSourcePipelines = now.dualSourcePipelines - before.dualSourcePipelines,
      .prepassPipelines = now.prepassPipelines - before.prepassPipelines,
  };
}

void add(Report& report, const Counts& delta, uint32_t colorAttachments, uint64_t frames) {
  report.frames += frames;
  report.dualSourcePipelines += delta.dualSourcePipelines;
  report.prepassPipelines += delta.prepassPipelines;
  auto it = std::ranges::find(report.layouts, colorAttachments, &LayoutCounts::colorAttachments);
  if (it == report.layouts.end()) {
    report.layouts.push_back({.colorAttachments = colorAttachments});
    it = report.layouts.end() - 1;
  }
  it->frames += frames;
  it->dstAlphaDraws += delta.dstAlphaDraws;
  it->prepassDraws += delta.prepassDraws;
}
} // namespace

void accumulate(Report& report, const Counts& delta, uint32_t colorAttachments) {
  add(report, delta, colorAttachments, 1);
}

uint64_t lost_dual_source_draws(const Report& report) noexcept {
  if (!report.dualSourceBlending) {
    return 0;
  }
  uint64_t lost = 0;
  for (const LayoutCounts& layout : report.layouts) {
    lost += layout.prepassDraws;
  }
  return lost;
}

std::string format_json(const Report& report) {
  const json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  if (!doc) {
    return {};
  }
  yyjson_mut_doc* d = doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(d);
  yyjson_mut_doc_set_root(d, root);
  yyjson_mut_obj_add_strncpy(d, root, "schema", Schema.data(), Schema.size());
  yyjson_mut_obj_add_uint(d, root, "version", 1);
  yyjson_mut_val* device = yyjson_mut_obj_add_obj(d, root, "device");
  yyjson_mut_obj_add_bool(d, device, "dualSourceBlending", report.dualSourceBlending);
  yyjson_mut_obj_add_strncpy(d, device, "backend", report.backend.data(), report.backend.size());
  yyjson_mut_obj_add_uint(d, root, "frames", report.frames);
  yyjson_mut_val* pipelines = yyjson_mut_obj_add_obj(d, root, "pipelines");
  yyjson_mut_obj_add_uint(d, pipelines, "dualSource", report.dualSourcePipelines);
  yyjson_mut_obj_add_uint(d, pipelines, "alphaPrepass", report.prepassPipelines);
  yyjson_mut_val* layouts = yyjson_mut_obj_add_arr(d, root, "layouts");
  std::vector<LayoutCounts> sorted = report.layouts;
  std::ranges::sort(sorted, {}, &LayoutCounts::colorAttachments);
  for (const LayoutCounts& layout : sorted) {
    yyjson_mut_val* obj = yyjson_mut_arr_add_obj(d, layouts);
    yyjson_mut_obj_add_uint(d, obj, "colorAttachments", layout.colorAttachments);
    yyjson_mut_obj_add_uint(d, obj, "frames", layout.frames);
    yyjson_mut_obj_add_uint(d, obj, "dstAlphaDraws", layout.dstAlphaDraws);
    yyjson_mut_obj_add_uint(d, obj, "prepassDraws", layout.prepassDraws);
  }
  const uint64_t lost = lost_dual_source_draws(report);
  yyjson_mut_obj_add_uint(d, root, "lostDualSourceDraws", lost);
  const std::string_view note =
      !report.dualSourceBlending
          ? "the device has no dual-source blending, so every layout draws the alpha prepass; MRT loses nothing"
      : lost == 0 ? "no draw lost dual-source blending to a multi-target layout"
                  : "prepass draws under multi-target layouts lost dual-source blending";
  yyjson_mut_obj_add_strncpy(d, root, "note", note.data(), note.size());
  return json::write(d, true);
}

void initialize() {
  const char* value = SDL_getenv("AURORA_RENDER_DUAL_SOURCE_JSON");
  if (value == nullptr || *value == '\0') {
    return;
  }
  g_path = value;
  g_report = {
      .dualSourceBlending = webgpu::g_dualSourceBlendingSupported,
      .backend = std::string{magic_enum::enum_name(webgpu::g_backendType)},
  };
  g_last = live_counts();
  Log.warn("AURORA_RENDER_DUAL_SOURCE_JSON: counting dual-source loss into {}", *g_path);
}

void on_latch() {
  if (!g_path) {
    return;
  }
  const Counts now = live_counts();
  accumulate(g_report, difference(now, g_last), gfx::scene_render_target_layout().colorAttachmentCount);
  g_last = now;
}

void shutdown() noexcept {
  if (!g_path) {
    return;
  }
  const Counts tail = difference(live_counts(), g_last);
  if (tail.dstAlphaDraws != 0 || tail.prepassDraws != 0 || tail.dualSourcePipelines != 0 ||
      tail.prepassPipelines != 0) {
    add(g_report, tail, gfx::scene_render_target_layout().colorAttachmentCount, 0);
  }
  const std::string text = format_json(g_report);
  const auto path = io::fs_path_from_string(*g_path);
  if (text.empty() || !io::write_file_atomic(path, {reinterpret_cast<const uint8_t*>(text.data()), text.size()})) {
    Log.warn("AURORA_RENDER_DUAL_SOURCE_JSON: failed to write {}", *g_path);
  } else {
    Log.info("Dual-source report: {} lost draws over {} frames, written to {}", lost_dual_source_draws(g_report),
             g_report.frames, *g_path);
  }
  g_path.reset();
  g_report = {};
  g_last = {};
}

bool active() noexcept { return g_path.has_value(); }
} // namespace aurora::render::debug::dual_source
