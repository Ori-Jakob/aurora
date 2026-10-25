#include "graph_inspection.hpp"

#include "../json.hpp"
#include "../../webgpu/gpu_prof.hpp"

#include <fmt/format.h>

#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug {
namespace {
constexpr std::string_view StageNotRun = "StageNotRun";

class Writer {
public:
  explicit Writer(yyjson_mut_doc* doc) noexcept : m_doc(doc) {}

  [[nodiscard]] yyjson_mut_val* object() const noexcept { return yyjson_mut_obj(m_doc); }

  // yyjson keeps string pointers unless told to copy, and some values here are temporaries
  void add(yyjson_mut_val* obj, const char* key, std::string_view value) const noexcept {
    yyjson_mut_obj_add_strncpy(m_doc, obj, key, value.data(), value.size());
  }
  void add(yyjson_mut_val* obj, const char* key, const std::optional<std::string>& value) const noexcept {
    if (value) {
      add(obj, key, std::string_view{*value});
    } else {
      yyjson_mut_obj_add_null(m_doc, obj, key);
    }
  }
  void add(yyjson_mut_val* obj, const char* key, uint64_t value) const noexcept {
    yyjson_mut_obj_add_uint(m_doc, obj, key, value);
  }
  void add(yyjson_mut_val* obj, const char* key, std::optional<uint64_t> value) const noexcept {
    if (value) {
      add(obj, key, *value);
    } else {
      yyjson_mut_obj_add_null(m_doc, obj, key);
    }
  }
  void add_bool(yyjson_mut_val* obj, const char* key, bool value) const noexcept {
    yyjson_mut_obj_add_bool(m_doc, obj, key, value);
  }
  void add_ms(yyjson_mut_val* obj, const char* key, double ms) const noexcept {
    yyjson_mut_obj_add_real(m_doc, obj, key, std::round(ms * 1000.0) / 1000.0);
  }
  [[nodiscard]] yyjson_mut_val* add_array(yyjson_mut_val* obj, const char* key) const noexcept {
    return yyjson_mut_obj_add_arr(m_doc, obj, key);
  }
  [[nodiscard]] yyjson_mut_val* add_object(yyjson_mut_val* obj, const char* key) const noexcept {
    return yyjson_mut_obj_add_obj(m_doc, obj, key);
  }
  [[nodiscard]] yyjson_mut_val* append_object(yyjson_mut_val* array) const noexcept {
    return yyjson_mut_arr_add_obj(m_doc, array);
  }
  void append(yyjson_mut_val* array, std::string_view value) const noexcept {
    yyjson_mut_arr_add_strncpy(m_doc, array, value.data(), value.size());
  }

private:
  yyjson_mut_doc* m_doc;
};

std::optional<uint64_t> widen(std::optional<uint32_t> value) noexcept {
  return value ? std::optional<uint64_t>{*value} : std::nullopt;
}

void write_pass(const Writer& w, yyjson_mut_val* passes, const InspectPass& pass, size_t index) {
  yyjson_mut_val* obj = w.append_object(passes);
  w.add(obj, "index", uint64_t{index});
  w.add(obj, "name", std::string_view{pass.name});
  w.add(obj, "stage", std::string_view{pass.stage});
  w.add(obj, "node", std::string_view{pass.nodeKind});
  w.add(obj, "kind", pass.kind);
  w.add(obj, "class", pass.cls);
  w.add(obj, "order", widen(pass.order));
  w.add(obj, "registration", widen(pass.registration));
  w.add_bool(obj, "active", pass.active);
  w.add(obj, "cull", pass.cull);
  w.add(obj, "detail", pass.detail);
  w.add_bool(obj, "sideEffect", pass.sideEffect);
  w.add_bool(obj, "external", pass.external);
  yyjson_mut_val* reads = w.add_array(obj, "reads");
  for (const InspectRead& read : pass.reads) {
    yyjson_mut_val* entry = w.append_object(reads);
    w.add(entry, "resource", std::string_view{read.resource});
    w.add(entry, "access", std::string_view{read.access});
    w.add(entry, "missing", std::string_view{read.missing});
    w.add(entry, "binding", std::string_view{read.binding});
  }
  yyjson_mut_val* writes = w.add_array(obj, "writes");
  for (const InspectWrite& write : pass.writes) {
    yyjson_mut_val* entry = w.append_object(writes);
    w.add(entry, "resource", std::string_view{write.resource});
    w.add(entry, "access", std::string_view{write.access});
  }
}

std::string dot_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out;
}

std::optional<std::string_view> frame_reason(const InspectPass& pass, const FrameCullSnapshot* frame) {
  if (frame == nullptr || !pass.frameSlot) {
    return std::nullopt;
  }
  if (*pass.frameSlot >= frame->reasons.size()) {
    return StageNotRun;
  }
  const std::optional<std::string>& reason = frame->reasons[*pass.frameSlot];
  return reason ? std::optional<std::string_view>{*reason} : std::nullopt;
}

void write_dot_node(std::string& out, const InspectPass& pass, size_t index, const FrameCullSnapshot* frame) {
  std::string label = dot_escape(pass.name);
  bool dashed = false;
  if (!pass.active) {
    label += "\\n" + dot_escape(pass.cull.value_or("inactive"));
    dashed = true;
  } else if (const std::optional<std::string_view> reason = frame_reason(pass, frame)) {
    label += "\\nlast frame: " + dot_escape(*reason);
    dashed = true;
  }
  const bool chain = !pass.kind.has_value();
  out += fmt::format("    n{} [label=\"{}\"{}{}];\n", index, label, chain ? ", shape=ellipse" : "",
                     dashed ? ", style=dashed" : "");
}

void write_resource(const Writer& w, yyjson_mut_val* resources, const InspectResource& resource) {
  yyjson_mut_val* obj = w.append_object(resources);
  w.add(obj, "name", std::string_view{resource.name});
  w.add(obj, "kind", std::string_view{resource.kind});
  w.add(obj, "type", std::string_view{resource.type});
  w.add(obj, "format", std::string_view{resource.format});
  w.add(obj, "size", std::string_view{resource.size});
  w.add(obj, "width", uint64_t{resource.width});
  w.add(obj, "height", uint64_t{resource.height});
  w.add(obj, "depthOrLayers", uint64_t{resource.depthOrLayers});
  w.add(obj, "mips", uint64_t{resource.mips});
  w.add(obj, "samples", uint64_t{resource.samples});
  yyjson_mut_val* usage = w.add_array(obj, "usage");
  for (const std::string& flag : resource.usage) {
    w.append(usage, flag);
  }
  w.add(obj, "bytes", resource.bytes);
  w.add(obj, "slot", widen(resource.slot));
  w.add(obj, "firstPass", widen(resource.firstPass));
  w.add(obj, "lastPass", widen(resource.lastPass));
}
} // namespace

std::string format_graph_json(const GraphInspection& graph, const FrameCullSnapshot* frame,
                              const webgpu::gpu_prof::FrameTimes* timings) {
  const json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  if (!doc) {
    return {};
  }
  const Writer w{doc.get()};
  yyjson_mut_val* root = w.object();
  yyjson_mut_doc_set_root(doc.get(), root);
  w.add(root, "schema", std::string_view{"aurora.render.graph"});
  w.add(root, "version", uint64_t{1});
  w.add_bool(root, "compiled", true);
  w.add(root, "graphHash", std::string_view{fmt::format("0x{:016x}", graph.graphHash)});
  w.add(root, "sceneDepth", std::string_view{graph.sceneDepth});

  yyjson_mut_val* stages = w.add_array(root, "stages");
  for (const InspectStage& stage : graph.stages) {
    yyjson_mut_val* obj = w.append_object(stages);
    w.add(obj, "name", std::string_view{stage.name});
    w.add(obj, "firstPass", uint64_t{stage.firstPass});
    w.add(obj, "passCount", uint64_t{stage.passCount});
  }

  yyjson_mut_val* passes = w.add_array(root, "passes");
  for (size_t i = 0; i < graph.passes.size(); ++i) {
    write_pass(w, passes, graph.passes[i], i);
  }

  yyjson_mut_val* resources = w.add_array(root, "resources");
  for (const InspectResource& resource : graph.resources) {
    write_resource(w, resources, resource);
  }

  yyjson_mut_val* edges = w.add_array(root, "edges");
  for (const InspectEdge& edge : graph.edges) {
    yyjson_mut_val* obj = w.append_object(edges);
    w.add(obj, "from", uint64_t{edge.from});
    w.add(obj, "to", uint64_t{edge.to});
    w.add(obj, "kind", std::string_view{edge.kind});
    w.add(obj, "resource", edge.resource);
  }

  yyjson_mut_val* pool = w.add_object(root, "pool");
  w.add(pool, "slots", uint64_t{graph.poolSlots});
  w.add(pool, "logicalBytes", graph.logicalBytes);
  w.add(pool, "slotBytes", graph.slotBytes);
  w.add(pool, "claimedBytes", graph.claimedBytes);
  w.add(pool, "idleBytes", graph.idleBytes);

  if (frame != nullptr) {
    yyjson_mut_val* obj = w.add_object(root, "frame");
    w.add(obj, "frameId", frame->frameId);
    yyjson_mut_val* framePasses = w.add_array(obj, "passes");
    for (size_t i = 0; i < graph.passes.size(); ++i) {
      const std::optional<uint32_t> slot = graph.passes[i].frameSlot;
      if (!slot) {
        continue;
      }
      yyjson_mut_val* entry = w.append_object(framePasses);
      w.add(entry, "index", uint64_t{i});
      if (*slot < frame->reasons.size()) {
        w.add(entry, "reason", frame->reasons[*slot]);
      } else {
        w.add(entry, "reason", StageNotRun);
      }
    }
  }

  if (timings != nullptr) {
    yyjson_mut_val* obj = w.add_object(root, "timings");
    w.add(obj, "sequence", timings->sequence);
    w.add_ms(obj, "frameMs", timings->frameMs);
    w.add(obj, "dropped", uint64_t{timings->dropped});
    yyjson_mut_val* timed = w.add_array(obj, "passes");
    for (const webgpu::gpu_prof::PassTime& pass : timings->passes) {
      yyjson_mut_val* entry = w.append_object(timed);
      w.add(entry, "name", std::string_view{pass.name != nullptr ? pass.name : ""});
      w.add_ms(entry, "ms", pass.ms);
    }
  }
  return json::write(doc.get(), true);
}

std::string format_graph_dot(const GraphInspection& graph, const FrameCullSnapshot* frame) {
  std::string out = fmt::format("digraph render_graph {{\n  label=\"0x{:016x}\";\n  rankdir=LR;\n  node [shape=box];\n",
                                graph.graphHash);
  std::vector<bool> written(graph.passes.size(), false);
  for (size_t s = 0; s < graph.stages.size(); ++s) {
    const InspectStage& stage = graph.stages[s];
    out += fmt::format("  subgraph cluster_{} {{\n    label=\"{}\";\n", s, dot_escape(stage.name));
    for (size_t i = 0; i < graph.passes.size(); ++i) {
      if (!written[i] && graph.passes[i].stage == stage.name) {
        write_dot_node(out, graph.passes[i], i, frame);
        written[i] = true;
      }
    }
    out += "  }\n";
  }
  for (size_t i = 0; i < graph.passes.size(); ++i) {
    if (!written[i]) {
      write_dot_node(out, graph.passes[i], i, frame);
    }
  }
  for (const InspectEdge& edge : graph.edges) {
    std::string attributes;
    if (edge.kind == "Anchor" || edge.kind == "Bound") {
      attributes = " [style=dotted]";
    } else if (edge.kind == "Raw" || edge.kind == "War" || edge.kind == "Waw") {
      const std::string resource = dot_escape(edge.resource.value_or(std::string{}));
      attributes = edge.kind == "Raw" ? fmt::format(" [label=\"{}\"]", resource)
                                      : fmt::format(" [label=\"{} {}\"]", edge.kind, resource);
    }
    out += fmt::format("  n{} -> n{}{};\n", edge.from, edge.to, attributes);
  }
  out += "}\n";
  return out;
}
} // namespace aurora::render::debug
