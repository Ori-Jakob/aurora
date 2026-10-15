#pragma once

#include <aurora/render.hpp>

#include <cstdint>
#include <memory>

namespace aurora::render::detail {
struct CompiledGraph {
  uint64_t graphHash = 0;
};

inline const std::shared_ptr<const CompiledGraph>& empty_graph() noexcept {
  static const std::shared_ptr<const CompiledGraph> graph = std::make_shared<const CompiledGraph>();
  return graph;
}

std::shared_ptr<const CompiledGraph> compile_graph(const Settings& settings, const Caps& caps);
} // namespace aurora::render::detail
