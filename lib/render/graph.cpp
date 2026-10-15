#include "graph.hpp"

namespace aurora::render::detail {
std::shared_ptr<const CompiledGraph> compile_graph(const Settings&, const Caps&) { return empty_graph(); }
} // namespace aurora::render::detail
