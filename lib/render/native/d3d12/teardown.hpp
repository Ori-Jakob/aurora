#pragma once

namespace aurora::render::detail::native::d3d12 {
using Teardown = void (*)() noexcept;

void register_teardown(Teardown teardown) noexcept;
void run_teardowns() noexcept;
} // namespace aurora::render::detail::native::d3d12
