include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/AuroraCopyRuntimeDLLs.cmake)

add_executable(render_demo
  bench_report.cpp
  bench_report.hpp
  cull_pass.cpp
  cull_pass.hpp
  cull_reference.cpp
  cull_reference.hpp
  graph_checks.cpp
  graph_checks.hpp
  graph_passes.cpp
  graph_passes.hpp
  hint_checks.cpp
  hint_checks.hpp
  hints.cpp
  hints.hpp
  indirect_bench.cpp
  indirect_bench.hpp
  main.cpp
  options.cpp
  options.hpp
  procedural.cpp
  procedural.hpp
  scene.cpp
  scene.hpp
)
target_link_libraries(render_demo PRIVATE aurora::core aurora::gx aurora::main aurora::vi aurora::mtx)
set_target_properties(render_demo PROPERTIES FOLDER "aurora/examples")
aurora_copy_runtime_dlls(render_demo)
