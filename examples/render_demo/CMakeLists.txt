include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/AuroraCopyRuntimeDLLs.cmake)

add_executable(render_demo
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
