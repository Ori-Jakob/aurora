include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/AuroraCopyRuntimeDLLs.cmake)

add_executable(aurora_gxreplay
  main.cpp
  options.cpp options.hpp
  readback.cpp readback.hpp
  replayer.cpp replayer.hpp
  report.cpp report.hpp
  self_test.cpp self_test.hpp
)
target_include_directories(aurora_gxreplay PRIVATE ../../lib ../../lib/dolphin/gx)
target_compile_definitions(aurora_gxreplay PRIVATE AURORA_ENABLE_RENDER=$<BOOL:${AURORA_ENABLE_RENDER}>)
target_link_libraries(aurora_gxreplay PRIVATE aurora::core aurora::gx aurora::vi aurora::main absl::flat_hash_map
  absl::flat_hash_set)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(aurora_gxreplay PRIVATE $<LINK_GROUP:RESCAN,aurora::gx,aurora::vi,aurora::core>)
endif ()
set_target_properties(aurora_gxreplay PROPERTIES FOLDER "aurora/tools")
aurora_copy_runtime_dlls(aurora_gxreplay)
