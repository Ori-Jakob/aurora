include(FetchContent)
include(GoogleTest)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/AuroraCopyRuntimeDLLs.cmake)

set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_compile_options(/Zc:__cplusplus)
endif ()

if (NOT TARGET gtest)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(googletest)
endif ()

add_executable(io_tests
  io_test.cpp
  ../lib/card/FileIO.cpp
  ../lib/io.cpp
)
target_include_directories(io_tests PRIVATE ../lib)
target_link_libraries(io_tests PRIVATE gtest gtest_main ${AURORA_SDL3_TARGET})
aurora_copy_runtime_dlls(io_tests)
gtest_discover_tests(io_tests)

add_executable(input_tests
  input_binding_test.cpp
  input_router_test.cpp
  ../lib/input/binding.cpp
  ../lib/input/capture.cpp
  ../lib/input/router.cpp
  ../lib/input/source_state.cpp
)
target_include_directories(input_tests PRIVATE
  ../include
  ../lib
)
target_compile_definitions(input_tests PRIVATE AURORA TARGET_PC)
target_link_libraries(input_tests PRIVATE gtest gtest_main ${AURORA_SDL3_TARGET})
aurora_copy_runtime_dlls(input_tests)
gtest_discover_tests(input_tests)

add_executable(pad_adapter_tests
  pad_adapter_test.cpp
)
target_include_directories(pad_adapter_tests PRIVATE
  ../include
  ../lib
)
target_link_libraries(pad_adapter_tests PRIVATE aurora::pad gtest gtest_main)
if (AURORA_ENABLE_GX)
  target_link_libraries(pad_adapter_tests PRIVATE aurora::gx aurora::vi)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(pad_adapter_tests PRIVATE $<LINK_GROUP:RESCAN,aurora::gx,aurora::vi,aurora::core>)
  endif ()
endif ()
aurora_copy_runtime_dlls(pad_adapter_tests)
gtest_discover_tests(pad_adapter_tests)

add_executable(time_tests
  time_test.cpp
  ../lib/time.cpp
)
target_include_directories(time_tests PRIVATE
  ../include
  ../lib
)
target_link_libraries(time_tests PRIVATE gtest gtest_main)
gtest_discover_tests(time_tests)

add_executable(os_time_tests
  os_time_test.cpp
  os_time_test_stubs.cpp
  ../lib/time.cpp
  ../lib/dolphin/os/OSTime.cpp
)
target_include_directories(os_time_tests PRIVATE
  ../include
  ../lib
)
target_compile_definitions(os_time_tests PRIVATE AURORA TARGET_PC=1)
target_link_libraries(os_time_tests PRIVATE fmt::fmt gtest gtest_main)
gtest_discover_tests(os_time_tests)

if (AURORA_ENABLE_GX)
  # GX FIFO round-trip tests
  # Compile GX sources directly to avoid pulling in the full renderer/WebGPU runtime
  add_executable(gx_fifo_tests
    gx_fifo_test.cpp
    gx_gpu_channel_test.cpp
    gx_hints_test.cpp
    gx_draw_route_test.cpp
    gx_test_stubs.cpp
    # GX API implementations (encoders)
    ../lib/dolphin/gx/GXBump.cpp
    ../lib/dolphin/gx/GXCull.cpp
    ../lib/dolphin/gx/GXCpu2Efb.cpp
    ../lib/dolphin/gx/GXDispList.cpp
    ../lib/dolphin/gx/GXDraw.cpp
    ../lib/dolphin/gx/GXExtra.cpp
    ../lib/dolphin/gx/GXFifo.cpp
    ../lib/dolphin/gx/GXFrameBuffer.cpp
    ../lib/dolphin/gx/GXGeometry.cpp
    ../lib/dolphin/gx/GXGet.cpp
    ../lib/dolphin/gx/GXLighting.cpp
    ../lib/dolphin/gx/GXManage.cpp
    ../lib/dolphin/gx/GXAurora.cpp
    ../lib/dolphin/gx/GXPerf.cpp
    ../lib/dolphin/gx/GXPixel.cpp
    ../lib/dolphin/gx/GXTev.cpp
    ../lib/dolphin/gx/GXTexture.cpp
    ../lib/dolphin/gx/GXTransform.cpp
    ../lib/dolphin/gx/GXVert.cpp
    # FIFO/command processor (encoder + decoder)
    ../lib/gx/fifo.cpp
    ../lib/gx/command_processor.cpp
    ../lib/gx/gpu_channel.cpp
    ../lib/gx/hints.cpp
    ../lib/gx/regs.cpp
    ../lib/thread.cpp
    # Display list reader/optimizer
    ../lib/gx/attr_fmt.cpp
    ../lib/gx/dl.cpp
    gx_dl_test.cpp
    # Trace capture
    gx_trace_test.cpp
    ../lib/io.cpp
    ../lib/trace/address_map.cpp
    ../lib/trace/capture_session.cpp
    ../lib/trace/snapshot_store.cpp
    ../lib/trace/state_preamble.cpp
    ../lib/trace/trace_format.cpp
    ../lib/trace/trace_reader.cpp
    ../lib/trace/trace_writer.cpp
    ../tools/gxreplay/self_test.cpp
  )

  target_include_directories(gx_fifo_tests PRIVATE
    ../include
    ../lib
    ../lib/dolphin/gx
  )

  target_compile_definitions(gx_fifo_tests PRIVATE AURORA TARGET_PC=1)

  target_link_libraries(gx_fifo_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    xxHash::xxhash
    absl::flat_hash_map
    absl::btree
    dawn::dawncpp_headers
    Tracy::TracyClient
    ${AURORA_SDL3_TARGET}
  )
  if (TARGET zstd::libzstd)
    target_compile_definitions(gx_fifo_tests PRIVATE AURORA_TRACE_ZSTD)
    target_link_libraries(gx_fifo_tests PRIVATE zstd::libzstd)
  endif ()

  gtest_discover_tests(gx_fifo_tests PROPERTIES TIMEOUT 60)

  add_executable(gx_texture_cache_tests
    gx_texture_cache_test.cpp
    gx_texture_cache_test_stubs.cpp
    ../lib/gx/texture.cpp
  )
  target_include_directories(gx_texture_cache_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(gx_texture_cache_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(gx_texture_cache_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    xxHash::xxhash
    absl::flat_hash_map
    dawn::webgpu_dawn
    Tracy::TracyClient
    ${AURORA_SDL3_TARGET}
  )
  aurora_copy_runtime_dlls(gx_texture_cache_tests)
  gtest_discover_tests(gx_texture_cache_tests)

  add_executable(lru_cache_tests lru_cache_test.cpp)
  target_include_directories(lru_cache_tests PRIVATE ../lib)
  target_link_libraries(lru_cache_tests PRIVATE gtest gtest_main absl::flat_hash_map)
  gtest_discover_tests(lru_cache_tests)

  add_executable(texture_replacement_streaming_tests
    texture_replacement_streaming_test.cpp
  )
  target_include_directories(texture_replacement_streaming_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(texture_replacement_streaming_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(texture_replacement_streaming_tests PRIVATE
    aurora::gx
    aurora::vi
    gtest
    gtest_main
  )
  aurora_copy_runtime_dlls(texture_replacement_streaming_tests)
  gtest_discover_tests(texture_replacement_streaming_tests)

  add_executable(material_contract_tests
    material_contract_test.cpp
    material_contract_check.cpp
  )
  target_include_directories(material_contract_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(material_contract_tests PRIVATE AURORA TARGET_PC=1
    AURORA_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  target_link_libraries(material_contract_tests PRIVATE
    aurora::gx
    aurora::vi
    gtest
    gtest_main
  )
  if (AURORA_ENABLE_RENDER)
    target_sources(material_contract_tests PRIVATE material_registry_test.cpp material_user_layer_test.cpp)
    target_link_libraries(material_contract_tests PRIVATE yyjson::yyjson)
  endif ()
  aurora_copy_runtime_dlls(material_contract_tests)
  gtest_discover_tests(material_contract_tests)

  add_executable(render_worker_tests
    render_worker_test.cpp
    thread_test.cpp
    ../lib/gfx/render_worker.cpp
    ../lib/thread.cpp
  )
  target_include_directories(render_worker_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(render_worker_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(render_worker_tests PRIVATE
    gtest
    gtest_main
    Tracy::TracyClient
    ${AURORA_SDL3_TARGET}
  )
  gtest_discover_tests(render_worker_tests)

  add_executable(gfx_recording_tests
    gfx_recording_test.cpp
    gfx_streaming_sizes_test.cpp
    render_target_layout_test.cpp
    trace_capture_test.cpp
  )
  target_include_directories(gfx_recording_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(gfx_recording_tests PRIVATE AURORA TARGET_PC=1
    AURORA_ENABLE_RENDER=$<BOOL:${AURORA_ENABLE_RENDER}>)
  target_link_libraries(gfx_recording_tests PRIVATE
    aurora::gx
    aurora::vi
    absl::flat_hash_map
    gtest
    gtest_main
  )
  aurora_copy_runtime_dlls(gfx_recording_tests)
  gtest_discover_tests(gfx_recording_tests)

  add_executable(render_tests
    render_api_test.cpp
    render_hints_test.cpp
    render_settings_test.cpp
  )
  target_include_directories(render_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(render_tests PRIVATE AURORA TARGET_PC=1
    AURORA_ENABLE_RENDER=$<BOOL:${AURORA_ENABLE_RENDER}>)
  target_link_libraries(render_tests PRIVATE
    aurora::gx
    aurora::vi
    gtest
    gtest_main
  )
  if (AURORA_ENABLE_RENDER)
    target_sources(render_tests PRIVATE render_json_test.cpp)
    target_link_libraries(render_tests PRIVATE yyjson::yyjson)
    target_sources(render_tests PRIVATE render_classify_glue_test.cpp)
    target_link_libraries(render_tests PRIVATE absl::flat_hash_map)
    target_sources(render_tests PRIVATE render_views_test.cpp)
    target_sources(render_tests PRIVATE render_gx_ext_test.cpp)
    target_sources(render_tests PRIVATE render_motion_test.cpp)
    target_sources(render_tests PRIVATE render_texture_inspector_test.cpp)
    target_sources(render_tests PRIVATE render_fake_shadow_test.cpp)
    target_sources(render_tests PRIVATE render_fake_shadows_tab_test.cpp)
  endif ()
  aurora_copy_runtime_dlls(render_tests)
  gtest_discover_tests(render_tests)

  add_library(gx_shader_corpus_lib STATIC
    gx_shader_corpus.cpp
  )
  target_include_directories(gx_shader_corpus_lib PUBLIC
    ../include
    ../lib
  )
  target_compile_definitions(gx_shader_corpus_lib PUBLIC AURORA TARGET_PC=1)
  target_link_libraries(gx_shader_corpus_lib PUBLIC
    aurora::gx
    aurora::vi
    absl::flat_hash_map
  )

  add_executable(gx_shader_corpus gx_shader_corpus_tool.cpp)
  target_link_libraries(gx_shader_corpus PRIVATE gx_shader_corpus_lib)
  aurora_copy_runtime_dlls(gx_shader_corpus)

  add_executable(gx_shader_golden_tests gx_shader_golden_test.cpp gx_shader_ext_test.cpp)
  target_compile_definitions(gx_shader_golden_tests PRIVATE
    AURORA_GX_SHADER_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shaders/gx_shader_golden.txt"
    AURORA_GX_SHADER_GOLDEN_OUT="${CMAKE_CURRENT_BINARY_DIR}/gx_shader_golden_out"
  )
  target_link_libraries(gx_shader_golden_tests PRIVATE gx_shader_corpus_lib gtest gtest_main)
  aurora_copy_runtime_dlls(gx_shader_golden_tests)
  gtest_discover_tests(gx_shader_golden_tests)

  add_executable(feature_profile_tests
    feature_profile_test.cpp
    ../lib/webgpu/feature_profile.cpp
  )
  target_include_directories(feature_profile_tests PRIVATE ../include ../lib)
  target_compile_definitions(feature_profile_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(feature_profile_tests PRIVATE gtest gtest_main dawn::dawncpp_headers)
  gtest_discover_tests(feature_profile_tests)

  add_executable(caps_tests
    caps_test.cpp
    ../lib/webgpu/caps.cpp
    ../lib/webgpu/feature_profile.cpp
  )
  target_include_directories(caps_tests PRIVATE ../include ../lib)
  target_compile_definitions(caps_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(caps_tests PRIVATE gtest gtest_main dawn::dawncpp_headers)
  gtest_discover_tests(caps_tests)

  if (AURORA_ENABLE_RENDER)
    add_executable(render_graph_tests
      render_graph_test.cpp
      render_graph_planned.cpp
      render_graph_slots_test.cpp
      render_graph_layout_test.cpp
      render_graph_dump_test.cpp
      render_graph_api_test.cpp
      scene_depth_test.cpp
      os_test_globals.cpp
      ../lib/logging.cpp
      ../lib/render/graph.cpp
      ../lib/render/graph_api.cpp
      ../lib/render/graph_layout.cpp
      ../lib/render/graph_slots.cpp
      ../lib/render/debug/graph_dump.cpp
    )
    target_include_directories(render_graph_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_graph_tests PRIVATE AURORA TARGET_PC=1
      AURORA_RENDER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/render"
      AURORA_RENDER_FIXTURE_OUT="${CMAKE_CURRENT_BINARY_DIR}/render_fixture_out")
    target_link_libraries(render_graph_tests PRIVATE gtest gtest_main fmt::fmt xxHash::xxhash dawn::dawncpp_headers
      yyjson::yyjson)
    gtest_discover_tests(render_graph_tests)
  endif ()

  if (AURORA_ENABLE_RENDER)
    add_executable(render_tlsf_tests render_tlsf_test.cpp ../lib/render/tlsf.cpp)
    target_include_directories(render_tlsf_tests PRIVATE ../lib)
    target_link_libraries(render_tlsf_tests PRIVATE gtest gtest_main)
    gtest_discover_tests(render_tlsf_tests)

    add_executable(render_classify_tests render_classify_test.cpp ../lib/render/scene/classifier.cpp)
    target_include_directories(render_classify_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_classify_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1)
    target_link_libraries(render_classify_tests PRIVATE gtest gtest_main)
    gtest_discover_tests(render_classify_tests)

    add_executable(render_gpu_tests render_test_device.cpp render_gpu_heap_test.cpp render_graph_pool_test.cpp
      render_graph_bind_test.cpp render_graph_planned.cpp render_graph_api_gpu_test.cpp render_compact_test.cpp
      render_hzb_test.cpp render_hzb_reference.cpp render_gpu_systems_test.cpp)
    target_include_directories(render_gpu_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_gpu_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1)
    target_link_libraries(render_gpu_tests PRIVATE aurora::gx aurora::vi gtest gtest_main)
    aurora_copy_runtime_dlls(render_gpu_tests)
    gtest_discover_tests(render_gpu_tests PROPERTIES TIMEOUT 120)
  endif ()

  add_executable(trace_tests
    trace_format_test.cpp
    trace_store_test.cpp
    os_test_globals.cpp
    ../lib/io.cpp
    ../lib/logging.cpp
    ../lib/trace/address_map.cpp
    ../lib/trace/capture_session.cpp
    ../lib/trace/snapshot_store.cpp
    ../lib/trace/trace_format.cpp
    ../lib/trace/trace_reader.cpp
    ../lib/trace/trace_writer.cpp
  )
  target_include_directories(trace_tests PRIVATE ../include ../lib)
  target_compile_definitions(trace_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(trace_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    xxHash::xxhash
    absl::flat_hash_map
    ${AURORA_SDL3_TARGET}
  )
  if (TARGET zstd::libzstd)
    target_compile_definitions(trace_tests PRIVATE AURORA_TRACE_ZSTD)
    target_link_libraries(trace_tests PRIVATE zstd::libzstd)
  endif ()
  aurora_copy_runtime_dlls(trace_tests)
  gtest_discover_tests(trace_tests)

  if (AURORA_ENABLE_RENDER)
    add_executable(render_jobs_tests render_jobs_test.cpp ../lib/render/jobs.cpp ../lib/thread.cpp)
    target_include_directories(render_jobs_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_jobs_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1)
    target_link_libraries(render_jobs_tests PRIVATE gtest gtest_main Tracy::TracyClient ${AURORA_SDL3_TARGET})
    gtest_discover_tests(render_jobs_tests PROPERTIES TIMEOUT 60)

    file(GLOB_RECURSE _render_test_wgsl CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shaders/render/*.wgsl")
    add_executable(render_shaders_tests render_shaders_test.cpp render_test_device.cpp gx_shader_ext_device_test.cpp)
    target_include_directories(render_shaders_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_shaders_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1)
    target_link_libraries(render_shaders_tests PRIVATE aurora::gx aurora::vi absl::flat_hash_map gx_shader_corpus_lib gtest
      gtest_main)
    aurora_embed_wgsl(render_shaders_tests
      ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shaders/render"
      TABLE embedded_test_shaders
      HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../lib/render/shaders_embed.hpp"
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/render_test_wgsl.cpp"
      SOURCES ${_render_test_wgsl})
    aurora_copy_runtime_dlls(render_shaders_tests)
    gtest_discover_tests(render_shaders_tests PROPERTIES TIMEOUT 120)

    add_executable(render_debug_tests render_debug_test.cpp render_test_device.cpp)
    target_include_directories(render_debug_tests PRIVATE ../include ../lib)
    target_compile_definitions(render_debug_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1
      AURORA_RENDER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/render"
      AURORA_RENDER_FIXTURE_OUT="${CMAKE_CURRENT_BINARY_DIR}/render_fixture_out")
    # Linked for its PUBLIC TRACY_ENABLE, which decides whether timings collect by default
    target_link_libraries(render_debug_tests PRIVATE aurora::gx aurora::vi yyjson::yyjson Tracy::TracyClient gtest
      gtest_main)
    aurora_copy_runtime_dlls(render_debug_tests)
    gtest_discover_tests(render_debug_tests)
  endif ()
endif () # AURORA_ENABLE_GX

if (TARGET render_demo)
  add_executable(render_demo_tests
    render_demo_test.cpp
    ../examples/render_demo/bench_report.cpp
    ../examples/render_demo/cull_reference.cpp
    ../examples/render_demo/graph_checks.cpp
    ../examples/render_demo/hint_checks.cpp
    ../examples/render_demo/options.cpp
    ../examples/render_demo/procedural.cpp
  )
  target_include_directories(render_demo_tests PRIVATE ../include ../examples/render_demo)
  target_compile_definitions(render_demo_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(render_demo_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(render_demo_tests)

  add_test(NAME render_demo_usage COMMAND $<TARGET_FILE:render_demo> --help)
  add_test(NAME render_demo_rejects_bad_args COMMAND $<TARGET_FILE:render_demo> --frames abc)
  set_tests_properties(render_demo_rejects_bad_args PROPERTIES WILL_FAIL TRUE)
  foreach (_profile IN ITEMS full compat metal metal32)
    add_test(NAME render_demo_smoke_${_profile}
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile} --frames 120 --timeout 60
              --fail-on-error --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_smoke/${_profile})
    set_tests_properties(render_demo_smoke_${_profile} PROPERTIES LABELS smoke TIMEOUT 180)
  endforeach ()
  add_test(NAME render_demo_smoke_streaming_growth
    COMMAND $<TARGET_FILE:render_demo> --backend null --profile compat --frames 90 --timeout 60
            --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_smoke/streaming)
  set_tests_properties(render_demo_smoke_streaming_growth PROPERTIES LABELS smoke TIMEOUT 180
    ENVIRONMENT "AURORA_TEST_STREAMING_GROWTH=30"
    PASS_REGULAR_EXPRESSION "Streaming buffers resized: .*storage 16 MiB"
    FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
  add_test(NAME render_demo_custom_draw_probe_compat
    COMMAND $<TARGET_FILE:render_demo> --backend null --profile compat --frames 30 --timeout 60 --fail-on-error
            --custom-draw-probe --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_smoke/custom_draw_probe)
  set_tests_properties(render_demo_custom_draw_probe_compat PROPERTIES LABELS smoke TIMEOUT 180)

  if (AURORA_ENABLE_RENDER)
    foreach (_profile IN ITEMS full compat metal metal32)
      add_test(NAME render_demo_vignette_${_profile}
        COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile} --frames 120 --timeout 60
                --fail-on-error --vignette --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_vignette/${_profile})
      set_tests_properties(render_demo_vignette_${_profile} PROPERTIES LABELS smoke TIMEOUT 180)
    endforeach ()
    add_test(NAME render_demo_vignette_msaa
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile metal --msaa 4 --frames 60 --timeout 60
              --fail-on-error --vignette --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_vignette/msaa)
    set_tests_properties(render_demo_vignette_msaa PROPERTIES LABELS smoke TIMEOUT 180)
    add_test(NAME render_demo_vignette_unknown_anchor
      COMMAND $<TARGET_FILE:render_demo> --backend null --frames 30 --timeout 60 --fail-on-error
              --vignette-anchor phase.not_a_phase --expect-anchor-warnings 1
              --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_vignette/unknown_anchor)
    set_tests_properties(render_demo_vignette_unknown_anchor PROPERTIES LABELS smoke TIMEOUT 180)
    add_test(NAME render_demo_rejects_vignette_with_trace
      COMMAND $<TARGET_FILE:render_demo> --vignette --trace ${CMAKE_CURRENT_BINARY_DIR}/never.gxtrace)
    set_tests_properties(render_demo_rejects_vignette_with_trace PROPERTIES WILL_FAIL TRUE)
    foreach (_profile IN ITEMS full compat metal metal32)
      add_test(NAME render_demo_depth_probe_${_profile}
        COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile} --frames 60 --timeout 60
                --fail-on-error --depth-check --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_depth/${_profile})
      set_tests_properties(render_demo_depth_probe_${_profile} PROPERTIES LABELS smoke TIMEOUT 180)
    endforeach ()
    add_test(NAME render_demo_depth_probe_msaa
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile metal --msaa 4 --frames 60 --timeout 60
              --fail-on-error --depth-check --vignette --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_depth/msaa)
    set_tests_properties(render_demo_depth_probe_msaa PROPERTIES LABELS smoke TIMEOUT 180)

    add_executable(render_demo_gpu_tests render_demo_gpu_test.cpp render_test_device.cpp
      ../examples/render_demo/cull_reference.cpp)
    target_include_directories(render_demo_gpu_tests PRIVATE ../include ../lib ../examples/render_demo)
    target_compile_definitions(render_demo_gpu_tests PRIVATE AURORA TARGET_PC=1 AURORA_ENABLE_RENDER=1)
    target_link_libraries(render_demo_gpu_tests PRIVATE aurora::gx aurora::vi gtest gtest_main)
    aurora_copy_runtime_dlls(render_demo_gpu_tests)
    gtest_discover_tests(render_demo_gpu_tests PROPERTIES TIMEOUT 120)

    # No PASS_REGULAR_EXPRESSION, so the exit code still counts
    set(_cull_args --gpu-cull 1048576 --cull-check --frames 60 --timeout 60 --fail-on-error)
    foreach (_profile IN ITEMS full compat metal metal32)
      add_test(NAME render_demo_gpu_cull_${_profile}
        COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile} ${_cull_args}
                --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_cull/${_profile})
      set_tests_properties(render_demo_gpu_cull_${_profile} PROPERTIES LABELS smoke TIMEOUT 180
        FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    endforeach ()
    add_test(NAME render_demo_gpu_cull_msaa
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile metal --msaa 4 ${_cull_args}
              --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_cull/msaa)
    add_test(NAME render_demo_gpu_cull_no_hints
      COMMAND $<TARGET_FILE:render_demo> --backend null --no-hints ${_cull_args}
              --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_cull/no_hints)
    add_test(NAME render_demo_gpu_cull_with_probe
      COMMAND $<TARGET_FILE:render_demo> --backend null --gpu-cull 4096 --vignette --depth-check --frames 60
              --timeout 60 --fail-on-error --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_cull/with_probe)
    add_test(NAME render_demo_gpu_cull_timings
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile full --gpu-cull 1048576 --cull-check --cull-timings
              --frames 90 --timeout 60 --fail-on-error --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_cull/timings)
    set_tests_properties(render_demo_gpu_cull_msaa render_demo_gpu_cull_no_hints render_demo_gpu_cull_with_probe
      render_demo_gpu_cull_timings PROPERTIES LABELS smoke TIMEOUT 180 FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    add_test(NAME render_demo_rejects_gpu_cull_with_trace
      COMMAND $<TARGET_FILE:render_demo> --gpu-cull 4096 --trace ${CMAKE_CURRENT_BINARY_DIR}/never.gxtrace)
    set_tests_properties(render_demo_rejects_gpu_cull_with_trace PROPERTIES WILL_FAIL TRUE)
  endif ()

  # The bench uses core WebGPU only, so it runs with the render module off too
  foreach (_profile IN ITEMS full compat metal metal32)
    add_test(NAME render_demo_bench_${_profile}
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile}
              --bench-indirect ${CMAKE_CURRENT_BINARY_DIR}/render_demo_bench/${_profile}.json --bench-blocks 1
              --fail-on-error --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_bench/data_${_profile})
    set_tests_properties(render_demo_bench_${_profile} PROPERTIES LABELS smoke TIMEOUT 180
      FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): |errored [1-9]")
  endforeach ()
  add_test(NAME render_demo_rejects_bench_with_frames
    COMMAND $<TARGET_FILE:render_demo> --bench-indirect ${CMAKE_CURRENT_BINARY_DIR}/never.json --frames 10)
  set_tests_properties(render_demo_rejects_bench_with_frames PROPERTIES WILL_FAIL TRUE)

  foreach (_profile IN ITEMS full compat metal metal32)
    add_test(NAME render_demo_hints_${_profile}
      COMMAND $<TARGET_FILE:render_demo> --backend null --profile ${_profile} --frames 120 --timeout 60
              --fail-on-error --check-hints --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_hints/${_profile})
    set_tests_properties(render_demo_hints_${_profile} PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1;AURORA_IMMUTABLE_ARRAYS=validate")
  endforeach ()
  add_test(NAME render_demo_hints_disabled
    COMMAND $<TARGET_FILE:render_demo> --backend null --frames 120 --timeout 60 --fail-on-error --check-hints
            --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_hints/disabled)
  set_tests_properties(render_demo_hints_disabled PROPERTIES LABELS smoke TIMEOUT 180
    ENVIRONMENT "AURORA_RENDER_HINTS=0")
  add_test(NAME render_demo_no_hints
    COMMAND $<TARGET_FILE:render_demo> --backend null --frames 120 --timeout 60 --fail-on-error --no-hints
            --check-hints --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_hints/none)
  set_tests_properties(render_demo_no_hints PROPERTIES LABELS smoke TIMEOUT 180)

  # GPU-system commands decode in core, so these run with the render module off too
  if (AURORA_ENABLE_RENDER)
    set(_gpu_systems "dispatches 120, systems ready 2/2")
    set(_gpu_reserves 4)
  else ()
    set(_gpu_systems "dispatches 0, systems unavailable")
    set(_gpu_reserves 0)
  endif ()
  string(CONCAT _gpu_pass "gpu: uploads 120, binds [1-9][0-9]*, suppressed 0, failed 0, rejected 0, refused 0, "
                          "stale 0, ${_gpu_systems}")
  set(_gpu_fail "\\[(ERROR|FATAL): |hint check FAILED")
  set(_gpu_args --backend null --frames 120 --timeout 60 --fail-on-error --gpu-arrays)
  foreach (_profile IN ITEMS full compat metal metal32)
    add_test(NAME render_demo_gpu_arrays_${_profile}
      COMMAND $<TARGET_FILE:render_demo> ${_gpu_args} --profile ${_profile}
              --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_gpu_arrays/${_profile})
  endforeach ()
  add_test(NAME render_demo_gpu_arrays_msaa
    COMMAND $<TARGET_FILE:render_demo> ${_gpu_args} --profile metal --msaa 4
            --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_gpu_arrays/msaa)
  add_test(NAME render_demo_gpu_arrays_hints_disabled
    COMMAND $<TARGET_FILE:render_demo> ${_gpu_args} --check-hints
            --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_gpu_arrays/hints_disabled)
  set_tests_properties(render_demo_gpu_arrays_hints_disabled PROPERTIES ENVIRONMENT "AURORA_RENDER_HINTS=0")
  add_test(NAME render_demo_gpu_arrays_no_hints
    COMMAND $<TARGET_FILE:render_demo> ${_gpu_args} --no-hints --check-hints
            --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_gpu_arrays/no_hints)
  set_tests_properties(render_demo_gpu_arrays_full render_demo_gpu_arrays_compat render_demo_gpu_arrays_metal
    render_demo_gpu_arrays_metal32 render_demo_gpu_arrays_msaa render_demo_gpu_arrays_hints_disabled
    render_demo_gpu_arrays_no_hints PROPERTIES LABELS smoke TIMEOUT 180 PASS_REGULAR_EXPRESSION "${_gpu_pass}"
    FAIL_REGULAR_EXPRESSION "${_gpu_fail}")
  if (AURORA_ENABLE_RENDER)
    add_test(NAME render_demo_hints_vignette
      COMMAND $<TARGET_FILE:render_demo> --backend null --frames 120 --timeout 60 --fail-on-error --check-hints
              --vignette --data-dir ${CMAKE_CURRENT_BINARY_DIR}/render_demo_hints/vignette)
    set_tests_properties(render_demo_hints_vignette PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1;AURORA_IMMUTABLE_ARRAYS=validate")
  endif ()
endif ()

if (TARGET aurora_gxreplay)
  add_executable(gxreplay_tests gxreplay_test.cpp ../tools/gxreplay/options.cpp ../tools/gxreplay/report.cpp)
  target_include_directories(gxreplay_tests PRIVATE ../include ../lib ../tools/gxreplay)
  target_compile_definitions(gxreplay_tests PRIVATE AURORA TARGET_PC=1)
  target_link_libraries(gxreplay_tests PRIVATE gtest gtest_main fmt::fmt dawn::dawncpp_headers xxHash::xxhash)
  gtest_discover_tests(gxreplay_tests)

  set(_gxreplay_dir ${CMAKE_CURRENT_BINARY_DIR}/gxreplay)
  add_test(NAME gxreplay_usage COMMAND $<TARGET_FILE:aurora_gxreplay> --help)
  add_test(NAME gxreplay_rejects_bad_args COMMAND $<TARGET_FILE:aurora_gxreplay> --repeat 0 missing.gxtrace)
  set_tests_properties(gxreplay_rejects_bad_args PROPERTIES WILL_FAIL TRUE)
  add_test(NAME gxreplay_self_test_null
    COMMAND $<TARGET_FILE:aurora_gxreplay> --self-test --backend null --out ${_gxreplay_dir}/self_test)
  set_tests_properties(gxreplay_self_test_null PROPERTIES LABELS smoke TIMEOUT 180)
  add_test(NAME gxreplay_self_test_cpu_zones_null
    COMMAND $<TARGET_FILE:aurora_gxreplay> --self-test --backend null --cpu-zones
            --out ${_gxreplay_dir}/self_test_cpu_zones)
  set_tests_properties(gxreplay_self_test_cpu_zones_null PROPERTIES LABELS smoke TIMEOUT 180)
  add_test(NAME gxreplay_self_test_cpu_draw_zones_null
    COMMAND $<TARGET_FILE:aurora_gxreplay> --self-test --backend null --cpu-draw-zones
            --out ${_gxreplay_dir}/self_test_cpu_draw_zones)
  set_tests_properties(gxreplay_self_test_cpu_draw_zones_null PROPERTIES LABELS smoke TIMEOUT 180)
  set(AURORA_TEST_HARDWARE_BACKENDS "" CACHE STRING "Backends that run the hardware self-test, such as d3d12;vulkan")
  foreach (_backend IN LISTS AURORA_TEST_HARDWARE_BACKENDS)
    add_test(NAME gxreplay_self_test_gpu_${_backend}
      COMMAND $<TARGET_FILE:aurora_gxreplay> --self-test --backend ${_backend}
              --out ${_gxreplay_dir}/self_test_gpu_${_backend})
    set_tests_properties(gxreplay_self_test_gpu_${_backend} PROPERTIES LABELS hardware TIMEOUT 180)
  endforeach ()
  if (TARGET render_demo)
    add_test(NAME gxreplay_render_demo_capture
      COMMAND $<TARGET_FILE:render_demo> --backend null --frames 8 --timeout 60
              --data-dir ${_gxreplay_dir}/demo_data --trace ${_gxreplay_dir}/render_demo.gxtrace
              --trace-start 2 --trace-frames 4 --check-hints)
    set_tests_properties(gxreplay_render_demo_capture PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1;AURORA_IMMUTABLE_ARRAYS=validate" FIXTURES_SETUP gxreplay_demo_trace)
    add_test(NAME gxreplay_render_demo_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --repeat 2
              --no-png --out ${_gxreplay_dir}/render_demo_replay)
    set_tests_properties(gxreplay_render_demo_replay PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_trace)
    add_test(NAME gxreplay_render_demo_bind_counts
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --repeat 2
              --no-png --cpu-zones --out ${_gxreplay_dir}/render_demo_binds)
    set_tests_properties(gxreplay_render_demo_bind_counts PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_trace FIXTURES_SETUP gxreplay_demo_binds)
    # Only exact counts catch a skip that compares the wrong field; the self-test check needs just one bind
    add_test(NAME gxreplay_render_demo_bind_counts_check
      COMMAND ${CMAKE_COMMAND} -DZONES=${_gxreplay_dir}/render_demo_binds/cpu_zones.json
              -DCOUNTS=encodedDraws=13,texturedDraws=1,uniformBinds=13,textureBinds=1,indexBinds=1
              -P ${CMAKE_CURRENT_SOURCE_DIR}/check_cpu_zone_counts.cmake)
    set_tests_properties(gxreplay_render_demo_bind_counts_check PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED gxreplay_demo_binds)
    # PASS_REGULAR_EXPRESSION hides the exit code, so the failure messages are matched instead
    set(_resident_fail "\\[(ERROR|FATAL): |violations [1-9]|is not bit-identical|guest reads were not resolved")
    add_test(NAME gxreplay_render_demo_resident
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --repeat 2
              --no-png --out ${_gxreplay_dir}/render_demo_resident)
    set_tests_properties(gxreplay_render_demo_resident PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1;AURORA_IMMUTABLE_ARRAYS=validate" FIXTURES_REQUIRED gxreplay_demo_trace
      PASS_REGULAR_EXPRESSION "resident arrays: run 1, stages [1-9][0-9]*, hits [1-9]"
      FAIL_REGULAR_EXPRESSION "${_resident_fail}")
    # Growth after the first stage recreates the storage buffer, so every array must stage again in a new epoch
    string(CONCAT _resident_growth "resident arrays: run 0, stages 6, hits 3, resets 0, epochs 3, mismatches 0, "
                                   "violations 0, arena full 0")
    add_test(NAME gxreplay_render_demo_resident_growth
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --no-png
              --out ${_gxreplay_dir}/render_demo_resident_growth)
    set_tests_properties(gxreplay_render_demo_resident_growth PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1;AURORA_IMMUTABLE_ARRAYS=validate;AURORA_TEST_STREAMING_GROWTH=2"
      FIXTURES_REQUIRED gxreplay_demo_trace PASS_REGULAR_EXPRESSION "${_resident_growth}"
      FAIL_REGULAR_EXPRESSION "${_resident_fail}")

    set(_gpu_capture --backend null --frames 16 --timeout 60 --trace-start 10 --trace-frames 4)
    add_test(NAME gxreplay_render_demo_gpu_capture
      COMMAND $<TARGET_FILE:render_demo> ${_gpu_capture} --gpu-arrays --fail-on-error
              --data-dir ${_gxreplay_dir}/demo_gpu_data --trace ${_gxreplay_dir}/render_demo_gpu.gxtrace)
    set_tests_properties(gxreplay_render_demo_gpu_capture PROPERTIES FIXTURES_SETUP gxreplay_demo_gpu_trace)
    add_test(NAME gxreplay_render_demo_gpu_default_capture
      COMMAND $<TARGET_FILE:render_demo> ${_gpu_capture} --fail-on-error
              --data-dir ${_gxreplay_dir}/demo_gpu_default_data
              --trace ${_gxreplay_dir}/render_demo_gpu_default.gxtrace)
    set_tests_properties(gxreplay_render_demo_gpu_default_capture PROPERTIES FIXTURES_SETUP gxreplay_demo_gpu_default)
    set_tests_properties(gxreplay_render_demo_gpu_capture gxreplay_render_demo_gpu_default_capture PROPERTIES
      LABELS smoke TIMEOUT 180 FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    string(CONCAT _gpu_replay_fail "\\[(ERROR|FATAL): |suppressed [1-9]|failed [1-9]|misses [1-9]|rejected [1-9]|"
                                   "is not bit-identical|guest reads were not resolved")
    add_test(NAME gxreplay_render_demo_gpu_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_gpu.gxtrace --backend null --repeat 2
              --no-png --out ${_gxreplay_dir}/render_demo_gpu_replay)
    set_tests_properties(gxreplay_render_demo_gpu_replay PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_REQUIRED gxreplay_demo_gpu_trace FIXTURES_SETUP gxreplay_demo_gpu_replay
      PASS_REGULAR_EXPRESSION "gpu: run 1, uploads 4, reserves ${_gpu_reserves}, binds [1-9]"
      FAIL_REGULAR_EXPRESSION "${_gpu_replay_fail}")
    add_test(NAME gxreplay_render_demo_gpu_default_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_gpu_default.gxtrace --backend null --no-png
              --out ${_gxreplay_dir}/render_demo_gpu_default_replay)
    set_tests_properties(gxreplay_render_demo_gpu_default_replay PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_REQUIRED gxreplay_demo_gpu_default FIXTURES_SETUP gxreplay_demo_gpu_default_replay)
    add_test(NAME gxreplay_render_demo_gpu_draws_ab
      COMMAND ${CMAKE_COMMAND} -DA=${_gxreplay_dir}/render_demo_gpu_replay/frames.json
              -DB=${_gxreplay_dir}/render_demo_gpu_default_replay/frames.json -DKEY=draws
              -P ${CMAKE_CURRENT_SOURCE_DIR}/compare_replay_frames.cmake)
    set_tests_properties(gxreplay_render_demo_gpu_draws_ab PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED "gxreplay_demo_gpu_replay;gxreplay_demo_gpu_default_replay")
    # Replay must grow to the storage size live recorded before frame 0; the null device caps memory, so pad little
    add_test(NAME gxreplay_render_demo_gpu_sized_capture
      COMMAND $<TARGET_FILE:render_demo> --backend null --frames 16 --timeout 60 --gpu-pad-mib 6 --trace-start 10
              --trace-frames 3 --data-dir ${_gxreplay_dir}/demo_gpu_sized_data
              --trace ${_gxreplay_dir}/render_demo_gpu_sized.gxtrace)
    set_tests_properties(gxreplay_render_demo_gpu_sized_capture PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_SETUP gxreplay_demo_gpu_sized FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    add_test(NAME gxreplay_render_demo_gpu_sized_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_gpu_sized.gxtrace --backend null --no-png
              --out ${_gxreplay_dir}/render_demo_gpu_sized_replay)
    string(CONCAT _gpu_sized_pass "Streaming buffers resized: [^\n]*storage 16 MiB(.|\n)*"
                                  "gpu: run 0, uploads 3, reserves 0, binds 0, suppressed 0, failed 0, misses 0")
    set_tests_properties(gxreplay_render_demo_gpu_sized_replay PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_REQUIRED gxreplay_demo_gpu_sized PASS_REGULAR_EXPRESSION "${_gpu_sized_pass}"
      FAIL_REGULAR_EXPRESSION "${_gpu_replay_fail}")
  endif ()
  if (AURORA_ENABLE_RENDER)
    set(_classify_trace ${_gxreplay_dir}/classify/self_test_hints.gxtrace)
    add_test(NAME gxreplay_classify_trace COMMAND $<TARGET_FILE:aurora_gxreplay> --write-self-test ${_classify_trace})
    set_tests_properties(gxreplay_classify_trace PROPERTIES LABELS smoke TIMEOUT 60 FIXTURES_SETUP gxreplay_classify)
    add_test(NAME gxreplay_classify_counts
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_classify_trace} --backend null --no-png --classify-objects
              --classify-draws --out ${_gxreplay_dir}/classify/counts)
    set_tests_properties(gxreplay_classify_counts PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_REQUIRED gxreplay_classify
      PASS_REGULAR_EXPRESSION "frames 3, draws 9, main 0, sky 0, sky misses 0, suppressed 0, hint errors 0"
      FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    foreach (_view IN ITEMS role kind main window category object scope bit)
      add_test(NAME gxreplay_debug_view_${_view}
        COMMAND $<TARGET_FILE:aurora_gxreplay> ${_classify_trace} --backend null --no-png --debug-view ${_view}
                --out ${_gxreplay_dir}/classify/view_${_view})
      set_tests_properties(gxreplay_debug_view_${_view} PROPERTIES LABELS smoke TIMEOUT 180
        FIXTURES_REQUIRED gxreplay_classify
        PASS_REGULAR_EXPRESSION "hint errors 0, .*view substituted [1-9]"
        FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
    endforeach ()
  endif ()
  if (TARGET render_demo AND AURORA_ENABLE_RENDER)
    set(_demo_capture --backend null --frames 8 --timeout 60 --trace-start 2 --trace-frames 4 --check-hints)
    add_test(NAME gxreplay_render_demo_plain_capture
      COMMAND $<TARGET_FILE:render_demo> ${_demo_capture} --no-hints --data-dir ${_gxreplay_dir}/demo_plain_data
              --trace ${_gxreplay_dir}/render_demo_plain.gxtrace)
    set_tests_properties(gxreplay_render_demo_plain_capture PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_SETUP gxreplay_demo_plain_trace)
    add_test(NAME gxreplay_render_demo_off_capture
      COMMAND $<TARGET_FILE:render_demo> ${_demo_capture} --data-dir ${_gxreplay_dir}/demo_off_data
              --trace ${_gxreplay_dir}/render_demo_off.gxtrace)
    set_tests_properties(gxreplay_render_demo_off_capture PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=0" FIXTURES_SETUP gxreplay_demo_off_trace)
    add_test(NAME gxreplay_render_demo_plain_resident
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_plain.gxtrace --backend null --no-png
              --out ${_gxreplay_dir}/render_demo_plain_resident)
    set_tests_properties(gxreplay_render_demo_plain_resident PROPERTIES LABELS smoke TIMEOUT 180
      FIXTURES_REQUIRED gxreplay_demo_plain_trace PASS_REGULAR_EXPRESSION "resident arrays: run 0, stages 0, hits 0"
      FAIL_REGULAR_EXPRESSION "${_resident_fail}")

    set(_demo_ab ${_gxreplay_dir}/hints_ab)
    add_test(NAME gxreplay_render_demo_hints_ab_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --no-png
              --out ${_demo_ab}/hints)
    set_tests_properties(gxreplay_render_demo_hints_ab_replay PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_trace
      FIXTURES_SETUP gxreplay_demo_ab_hints)
    add_test(NAME gxreplay_render_demo_plain_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_plain.gxtrace --backend null --no-png
              --out ${_demo_ab}/plain)
    set_tests_properties(gxreplay_render_demo_plain_replay PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_plain_trace
      FIXTURES_SETUP gxreplay_demo_ab_plain)
    add_test(NAME gxreplay_render_demo_off_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo_off.gxtrace --backend null --no-png
              --out ${_demo_ab}/off_capture)
    set_tests_properties(gxreplay_render_demo_off_replay PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_off_trace
      FIXTURES_SETUP gxreplay_demo_ab_off_capture)
    add_test(NAME gxreplay_render_demo_hints_off_replay
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --no-png
              --out ${_demo_ab}/off_replay)
    set_tests_properties(gxreplay_render_demo_hints_off_replay PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=0" FIXTURES_REQUIRED gxreplay_demo_trace
      FIXTURES_SETUP gxreplay_demo_ab_off_replay)

    set(_compare_frames ${CMAKE_CURRENT_SOURCE_DIR}/compare_replay_frames.cmake)
    add_test(NAME gxreplay_render_demo_hint_ab
      COMMAND ${CMAKE_COMMAND} -DA=${_demo_ab}/hints/frames.json -DB=${_demo_ab}/plain/frames.json -DKEY=draws
              -P ${_compare_frames})
    set_tests_properties(gxreplay_render_demo_hint_ab PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED "gxreplay_demo_ab_hints;gxreplay_demo_ab_plain")
    add_test(NAME gxreplay_render_demo_off_draws_ab
      COMMAND ${CMAKE_COMMAND} -DA=${_demo_ab}/off_capture/frames.json -DB=${_demo_ab}/plain/frames.json
              -DKEY=draws -P ${_compare_frames})
    set_tests_properties(gxreplay_render_demo_off_draws_ab PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED "gxreplay_demo_ab_off_capture;gxreplay_demo_ab_plain")
    # Emitting hints with decoding off may add nothing to the GX stream but the hints themselves
    add_test(NAME gxreplay_render_demo_off_bytes_ab
      COMMAND ${CMAKE_COMMAND} -DA=${_demo_ab}/off_capture/frames.json -DB=${_demo_ab}/plain/frames.json
              -DKEY=fifoBytes -DMINUS_A=hintBytes -P ${_compare_frames})
    set_tests_properties(gxreplay_render_demo_off_bytes_ab PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED "gxreplay_demo_ab_off_capture;gxreplay_demo_ab_plain")
    # A replay that skips the hints draws the capture-only box, one draw per frame
    add_test(NAME gxreplay_render_demo_hints_off_ab
      COMMAND ${CMAKE_COMMAND} -DA=${_demo_ab}/off_replay/frames.json -DB=${_demo_ab}/plain/frames.json
              -DKEY=draws -DOFFSET=1 -P ${_compare_frames})
    set_tests_properties(gxreplay_render_demo_hints_off_ab PROPERTIES LABELS smoke TIMEOUT 60
      FIXTURES_REQUIRED "gxreplay_demo_ab_off_replay;gxreplay_demo_ab_plain")
    # The decoded trace has no extra draw, so this comparison must fail; it proves the script can
    add_test(NAME gxreplay_render_demo_compare_rejects
      COMMAND ${CMAKE_COMMAND} -DA=${_demo_ab}/hints/frames.json -DB=${_demo_ab}/plain/frames.json -DKEY=draws
              -DOFFSET=1 -P ${_compare_frames})
    set_tests_properties(gxreplay_render_demo_compare_rejects PROPERTIES LABELS smoke TIMEOUT 60 WILL_FAIL TRUE
      FIXTURES_REQUIRED "gxreplay_demo_ab_hints;gxreplay_demo_ab_plain")

    # A view forces identity collection and so adds draws, which is why the A/B replays above run without one
    add_test(NAME gxreplay_render_demo_view_kind
      COMMAND $<TARGET_FILE:aurora_gxreplay> ${_gxreplay_dir}/render_demo.gxtrace --backend null --no-png
              --debug-view kind --classify-json --out ${_gxreplay_dir}/view_kind)
    string(CONCAT _view_summary "main [1-9][0-9]*, sky 0, sky misses 0, suppressed 4, "
                                "hint errors 0, malformed hints 0, view substituted [1-9]")
    set_tests_properties(gxreplay_render_demo_view_kind PROPERTIES LABELS smoke TIMEOUT 180
      ENVIRONMENT "AURORA_RENDER_HINTS=1" FIXTURES_REQUIRED gxreplay_demo_trace
      PASS_REGULAR_EXPRESSION "${_view_summary}"
      FAIL_REGULAR_EXPRESSION "\\[(ERROR|FATAL): ")
  endif ()
endif ()

# DVD API tests
if (TARGET aurora::dvd)
  add_executable(dvd_tests
    test_dvd.cpp
    dvd_test_stubs.cpp
    ../lib/logging.cpp
  )
  target_link_libraries(dvd_tests PRIVATE aurora::dvd gtest gtest_main)
  aurora_copy_runtime_dlls(dvd_tests)

  # Pass disc image path as a compile definition if provided
  if (DEFINED DVD_TEST_IMAGE AND EXISTS "${DVD_TEST_IMAGE}")
    target_compile_definitions(dvd_tests PRIVATE DVD_TEST_IMAGE="${DVD_TEST_IMAGE}")
  endif ()

  gtest_discover_tests(dvd_tests)
endif ()

add_executable(os_alloc_tests
  os_alloc_test.cpp
  os_test_globals.cpp
  ../lib/dolphin/os/OSAlloc.cpp
  ../lib/logging.cpp
)
target_include_directories(os_alloc_tests PRIVATE
  ../include
)
target_compile_definitions(os_alloc_tests PRIVATE AURORA TARGET_PC=1)
target_link_libraries(os_alloc_tests PRIVATE gtest gtest_main fmt::fmt)
gtest_discover_tests(os_alloc_tests)
