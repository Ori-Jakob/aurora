add_library(aurora_gx STATIC
        lib/gfx/clear.cpp
        lib/gfx/depth_peek.cpp
        lib/gfx/encoding.cpp
        lib/gfx/frame.cpp
        lib/gfx/pipeline_cache.cpp
        lib/gfx/recording.cpp
        lib/gfx/render_worker.cpp
        lib/gfx/resource_cache.cpp
        lib/gfx/dds_io.cpp
        lib/gfx/tex_copy_conv.cpp
        lib/gfx/tex_palette_conv.cpp
        lib/gfx/texture.cpp
        lib/gfx/texture_format.cpp
        lib/gfx/texture_convert.cpp
        lib/gfx/texture_replacement.cpp
        lib/gx/attr_fmt.cpp
        lib/gx/command_processor.cpp
        lib/gx/regs.cpp
        lib/gx/dl.cpp
        lib/gx/fifo.cpp
        lib/gx/gx.cpp
        lib/gx/draw_route.hpp
        lib/gx/gpu_channel.cpp
        lib/gx/gpu_channel.hpp
        lib/gx/hints.cpp
        lib/gx/hints.hpp
        lib/gx/texture.cpp
        lib/gx/pipeline.cpp
        lib/gx/shader.cpp
        lib/gx/shader_info.cpp
        lib/dolphin/gx/GXBump.cpp
        lib/dolphin/gx/GXCull.cpp
        lib/dolphin/gx/GXCpu2Efb.cpp
        lib/dolphin/gx/GXDispList.cpp
        lib/dolphin/gx/GXDraw.cpp
        lib/dolphin/gx/GXExtra.cpp
        lib/dolphin/gx/GXFifo.cpp
        lib/dolphin/gx/GXFrameBuffer.cpp
        lib/dolphin/gx/GXGeometry.cpp
        lib/dolphin/gx/GXGet.cpp
        lib/dolphin/gx/GXLighting.cpp
        lib/dolphin/gx/GXManage.cpp
        lib/dolphin/gx/GXPerf.cpp
        lib/dolphin/gx/GXPixel.cpp
        lib/dolphin/gx/GXTev.cpp
        lib/dolphin/gx/GXTexture.cpp
        lib/dolphin/gx/GXTransform.cpp
        lib/dolphin/gx/GXVert.cpp
        lib/dolphin/gx/GXAurora.cpp
        lib/gfx/png_io.cpp
        lib/gfx/png_io.hpp
        lib/trace/address_map.cpp
        lib/trace/address_map.hpp
        lib/trace/capture.cpp
        lib/trace/capture.hpp
        lib/trace/capture_session.cpp
        lib/trace/capture_session.hpp
        lib/trace/fifo_encoder.hpp
        lib/trace/snapshot_store.cpp
        lib/trace/snapshot_store.hpp
        lib/trace/state_preamble.cpp
        lib/trace/state_preamble.hpp
        lib/trace/trace_format.cpp
        lib/trace/trace_format.hpp
        lib/trace/trace_reader.cpp
        lib/trace/trace_reader.hpp
        lib/trace/trace_writer.cpp
        lib/trace/trace_writer.hpp
)
add_library(aurora::gx ALIAS aurora_gx)
set_target_properties(aurora_gx PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_gx PUBLIC aurora::core dawn::webgpu_dawn xxHash::xxhash)
target_link_libraries(aurora_gx PRIVATE absl::btree absl::flat_hash_map sqlite3 Tracy::TracyClient PNG::PNG)
target_compile_definitions(aurora_gx PRIVATE WEBGPU_DAWN)
if (TARGET zstd::libzstd)
    target_compile_definitions(aurora_gx PRIVATE AURORA_TRACE_ZSTD)
    target_link_libraries(aurora_gx PRIVATE zstd::libzstd)
endif ()

if (AURORA_ENABLE_RMLUI)
    target_sources(aurora_gx PRIVATE
        lib/rmlui/pipeline.cpp
        lib/rmlui/pipeline.hpp
    )
endif ()

include(${CMAKE_CURRENT_LIST_DIR}/aurora_render.cmake)
