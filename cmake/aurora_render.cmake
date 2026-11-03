include(${CMAKE_CURRENT_LIST_DIR}/AuroraEmbedShaders.cmake)

target_sources(aurora_gx PRIVATE
        lib/render/frame_state.hpp
        lib/render/graph.hpp
        lib/render/graph_api.hpp
        lib/render/module.hpp
        lib/render/pass_context.hpp
        lib/render/phases.inl
        lib/render/render.cpp
        lib/render/scene_depth.hpp
        lib/render/settings.cpp
        lib/render/settings.hpp
        lib/render/stage.cpp
        lib/render/stage.hpp
        lib/render/stage_data.hpp
)
if (AURORA_ENABLE_RENDER)
    target_sources(aurora_gx PRIVATE
        lib/render/ao/ao.cpp
        lib/render/ao/ao.hpp
        lib/render/ao/denoise.cpp
        lib/render/ao/denoise.hpp
        lib/render/ao/gtao.cpp
        lib/render/ao/gtao.hpp
        lib/render/ao/hiz.cpp
        lib/render/ao/hiz.hpp
        lib/render/ao/upsample.cpp
        lib/render/ao/upsample.hpp
        lib/render/debug/capture.cpp
        lib/render/debug/capture.hpp
        lib/render/debug/class_view.cpp
        lib/render/debug/debug.hpp
        lib/render/debug/debug_window.cpp
        lib/render/debug/decomp_view.cpp
        lib/render/debug/decomp_view.hpp
        lib/render/debug/dual_source.cpp
        lib/render/debug/dual_source.hpp
        lib/render/debug/fake_shadows.cpp
        lib/render/debug/fake_shadows.hpp
        lib/render/debug/gbuf_view.cpp
        lib/render/debug/gbuf_view.hpp
        lib/render/debug/graph_dump.cpp
        lib/render/debug/graph_inspection.hpp
        lib/render/debug/graph_tab.cpp
        lib/render/debug/overview_tab.cpp
        lib/render/debug/pick.cpp
        lib/render/debug/pick.hpp
        lib/render/debug/scene_tab.cpp
        lib/render/debug/surface_expand.cpp
        lib/render/debug/surface_expand.hpp
        lib/render/debug/textures.cpp
        lib/render/debug/textures.hpp
        lib/render/debug/views.cpp
        lib/render/debug/views.hpp
        lib/render/debug/views_tab.cpp
        lib/render/fakeshadow/texture_gate.cpp
        lib/render/fakeshadow/texture_gate.hpp
        lib/render/geometry/compact.cpp
        lib/render/geometry/compact.hpp
        lib/render/geometry/hzb.cpp
        lib/render/geometry/hzb.hpp
        lib/render/geometry/indirect.cpp
        lib/render/geometry/indirect.hpp
        lib/render/gpu_heap.cpp
        lib/render/gpu_heap.hpp
        lib/render/graph.cpp
        lib/render/graph_api.cpp
        lib/render/graph_bind.cpp
        lib/render/graph_bind.hpp
        lib/render/graph_env.cpp
        lib/render/graph_layout.cpp
        lib/render/graph_layout.hpp
        lib/render/graph_pool.cpp
        lib/render/graph_pool.hpp
        lib/render/graph_slots.cpp
        lib/render/graph_slots.hpp
        lib/render/gx_ext/ext_provider.cpp
        lib/render/gx_ext/ext_provider.hpp
        lib/render/gx_ext/fwd_env.cpp
        lib/render/gx_ext/fwd_env.hpp
        lib/render/gx_ext/records.cpp
        lib/render/gx_ext/records.hpp
        lib/render/gx_ext/surface_table.cpp
        lib/render/gx_ext/surface_table.hpp
        lib/render/gx_ext/unlit.cpp
        lib/render/gx_ext/unlit.hpp
        lib/render/hints.cpp
        lib/render/hints.hpp
        lib/render/history_reset.cpp
        lib/render/history_reset.hpp
        lib/render/jobs.cpp
        lib/render/jobs.hpp
        lib/render/json.hpp
        lib/render/material/gx_lookup.cpp
        lib/render/material/gx_lookup.hpp
        lib/render/material/key_text.hpp
        lib/render/material/manifest.cpp
        lib/render/material/manifest.hpp
        lib/render/material/mip_repair.cpp
        lib/render/material/mip_repair.hpp
        lib/render/material/registry.cpp
        lib/render/material/registry.hpp
        lib/render/material/schema_text.cpp
        lib/render/material/user_layer.cpp
        lib/render/material/user_layer.hpp
    )
    if (AURORA_ENABLE_NATIVE_D3D12)
        target_sources(aurora_gx PRIVATE
            lib/render/native/d3d12/device.cpp
            lib/render/native/d3d12/device.hpp
            lib/render/native/d3d12/dlss.cpp
            lib/render/native/d3d12/dlss.hpp
            lib/render/native/d3d12/dlss_upscaler.cpp
            lib/render/native/d3d12/native_step.cpp
            lib/render/native/d3d12/native_step.hpp
            lib/render/native/d3d12/probe.cpp
            lib/render/native/d3d12/probe.hpp
            lib/render/native/d3d12/sentinel.cpp
            lib/render/native/d3d12/sentinel.hpp
            lib/render/native/d3d12/shared_texture.cpp
            lib/render/native/d3d12/shared_texture.hpp
            lib/render/native/d3d12/teardown.cpp
            lib/render/native/d3d12/teardown.hpp
            lib/render/native/d3d12/upscale_bridge.cpp
            lib/render/native/d3d12/upscale_bridge.hpp
            lib/render/native/native.cpp
            lib/render/native/native.hpp
        )
        target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_NATIVE_D3D12=1)
        target_compile_definitions(aurora_core PRIVATE AURORA_ENABLE_NATIVE_D3D12=1)
        target_link_libraries(aurora_gx PRIVATE d3d12 dxgi version)
        if (AURORA_ENABLE_DLSS)
            target_sources(aurora_gx PRIVATE lib/render/native/d3d12/dlss_ngx.cpp)
            target_link_libraries(aurora_gx PRIVATE aurora::ngx)
            target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_DLSS=1)
        else ()
            target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_DLSS=0)
        endif ()
        if (AURORA_ENABLE_XESS)
            target_sources(aurora_gx PRIVATE lib/render/native/d3d12/xess.cpp lib/render/native/d3d12/xess.hpp
                lib/render/native/d3d12/xess_upscaler.cpp)
            target_include_directories(aurora_gx SYSTEM PRIVATE "${AURORA_XESS_INCLUDE_DIR}")
            target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_XESS=1)
        else ()
            target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_XESS=0)
        endif ()
    else ()
        target_sources(aurora_gx PRIVATE lib/render/native/native.hpp lib/render/native/native_off.cpp)
        target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_NATIVE_D3D12=0)
    endif ()
    target_sources(aurora_gx PRIVATE
        lib/render/post/composite.cpp
        lib/render/post/composite.hpp
        lib/render/post/taa.cpp
        lib/render/post/taa.hpp
        lib/render/post/temporal_resolve.cpp
        lib/render/post/temporal_resolve.hpp
        lib/render/post/upscale/compose.cpp
        lib/render/post/upscale/compose.hpp
        lib/render/post/upscale/fsr/fsr.cpp
        lib/render/post/upscale/fsr/fsr.hpp
        lib/render/post/upscale/fsr/fsr_pass.cpp
        lib/render/post/upscale/fsr/fsr_pass.hpp
        lib/render/post/upscale/native_upscale.cpp
        lib/render/post/upscale/native_upscale.hpp
        lib/render/post/upscale/upscale.cpp
        lib/render/post/upscale/upscale.hpp
        lib/render/scene/classifier.cpp
        lib/render/scene/classifier.hpp
        lib/render/scene/classify.cpp
        lib/render/scene/classify.hpp
        lib/render/scene/motion.cpp
        lib/render/scene/motion.hpp
        lib/render/scene/overlay.cpp
        lib/render/scene/overlay.hpp
        lib/render/shader_resolve.cpp
        lib/render/shader_resolve.hpp
        lib/render/shaders.cpp
        lib/render/shaders.hpp
        lib/render/shaders_embed.hpp
        lib/render/stage_camera.cpp
        lib/render/stage_camera.hpp
        lib/render/systems/dispatch.cpp
        lib/render/systems/dispatch.hpp
        lib/render/systems/systems.cpp
        lib/render/systems/systems.hpp
        lib/render/tlsf.cpp
        lib/render/tlsf.hpp
    )
    file(GLOB_RECURSE _aurora_render_wgsl CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../lib/render/*.wgsl")
    aurora_embed_wgsl(aurora_gx
        ROOT "${CMAKE_CURRENT_LIST_DIR}/../lib/render"
        TABLE embedded_library
        HEADER "${CMAKE_CURRENT_LIST_DIR}/../lib/render/shaders_embed.hpp"
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/aurora_render_wgsl.cpp"
        SOURCES ${_aurora_render_wgsl})
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=1)
    target_link_libraries(aurora_gx PRIVATE yyjson::yyjson aurora_bccodec)
else ()
    target_sources(aurora_gx PRIVATE lib/render/graph_api_off.cpp lib/render/material/registry_off.cpp
        lib/render/native/native.hpp lib/render/native/native_off.cpp)
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=0 AURORA_ENABLE_NATIVE_D3D12=0)
endif ()
