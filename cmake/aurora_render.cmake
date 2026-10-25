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
        lib/render/debug/class_view.cpp
        lib/render/debug/debug.hpp
        lib/render/debug/debug_window.cpp
        lib/render/debug/graph_dump.cpp
        lib/render/debug/graph_inspection.hpp
        lib/render/debug/graph_tab.cpp
        lib/render/debug/overview_tab.cpp
        lib/render/debug/scene_tab.cpp
        lib/render/debug/views.cpp
        lib/render/debug/views.hpp
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
        lib/render/hints.cpp
        lib/render/hints.hpp
        lib/render/jobs.cpp
        lib/render/jobs.hpp
        lib/render/json.hpp
        lib/render/material/key_text.hpp
        lib/render/material/manifest.cpp
        lib/render/material/manifest.hpp
        lib/render/material/registry.cpp
        lib/render/material/registry.hpp
        lib/render/material/schema_text.cpp
        lib/render/scene/classifier.cpp
        lib/render/scene/classifier.hpp
        lib/render/scene/classify.cpp
        lib/render/scene/classify.hpp
        lib/render/shader_resolve.cpp
        lib/render/shader_resolve.hpp
        lib/render/shaders.cpp
        lib/render/shaders.hpp
        lib/render/shaders_embed.hpp
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
    target_link_libraries(aurora_gx PRIVATE yyjson::yyjson)
else ()
    target_sources(aurora_gx PRIVATE lib/render/graph_api_off.cpp lib/render/material/registry_off.cpp)
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=0)
endif ()
