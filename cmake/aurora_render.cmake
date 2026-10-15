target_sources(aurora_gx PRIVATE
        lib/render/frame_state.hpp
        lib/render/graph.hpp
        lib/render/module.hpp
        lib/render/pass_context.hpp
        lib/render/phases.inl
        lib/render/render.cpp
        lib/render/settings.cpp
        lib/render/settings.hpp
        lib/render/stage.cpp
        lib/render/stage.hpp
        lib/render/stage_data.hpp
)
if (AURORA_ENABLE_RENDER)
    target_sources(aurora_gx PRIVATE
        lib/render/gpu_heap.cpp
        lib/render/gpu_heap.hpp
        lib/render/graph.cpp
        lib/render/graph_env.cpp
        lib/render/json.hpp
        lib/render/tlsf.cpp
        lib/render/tlsf.hpp
    )
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=1)
    target_link_libraries(aurora_gx PRIVATE yyjson::yyjson)
else ()
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=0)
endif ()
