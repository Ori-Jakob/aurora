target_sources(aurora_gx PRIVATE
        lib/render/frame_state.hpp
        lib/render/graph.hpp
        lib/render/module.hpp
        lib/render/render.cpp
        lib/render/settings.cpp
        lib/render/settings.hpp
)
if (AURORA_ENABLE_RENDER)
    target_sources(aurora_gx PRIVATE
        lib/render/graph.cpp
        lib/render/json.hpp
    )
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=1)
    target_link_libraries(aurora_gx PRIVATE yyjson::yyjson)
else ()
    target_compile_definitions(aurora_gx PRIVATE AURORA_ENABLE_RENDER=0)
endif ()
