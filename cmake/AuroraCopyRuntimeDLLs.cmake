include_guard(GLOBAL)

# aurora_copy_runtime_dlls(<executable_target> [<additional_targets>...])
#
# POST_BUILD command that copies all TARGET_RUNTIME_DLLS from the given targets
# next to the executable, plus any extra DLLs that Dawn loads dynamically
# (dxcompiler.dll, dxil.dll).
function(aurora_copy_runtime_dlls target)
  if (NOT WIN32)
    return()
  endif ()

  # Copy TARGET_RUNTIME_DLLS for each target
  foreach (_t IN ITEMS ${target} ${ARGN})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${_t}> $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS
    )
  endforeach ()

  # Copy DXC DLLs that Dawn loads via LoadLibrary at runtime
  if (TARGET dawn::webgpu_dawn)
    get_target_property(_dawn_loc dawn::webgpu_dawn IMPORTED_LOCATION)
    if (NOT _dawn_loc)
      get_target_property(_dawn_loc dawn::webgpu_dawn IMPORTED_LOCATION_RELEASE)
    endif ()
    if (_dawn_loc)
      cmake_path(GET _dawn_loc PARENT_PATH _dawn_bin_dir)
      foreach (_dll IN ITEMS dxcompiler.dll dxil.dll)
        if (EXISTS "${_dawn_bin_dir}/${_dll}")
          add_custom_command(TARGET ${target} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dawn_bin_dir}/${_dll}" $<TARGET_FILE_DIR:${target}>
          )
        endif ()
      endforeach ()
    endif ()
  endif ()

  # A vendored Dawn builds dxcompiler and copies dxil.dll into the top-level build directory only
  if (TARGET dxcompiler)
    add_dependencies(${target} dxcompiler)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:dxcompiler> $<TARGET_FILE_DIR:${target}>
    )
  endif ()
  if (TARGET copy_dxil_dll)
    add_dependencies(${target} copy_dxil_dll)
    get_property(_aurora_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (_aurora_multi_config)
      set(_aurora_dxil_dll "${CMAKE_BINARY_DIR}/$<CONFIG>/dxil.dll")
    else ()
      set(_aurora_dxil_dll "${CMAKE_BINARY_DIR}/dxil.dll")
    endif ()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_aurora_dxil_dll}" $<TARGET_FILE_DIR:${target}>
    )
  endif ()
endfunction()

# aurora_install_runtime_dlls(<executable_target> <install_destination>)
#
# INSTALL rule that copies all DLLs already placed next to the built target
# into the install destination.
function(aurora_install_runtime_dlls target install_destination)
  if (NOT WIN32)
    return()
  endif ()

  install(CODE
    "file(GLOB _aurora_runtime_dlls \"$<TARGET_FILE_DIR:${target}>/*.dll\")\n\
if(_aurora_runtime_dlls)\n\
  file(INSTALL DESTINATION \"${install_destination}\" TYPE FILE FILES \${_aurora_runtime_dlls})\n\
endif()"
  )
endfunction()
