include(${CMAKE_CURRENT_LIST_DIR}/../cmake/AuroraCopyRuntimeDLLs.cmake)

if (NOT WORK_DIR)
  message(FATAL_ERROR "RuntimeDlls: WORK_DIR is not set")
endif ()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
foreach (_name IN ITEMS aurora.dll dxil.dll libxess.dll nvngx_dlss.dll NVNGX_DLSSD.dll Nvngx_dlssg.dll
                        mynvngx_shim.dll nvngx.txt)
  file(TOUCH "${WORK_DIR}/${_name}")
endforeach ()

aurora_runtime_dlls_to_install(_dlls "${WORK_DIR}")
set(_names "")
foreach (_dll IN LISTS _dlls)
  cmake_path(GET _dll FILENAME _name)
  list(APPEND _names "${_name}")
endforeach ()
list(SORT _names CASE INSENSITIVE)
file(REMOVE_RECURSE "${WORK_DIR}")

set(_expected aurora.dll dxil.dll libxess.dll mynvngx_shim.dll)
if (NOT _names STREQUAL _expected)
  message(FATAL_ERROR "RuntimeDlls: the install would copy [${_names}], expected [${_expected}]")
endif ()
message(STATUS "RuntimeDlls: no nvngx_*.dll reaches the install")
