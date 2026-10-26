include_guard(GLOBAL)
include(CMakeDependentOption)

set(_aurora_dlss_platform OFF)
if (AURORA_ENABLE_NATIVE_D3D12 AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|X64|AMD64)$")
  set(_aurora_dlss_platform ON)
endif ()
cmake_dependent_option(AURORA_ENABLE_DLSS "Fetch the NVIDIA DLSS SDK" OFF "_aurora_dlss_platform" OFF)
if (NOT _aurora_dlss_platform AND "$CACHE{AURORA_ENABLE_DLSS}")
  message(WARNING "aurora: AURORA_ENABLE_DLSS needs native D3D12 on Windows x64, so DLSS stays off")
endif ()

function(aurora_copy_dlss_runtime target)
  if (NOT AURORA_DLSS_RUNTIME_DLL)
    return()
  endif ()
  add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AURORA_DLSS_RUNTIME_DLL}" $<TARGET_FILE_DIR:${target}>
  )
endfunction()

if (NOT AURORA_ENABLE_DLSS)
  set(AURORA_DLSS_RUNTIME_DLL "" CACHE INTERNAL "")
  set(AURORA_DLSS_LICENSE_FILES "" CACHE INTERNAL "")
  return()
endif ()

set(_aurora_dlss_dir "${CMAKE_BINARY_DIR}/_deps/dlss")
foreach (_aurora_dlss_entry IN LISTS AURORA_DLSS_FILES)
  string(FIND "${_aurora_dlss_entry}" "=" _aurora_dlss_split)
  string(SUBSTRING "${_aurora_dlss_entry}" 0 ${_aurora_dlss_split} _aurora_dlss_path)
  math(EXPR _aurora_dlss_split "${_aurora_dlss_split} + 1")
  string(SUBSTRING "${_aurora_dlss_entry}" ${_aurora_dlss_split} -1 _aurora_dlss_hash)
  set(_aurora_dlss_file "${_aurora_dlss_dir}/${_aurora_dlss_path}")
  if (EXISTS "${_aurora_dlss_file}")
    file(SHA256 "${_aurora_dlss_file}" _aurora_dlss_have)
    if (_aurora_dlss_have STREQUAL _aurora_dlss_hash)
      continue()
    endif ()
  endif ()
  message(STATUS "aurora: Fetching DLSS ${AURORA_DLSS_VERSION} ${_aurora_dlss_path}")
  file(DOWNLOAD "https://raw.githubusercontent.com/NVIDIA/DLSS/${AURORA_DLSS_REF}/${_aurora_dlss_path}"
      "${_aurora_dlss_file}.part"
      EXPECTED_HASH SHA256=${_aurora_dlss_hash}
      TLS_VERIFY ON)
  file(RENAME "${_aurora_dlss_file}.part" "${_aurora_dlss_file}")
endforeach ()

add_library(aurora::ngx STATIC IMPORTED GLOBAL)
set_target_properties(aurora::ngx PROPERTIES
    IMPORTED_LOCATION "${_aurora_dlss_dir}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib"
    IMPORTED_LOCATION_DEBUG "${_aurora_dlss_dir}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${_aurora_dlss_dir}/include"
    INTERFACE_LINK_LIBRARIES "advapi32;user32")
set(AURORA_DLSS_RUNTIME_DLL "${_aurora_dlss_dir}/lib/Windows_x86_64/rel/nvngx_dlss.dll" CACHE INTERNAL "")
set(AURORA_DLSS_LICENSE_FILES "${_aurora_dlss_dir}/LICENSE.txt" CACHE INTERNAL "")
