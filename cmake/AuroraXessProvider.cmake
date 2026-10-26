include_guard(GLOBAL)
include(CMakeDependentOption)

set(_aurora_xess_platform OFF)
if (AURORA_ENABLE_NATIVE_D3D12 AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|X64|AMD64)$")
  set(_aurora_xess_platform ON)
endif ()
cmake_dependent_option(AURORA_ENABLE_XESS "Fetch Intel XeSS-SR into the build tree and load it at runtime" ON
    "_aurora_xess_platform" OFF)

function(aurora_copy_xess_runtime target)
  if (NOT AURORA_XESS_RUNTIME_DLL)
    return()
  endif ()
  add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AURORA_XESS_RUNTIME_DLL}" $<TARGET_FILE_DIR:${target}>
  )
endfunction()

if (NOT AURORA_ENABLE_XESS)
  set(AURORA_XESS_INCLUDE_DIR "" CACHE INTERNAL "")
  set(AURORA_XESS_RUNTIME_DLL "" CACHE INTERNAL "")
  set(AURORA_XESS_LICENSE_FILES "" CACHE INTERNAL "")
  return()
endif ()

set(_aurora_xess_dir "${CMAKE_BINARY_DIR}/_deps/xess")
foreach (_aurora_xess_entry IN LISTS AURORA_XESS_FILES)
  string(FIND "${_aurora_xess_entry}" "=" _aurora_xess_split)
  string(SUBSTRING "${_aurora_xess_entry}" 0 ${_aurora_xess_split} _aurora_xess_path)
  math(EXPR _aurora_xess_split "${_aurora_xess_split} + 1")
  string(SUBSTRING "${_aurora_xess_entry}" ${_aurora_xess_split} -1 _aurora_xess_hash)
  set(_aurora_xess_file "${_aurora_xess_dir}/${_aurora_xess_path}")
  if (EXISTS "${_aurora_xess_file}")
    file(SHA256 "${_aurora_xess_file}" _aurora_xess_have)
    if (_aurora_xess_have STREQUAL _aurora_xess_hash)
      continue()
    endif ()
  endif ()
  message(STATUS "aurora: Fetching XeSS ${AURORA_XESS_VERSION} ${_aurora_xess_path}")
  file(DOWNLOAD "https://raw.githubusercontent.com/intel/xess/${AURORA_XESS_REF}/${_aurora_xess_path}"
      "${_aurora_xess_file}.part"
      EXPECTED_HASH SHA256=${_aurora_xess_hash}
      TLS_VERIFY ON)
  file(RENAME "${_aurora_xess_file}.part" "${_aurora_xess_file}")
endforeach ()

set(AURORA_XESS_INCLUDE_DIR "${_aurora_xess_dir}/inc" CACHE INTERNAL "")
set(AURORA_XESS_RUNTIME_DLL "${_aurora_xess_dir}/bin/libxess.dll" CACHE INTERNAL "")
set(AURORA_XESS_LICENSE_FILES "${_aurora_xess_dir}/LICENSE.txt;${_aurora_xess_dir}/third-party-programs.txt"
    CACHE INTERNAL "")
