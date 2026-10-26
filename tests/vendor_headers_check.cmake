if (NOT GIT_EXECUTABLE OR NOT EXISTS "${SOURCE_DIR}/.git")
  message("VendorHeaders: skipped, ${SOURCE_DIR} is not a git checkout")
  return()
endif ()

set(_patterns
    "XESS_API"
    "(^|[^A-Za-z0-9_])_xess_"
    "xess_result_t[[:space:]]+xess[A-Za-z0-9]+[[:space:]]*[(]"
    "typedef.*xess_"
    "XESS_[A-Z0-9_]+[[:space:]]*=[[:space:]]*-?[0-9]"
    "NVSDK_NGX_API"
    "NVSDK_NGX_Result[[:space:]]+(NVSDK_CONV[[:space:]]+)?(NVSDK_)?NGX_[A-Za-z0-9_]+[[:space:]]*[(]"
    "typedef.*NVSDK_NGX_"
    "NVSDK_NGX_[A-Za-z0-9_]+[[:space:]]*=[[:space:]]*-?[0-9]"
    "#define[[:space:]]+NVSDK_NGX_"
    "Intel copyrighted materials"
    "LicenseRef-NvidiaProprietary")
set(_args "")
foreach (_pattern IN LISTS _patterns)
  list(APPEND _args -e "${_pattern}")
endforeach ()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" grep -n -I -E ${_args} --
    . ":(exclude)tests/vendor_headers_check.cmake" ":(exclude,glob)lib/render/native/d3d12/xess*"
    ":(exclude,glob)lib/render/native/d3d12/dlss*"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _matches
    ERROR_VARIABLE _error)
if (_result EQUAL 1)
  message(STATUS "VendorHeaders: no vendor SDK text in tracked files")
elseif (_result EQUAL 0)
  message(FATAL_ERROR "VendorHeaders: vendor SDK text in tracked files:\n${_matches}")
else ()
  message(FATAL_ERROR "VendorHeaders: git grep failed (${_result}): ${_error}")
endif ()
