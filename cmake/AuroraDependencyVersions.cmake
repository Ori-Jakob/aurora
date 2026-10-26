include_guard(GLOBAL)

# Specifies a cache string and previous values to forcibly migrate from
macro(_aurora_dependency_version name value doc)
  set(_aurora_old_defaults ${ARGN})
  if (DEFINED CACHE{${name}} AND ${name} IN_LIST _aurora_old_defaults)
    message(STATUS "aurora: Migrating ${name} from old default ${${name}} to ${value}")
    set(${name} "${value}" CACHE STRING "${doc}" FORCE)
  else ()
    set(${name} "${value}" CACHE STRING "${doc}")
  endif ()
  unset(_aurora_old_defaults)
endmacro()

# Dependency versions
_aurora_dependency_version(AURORA_DAWN_VERSION "v20260807.225922" "Dawn prebuilt version tag (https://github.com/encounter/dawn/releases)"
        "v20260523.201736" "v20260603.191052" "v20260618.032059") # Previous versions
_aurora_dependency_version(AURORA_DAWN_REF "1155e0ed531126f33a1279afa029349651ca1c93" "Dawn commit ref (https://github.com/encounter/dawn)"
        "9aa45f938d4b36626722bbfdc2f18447179337e6" "13abc3bc8ea2d3c2050f9e77a12d012108ceee24" "266c1cf8de969a364afa4fa49311631fc99a881e") # Previous versions
_aurora_dependency_version(AURORA_SDL3_VERSION "3.4.10" "SDL3 prebuilt version tag (https://github.com/libsdl-org/SDL/releases)")
_aurora_dependency_version(AURORA_SDL3_REF "refs/tags/release-3.4.10" "SDL3 commit ref (https://github.com/libsdl-org/SDL)")
_aurora_dependency_version(AURORA_NOD_VERSION "v2.0.0-alpha.12" "nod version tag (https://github.com/encounter/nod/releases)"
        "v2.0.0-alpha.10" "v2.0.0-alpha.11") # Previous versions
_aurora_dependency_version(AURORA_XESS_VERSION "v3.0.2" "Intel XeSS SDK version tag (https://github.com/intel/xess/releases)")
_aurora_dependency_version(AURORA_XESS_REF "8fe81bdbbaf00b3c1b733fd0d830c333dc84e6f0" "Intel XeSS commit ref (https://github.com/intel/xess)")
set(AURORA_XESS_FILES
    "inc/xess/xess.h=98dc435f5aed252d83a161fc1f86223f20c67eb863214fe3b5cb215a3f6e3319"
    "inc/xess/xess_d3d12.h=a582a60c3e39ee6fd6aa4871e23082cfb5ec510f98a19fc04acf7a7edb2f6b6c"
    "bin/libxess.dll=251659dd84a3e84de67c886a4186e01f3eca49b00641906fe38bb6b807e5d5b7"
    "LICENSE.txt=784be42c39a4a4d03cf85d88e15e59d75828fca773ce45b8dbfffbd7eaf208df"
    "third-party-programs.txt=e8d8899dbfc4de0e8ca25560be9358e03c1405d7c956d20a1838c3c709b9b51c")
_aurora_dependency_version(AURORA_DLSS_VERSION "v310.9.16" "NVIDIA DLSS SDK version tag (https://github.com/NVIDIA/DLSS/releases)")
_aurora_dependency_version(AURORA_DLSS_REF "9981c6350dc58c0ce21f024fd65e31e851c31689" "NVIDIA DLSS commit ref (https://github.com/NVIDIA/DLSS)")
set(AURORA_DLSS_FILES
    "include/nvsdk_ngx.h=dc38e7467cf415379c9d12ae1b6e4a494c453ed92720fb53e92aecb523e7b848"
    "include/nvsdk_ngx_defs.h=ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e"
    "include/nvsdk_ngx_params.h=943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc"
    "include/nvsdk_ngx_helpers.h=5bcbadfe7478b802cf6d3aca4dc5ddd7d0889b99726e69c63f9e9bd555f44471"
    "include/nvsdk_ngx_helpers_d3d.h=ec75224f36ed6580baaf250fa83620405ad81a1d98c2c436654a0c9cf6a6b8ba"
    "include/nvsdk_ngx_helpers_cuda.h=f2851a76107bdf4fbcb7b261f3c35573c312d8ecb4bc18fb02d9e2cec7705042"
    "lib/Windows_x86_64/x64/nvsdk_ngx_d.lib=bc43df9ec032004fde681d3e3afd01cde239e03a0125b822e88674c84287f99c"
    "lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib=3f804d50b0323baa2c4976c4c94e2454c4a3f9c6c04e7d1f2302392d5add0009"
    "lib/Windows_x86_64/rel/nvngx_dlss.dll=d00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e"
    "LICENSE.txt=d4216e39ebef5f9b50a6712ebb37beeb5379862a67733a9999c651f21592aaf0")
