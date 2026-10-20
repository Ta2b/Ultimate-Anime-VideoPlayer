# libplacebo ne se construit qu'avec GCC ou Clang : son meson.build passe ses -W et -f tels quels,
# que cl refuse (D8021). Sous Windows, clang-cl (livré avec Visual Studio) prend sa place, le reste
# comme la chaîne Windows par défaut de vcpkg.
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
# La racine de vcpkg : passée par vcpkg_cmake_get_vars, sinon à côté de sa chaîne vcpkg.cmake
# (les try_compile ne reçoivent que celle-ci).
if(_VCPKG_ROOT_DIR)
  set(ajp_vcpkg_root "${_VCPKG_ROOT_DIR}")
else()
  get_filename_component(ajp_vcpkg_root "${CMAKE_TOOLCHAIN_FILE}/../../.." ABSOLUTE)
endif()
include("${ajp_vcpkg_root}/scripts/toolchains/windows.cmake")
