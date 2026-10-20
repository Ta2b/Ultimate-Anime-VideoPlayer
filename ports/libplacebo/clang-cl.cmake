# libplacebo ne se construit qu'avec GCC ou Clang : son meson.build passe ses -W et -f tels quels,
# que cl refuse (D8021). Sous Windows, clang-cl (livré avec Visual Studio) prend sa place, le reste
# comme la chaîne Windows par défaut de vcpkg.
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
include("${Z_VCPKG_ROOT_DIR}/scripts/toolchains/windows.cmake")
