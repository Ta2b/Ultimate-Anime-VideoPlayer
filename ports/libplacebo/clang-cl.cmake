# libplacebo ne se construit qu'avec GCC ou Clang : son meson.build passe ses -W et -f tels quels,
# que cl refuse (D8021). Sous Windows, clang-cl (livré avec Visual Studio) prend sa place, le reste
# comme la chaîne Windows par défaut de vcpkg.
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
# La racine de vcpkg : celle de scripts/buildsystems/vcpkg.cmake, qui inclut ce fichier (les
# try_compile ne reçoivent ni _VCPKG_ROOT_DIR ni CMAKE_TOOLCHAIN_FILE).
get_filename_component(ajp_vcpkg_root "${CMAKE_PARENT_LIST_FILE}/../../.." ABSOLUTE)
include("${ajp_vcpkg_root}/scripts/toolchains/windows.cmake")
