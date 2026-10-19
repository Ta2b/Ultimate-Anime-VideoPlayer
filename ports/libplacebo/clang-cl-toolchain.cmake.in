# Chaîne du port libplacebo sous MSVC : celle de vcpkg pour Windows (CRT, /utf-8,
# drapeaux de configuration), avec clang-cl à la place de cl.exe.
#
# libplacebo est écrit pour GCC et Clang (__builtin_popcount, __attribute__((packed)),
# <stdatomic.h> en C) : cl.exe ne le compile pas. clang-cl produit des objets au même
# ABI, liés par le même éditeur de liens et contre le même CRT que le reste.
find_program(AJP_CLANG_CL NAMES clang-cl
  PATHS "$ENV{VCINSTALLDIR}/Tools/Llvm/x64/bin" "$ENV{ProgramFiles}/LLVM/bin"
  NO_DEFAULT_PATH)
if(NOT AJP_CLANG_CL)
  find_program(AJP_CLANG_CL NAMES clang-cl REQUIRED)
endif()
set(CMAKE_C_COMPILER "${AJP_CLANG_CL}")
set(CMAKE_CXX_COMPILER "${AJP_CLANG_CL}")

# CMAKE_TOOLCHAIN_FILE est scripts/buildsystems/vcpkg.cmake, qui charge ce fichier.
get_filename_component(z_ajp_vcpkg_scripts "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
include("${z_ajp_vcpkg_scripts}/../toolchains/windows.cmake")
