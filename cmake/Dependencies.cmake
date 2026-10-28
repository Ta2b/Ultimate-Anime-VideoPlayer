# Toutes les dépendances tierces sont trouvées ici, une fois, et exposées sous
# des cibles ajp::dep::<nom>. Les modules ne font jamais de find_package.
#
# Deux sources possibles, mêmes versions : le manifeste vcpkg (CI, Windows,
# versions livrées), ou un préfixe construit par tools/deps/linux-from-source.sh.
include_guard(GLOBAL)

function(ajp_alias_dep name target)
  add_library(ajp_dep_${name} INTERFACE)
  target_link_libraries(ajp_dep_${name} INTERFACE ${target})
  add_library(ajp::dep::${name} ALIAS ajp_dep_${name})
endfunction()

# toml++ en en-têtes seuls, partout : la bibliothèque compilée de vcpkg (Windows) n'exporte que
# l'API à exceptions (toml::ex), et ajp se construit avec TOML_EXCEPTIONS=0 (toml::noex). Ses
# en-têtes sont lus dans la copie corrigée de cmake/TomlFixes.cmake (<toml++-ajp/toml.hpp>), et
# src/core/toml_config.h complète sa configuration.
include("${CMAKE_CURRENT_LIST_DIR}/TomlFixes.cmake")
find_package(tomlplusplus CONFIG REQUIRED)
get_target_property(ajp_toml_include tomlplusplus::tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
ajp_toml_fixed_include("${ajp_toml_include}" ajp_toml_fixed)
add_library(ajp_toml_headers INTERFACE)
target_include_directories(ajp_toml_headers SYSTEM INTERFACE "${ajp_toml_fixed}")
target_compile_definitions(ajp_toml_headers INTERFACE TOML_HEADER_ONLY=1
                                                      "TOML_CONFIG_HEADER=\"${CMAKE_CURRENT_LIST_DIR}/../src/core/toml_config.h\"")
ajp_alias_dep(toml ajp_toml_headers)

find_package(spdlog CONFIG REQUIRED)
ajp_alias_dep(spdlog spdlog::spdlog)
target_compile_definitions(ajp_dep_spdlog INTERFACE SPDLOG_USE_STD_FORMAT)

# CUDA 13.4, seulement avec AJP_WITH_CUDA (backend TensorRT et ses noyaux). Le runtime statique
# charge le pilote (libcuda) à l'exécution : sans pilote, un binaire démarre et refuse proprement.
if(AJP_WITH_CUDA)
  find_package(CUDAToolkit 13 REQUIRED)
  ajp_alias_dep(cudart CUDA::cudart_static)
endif()

if(NOT AJP_BUILD_PLAYER)
  return()
endif()

find_package(PkgConfig REQUIRED)

# FFmpeg 9.0.2 (libavformat 63.1), LGPL partagée.
pkg_check_modules(AJP_FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=63.1 libavcodec>=63.1 libavutil>=61.1 libswresample>=7.1 libswscale>=10.1)
ajp_alias_dep(ffmpeg PkgConfig::AJP_FFMPEG)

# libplacebo v7.360.1 (overlay ports/libplacebo).
pkg_check_modules(AJP_PLACEBO REQUIRED IMPORTED_TARGET libplacebo>=7.360)
ajp_alias_dep(placebo PkgConfig::AJP_PLACEBO)

# libass du fork the-database, commit 896614fa (overlay ports/libass).
pkg_check_modules(AJP_ASS REQUIRED IMPORTED_TARGET libass)
ajp_alias_dep(ass PkgConfig::AJP_ASS)

# fontconfig, facultatif : la police du bureau pour l'interface (app/play/ui.cpp).
pkg_check_modules(AJP_FONTCONFIG QUIET IMPORTED_TARGET fontconfig)
if(AJP_FONTCONFIG_FOUND)
  ajp_alias_dep(fontconfig PkgConfig::AJP_FONTCONFIG)
endif()

# libva, facultative (Linux) : choisir le nœud de rendu VA-API qui décode le codec
# (media/vaapi_device.cpp). Sans elle, VA-API prend l'appareil par défaut.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  pkg_check_modules(AJP_VA QUIET IMPORTED_TARGET libva)
  if(AJP_VA_FOUND)
    ajp_alias_dep(va PkgConfig::AJP_VA)
  endif()
endif()

find_package(SDL3 CONFIG REQUIRED)

ajp_alias_dep(sdl3 SDL3::SDL3)

find_package(VulkanHeaders CONFIG QUIET)
if(TARGET Vulkan::Headers)
  ajp_alias_dep(vulkan Vulkan::Headers)
else()
  find_package(Vulkan REQUIRED)
  ajp_alias_dep(vulkan Vulkan::Headers)
endif()

# Compilateur GLSL vers SPIR-V, pour les shaders compilés hors ligne.
# Sans SDK Vulkan (Windows, vcpkg) : le glslc du port shaderc, outil de l'hôte.
find_program(AJP_GLSLC NAMES glslc
  HINTS "$ENV{VULKAN_SDK}/bin" "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/shaderc" REQUIRED)

# Dear ImGui : port vcpkg, ou sources installées par le script du préfixe.
find_package(imgui CONFIG QUIET)
if(TARGET imgui::imgui)
  ajp_alias_dep(imgui imgui::imgui)
else()
  find_path(AJP_IMGUI_DIR imgui.cpp PATH_SUFFIXES share/imgui REQUIRED)
  add_library(ajp_imgui STATIC
    "${AJP_IMGUI_DIR}/imgui.cpp" "${AJP_IMGUI_DIR}/imgui_draw.cpp"
    "${AJP_IMGUI_DIR}/imgui_tables.cpp" "${AJP_IMGUI_DIR}/imgui_widgets.cpp"
    "${AJP_IMGUI_DIR}/backends/imgui_impl_sdl3.cpp")
  target_include_directories(ajp_imgui SYSTEM PUBLIC "${AJP_IMGUI_DIR}" "${AJP_IMGUI_DIR}/backends")
  target_link_libraries(ajp_imgui PUBLIC SDL3::SDL3)
  # FreeType pour les polices : stb_truetype ne lit ni le CFF2 (Noto Sans CJK variable, installé par défaut sous Fedora) ni
  # les polices variables ; sans ce chargeur le japonais n'apparaît pas. Facultatif : sans FreeType ou sans les sources du
  # chargeur, ImGui garde stb_truetype.
  pkg_check_modules(AJP_FREETYPE QUIET IMPORTED_TARGET freetype2)
  if(AJP_FREETYPE_FOUND AND EXISTS "${AJP_IMGUI_DIR}/misc/freetype/imgui_freetype.cpp")
    target_sources(ajp_imgui PRIVATE "${AJP_IMGUI_DIR}/misc/freetype/imgui_freetype.cpp")
    target_include_directories(ajp_imgui SYSTEM PUBLIC "${AJP_IMGUI_DIR}/misc/freetype")
    target_link_libraries(ajp_imgui PUBLIC PkgConfig::AJP_FREETYPE)
    target_compile_definitions(ajp_imgui PUBLIC AJP_WITH_FREETYPE=1)
  endif()
  ajp_alias_dep(imgui ajp_imgui)
endif()

find_path(AJP_NANOBENCH_DIR nanobench.h REQUIRED)
add_library(ajp_dep_nanobench INTERFACE)
target_include_directories(ajp_dep_nanobench SYSTEM INTERFACE "${AJP_NANOBENCH_DIR}")
add_library(ajp::dep::nanobench ALIAS ajp_dep_nanobench)

# TensorRT 11.3, seulement avec AJP_WITH_CUDA : en-têtes publics et libnvinfer. Sous Linux,
# AJP_TENSORRT_ROOT pointe un dossier include/ lib/ bin/ (sur le PC de référence ~/ajp-sdk/tensorrt :
# bibliothèques du paquet PyPI tensorrt-cu13-libs, en-têtes de NVIDIA/TensorRT, trtexec).
# libnvonnxparser et trtexec ne servent qu'aux tests, qui construisent leurs propres moteurs.
if(AJP_WITH_CUDA)
  set(AJP_TENSORRT_ROOT "" CACHE PATH "Racine de TensorRT 11 : include/, lib/ et bin/trtexec")
  find_path(AJP_TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${AJP_TENSORRT_ROOT}/include" REQUIRED)
  # Sous Windows, les bibliothèques d'import portent la version (nvinfer_11.lib).
  find_library(AJP_NVINFER_LIBRARY NAMES nvinfer nvinfer_11 HINTS "${AJP_TENSORRT_ROOT}/lib" REQUIRED)
  find_library(AJP_NVONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_11 HINTS "${AJP_TENSORRT_ROOT}/lib"
               REQUIRED)
  find_program(AJP_TRTEXEC trtexec HINTS "${AJP_TENSORRT_ROOT}/bin")

  add_library(ajp_dep_tensorrt INTERFACE)
  target_include_directories(ajp_dep_tensorrt SYSTEM INTERFACE "${AJP_TENSORRT_INCLUDE_DIR}")
  target_link_libraries(ajp_dep_tensorrt INTERFACE "${AJP_NVINFER_LIBRARY}" ajp::dep::cudart)
  add_library(ajp::dep::tensorrt ALIAS ajp_dep_tensorrt)

  ajp_alias_dep(tensorrt_onnx "${AJP_NVONNXPARSER_LIBRARY}")
  target_link_libraries(ajp_dep_tensorrt_onnx INTERFACE ajp::dep::tensorrt)
endif()
