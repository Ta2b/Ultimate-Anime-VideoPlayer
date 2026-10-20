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

find_package(tomlplusplus CONFIG REQUIRED)
ajp_alias_dep(toml tomlplusplus::tomlplusplus)

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

find_package(SDL3 CONFIG REQUIRED)

# Téléchargement des moteurs et des modèles depuis l'index haché.
find_package(CURL REQUIRED)
ajp_alias_dep(curl CURL::libcurl)

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
  find_library(AJP_NVINFER_LIBRARY nvinfer HINTS "${AJP_TENSORRT_ROOT}/lib" REQUIRED)
  find_library(AJP_NVONNXPARSER_LIBRARY nvonnxparser HINTS "${AJP_TENSORRT_ROOT}/lib" REQUIRED)
  find_program(AJP_TRTEXEC trtexec HINTS "${AJP_TENSORRT_ROOT}/bin")

  add_library(ajp_dep_tensorrt INTERFACE)
  target_include_directories(ajp_dep_tensorrt SYSTEM INTERFACE "${AJP_TENSORRT_INCLUDE_DIR}")
  target_link_libraries(ajp_dep_tensorrt INTERFACE "${AJP_NVINFER_LIBRARY}" ajp::dep::cudart)
  add_library(ajp::dep::tensorrt ALIAS ajp_dep_tensorrt)

  ajp_alias_dep(tensorrt_onnx "${AJP_NVONNXPARSER_LIBRARY}")
  target_link_libraries(ajp_dep_tensorrt_onnx INTERFACE ajp::dep::tensorrt)
endif()
