# Toutes les dépendances tierces sont trouvées ici, une fois, et exposées sous
# des cibles ajp::dep::<nom>. Les modules ne font jamais de find_package.
#
# Deux sources possibles : le manifeste vcpkg (Windows), ou un préfixe construit par
# tools/deps/linux-from-source.sh (Linux, CI) ou macos-from-source.sh. Ce que le dépôt fixe lui-même y est aux
# mêmes références (tools/ci/check-deps-versions.sh le vérifie, y compris les "overrides" de vcpkg.json). Le reste ne l'est pas : sous Linux, dav1d,
# lcms2, shaderc, FriBidi, FreeType et HarfBuzz viennent de la distribution (Ubuntu 24.04 pour le paquet),
# sous Windows de la baseline de vcpkg, plus récente. Un écart mesuré entre les deux systèmes peut donc venir
# de ces bibliothèques (docs/audits/A-CONFIRMER.md, A27-04).
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
# Les minimums disent ce que le dépôt fixe (DEPS de tools/deps/linux-from-source.sh, overrides de vcpkg.json), pas
# « n'importe quelle version » (C27-03). tools/ci/check-deps-versions.sh recoupe ces nombres avec DEPS.
find_package(tomlplusplus 3.4 CONFIG REQUIRED)
get_target_property(ajp_toml_include tomlplusplus::tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
ajp_toml_fixed_include("${ajp_toml_include}" ajp_toml_fixed)
add_library(ajp_toml_headers INTERFACE)
target_include_directories(ajp_toml_headers SYSTEM INTERFACE "${ajp_toml_fixed}")
target_compile_definitions(ajp_toml_headers INTERFACE TOML_HEADER_ONLY=1
                                                      "TOML_CONFIG_HEADER=\"${CMAKE_CURRENT_LIST_DIR}/../src/core/toml_config.h\"")
ajp_alias_dep(toml ajp_toml_headers)

# zlib : le deflate des roues et des zip que le lecteur télécharge (core/inflate.h). Celui du système
# sous Linux et macOS, le port de vcpkg sous Windows.
find_package(ZLIB REQUIRED)
ajp_alias_dep(zlib ZLIB::ZLIB)

find_package(spdlog 1.17 CONFIG REQUIRED)
ajp_alias_dep(spdlog spdlog::spdlog)
target_compile_definitions(ajp_dep_spdlog INTERFACE SPDLOG_USE_STD_FORMAT)

# CUDA 13.4 (au moins ; la version exacte livrée est épinglée dans tools/release/versions.env), seulement avec AJP_WITH_CUDA (backend TensorRT et ses noyaux). Le runtime statique
# charge le pilote (libcuda) à l'exécution : sans pilote, un binaire démarre et refuse proprement.
if(AJP_WITH_CUDA)
  find_package(CUDAToolkit 13.4 REQUIRED)
  ajp_alias_dep(cudart CUDA::cudart_static)
  # NVTX3 (en-têtes seuls, kit CUDA) : plages de src/core/trace.h pour Nsight Systems.
  if(TARGET CUDA::nvtx3)
    get_target_property(AJP_NVTX_INCLUDES CUDA::nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
    if(AJP_NVTX_INCLUDES)
      include_directories(SYSTEM ${AJP_NVTX_INCLUDES})
      add_compile_definitions(AJP_WITH_NVTX)
    endif()
  endif()
endif()

if(NOT AJP_BUILD_PLAYER)
  return()
endif()

find_package(PkgConfig REQUIRED)

# Les bibliothèques d'image et de décodage que le dépôt ne fixe pas (dav1d, shaderc, lcms2, FreeType, HarfBuzz,
# FriBidi : celles de la distribution sous Linux, de la baseline de vcpkg sous Windows) : leurs versions à la
# construction entrent dans le rapport « Signaler un problème » (C27-04), pour qu'un écart de rendu ou de vitesse entre
# Windows et CachyOS se lise sans enquête.
set(ajp_lib_versions "")
foreach(ajp_lib IN ITEMS dav1d shaderc lcms2 freetype2 harfbuzz fribidi)
  pkg_check_modules(AJP_VERSION_${ajp_lib} QUIET ${ajp_lib})
  if(AJP_VERSION_${ajp_lib}_FOUND)
    list(APPEND ajp_lib_versions "${ajp_lib} ${AJP_VERSION_${ajp_lib}_VERSION}")
  endif()
endforeach()
list(JOIN ajp_lib_versions ", " AJP_SYSTEM_LIBS_TEXT)
message(STATUS "Bibliothèques du système : ${AJP_SYSTEM_LIBS_TEXT}")

# FFmpeg 9.0.2 (libavformat 63.1), LGPL partagée.
pkg_check_modules(AJP_FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=63.1 libavcodec>=63.1 libavfilter>=12.1 libavutil>=61.1 libswresample>=7.1 libswscale>=10.1)
ajp_alias_dep(ffmpeg PkgConfig::AJP_FFMPEG)

# Une dépendance que le dépôt corrige (libplacebo, libass) : son en-tête doit déclarer la fonction que le
# correctif ajoute. Sans ce contrôle, une bibliothèque officielle trouvée avant celle du préfixe passait la
# configuration et n'échouait qu'à l'édition de liens, loin de sa cause.
function(ajp_require_patched name include_dirs header symbol)
  find_file(AJP_${name}_HEADER NAMES ${header} HINTS ${include_dirs} NO_CACHE)
  if(AJP_${name}_HEADER)
    file(STRINGS "${AJP_${name}_HEADER}" found REGEX "${symbol}" LIMIT_COUNT 1)
  endif()
  if(NOT found)
    message(FATAL_ERROR
      "${name} : ${header} ne déclare pas ${symbol} (${AJP_${name}_HEADER}). Ce n'est pas la bibliothèque "
      "corrigée du dépôt : construire le préfixe avec tools/deps/linux-from-source.sh (ou "
      "macos-from-source.sh, ou les ports vcpkg de ports/) et le placer devant celui du système "
      "(AJP_DEPS_PREFIX, PKG_CONFIG_PATH).")
  endif()
endfunction()

# libplacebo v7.387.0 (overlay ports/libplacebo), avec le correctif de présentation.
pkg_check_modules(AJP_PLACEBO REQUIRED IMPORTED_TARGET libplacebo>=7.387)
ajp_require_patched(libplacebo "${AJP_PLACEBO_INCLUDE_DIRS}" libplacebo/vulkan.h pl_vulkan_swapchain_submit)
ajp_alias_dep(placebo PkgConfig::AJP_PLACEBO)

# libass 0.17.5 de l'amont plus le rendu sur plusieurs fils (overlay ports/libass, son correctif et son portfile).
pkg_check_modules(AJP_ASS REQUIRED IMPORTED_TARGET libass)
ajp_require_patched(libass "${AJP_ASS_INCLUDE_DIRS}" ass/ass.h ass_set_render_thread_count)
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

# wp_presentation (--present=wayland, docs/adr/0011) : wayland-client et wayland-scanner, facultatifs ; sans
# eux le niveau Wayland n'est jamais offert.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  pkg_check_modules(AJP_WAYLAND QUIET IMPORTED_TARGET wayland-client)
  find_program(AJP_WAYLAND_SCANNER NAMES wayland-scanner)
  if(AJP_WAYLAND_FOUND AND AJP_WAYLAND_SCANNER)
    enable_language(C)  # le code du protocole produit par wayland-scanner
    ajp_alias_dep(wayland PkgConfig::AJP_WAYLAND)
  endif()
endif()

# SDL 3.4.20 au moins : le HDR de Windows en dépend (host_facts.cpp) ; vcpkg.json et DEPS la fixent à 3.4.20.
find_package(SDL3 3.4.20 CONFIG REQUIRED)
message(STATUS "SDL3 ${SDL3_VERSION}, FFmpeg (libavformat ${AJP_FFMPEG_libavformat_VERSION}, libavcodec ${AJP_FFMPEG_libavcodec_VERSION})")

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

# Dear ImGui : sources installées par le script du préfixe (share/imgui), ou port vcpkg. Le préfixe passe d'abord
# (C27-03) : sans cela, un paquet imgui du système (CachyOS, Arch) qui installe un imguiConfig.cmake gagnait sur les
# sources du préfixe, avec une autre version et d'autres options de compilation (FreeType, backends).
set(AJP_IMGUI_VERSION_WANTED "1.93.0")  # DEPS de tools/deps/linux-from-source.sh et overrides de vcpkg.json
find_path(AJP_IMGUI_DIR imgui.cpp PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES share/imgui NO_DEFAULT_PATH)
if(NOT AJP_IMGUI_DIR)
  find_package(imgui CONFIG QUIET)
endif()
if(NOT AJP_IMGUI_DIR AND TARGET imgui::imgui)
  ajp_alias_dep(imgui imgui::imgui)
  # Le port avec sa fonction « freetype » (vcpkg.json) : le même chargeur de polices que sous Linux, pour les
  # polices CFF2 et variables que stb_truetype refuse.
  get_target_property(AJP_IMGUI_DEFINITIONS imgui::imgui INTERFACE_COMPILE_DEFINITIONS)
  find_path(AJP_IMGUI_FREETYPE_DIR imgui_freetype.h PATH_SUFFIXES misc/freetype)
  if("${AJP_IMGUI_DEFINITIONS}" MATCHES "IMGUI_ENABLE_FREETYPE" AND AJP_IMGUI_FREETYPE_DIR)
    target_include_directories(ajp_dep_imgui SYSTEM INTERFACE "${AJP_IMGUI_FREETYPE_DIR}")
    target_compile_definitions(ajp_dep_imgui INTERFACE AJP_WITH_FREETYPE=1)
  else()
    message(STATUS "ImGui sans FreeType : les polices CFF2 et variables seront refusées")
  endif()
else()
  find_path(AJP_IMGUI_DIR imgui.cpp PATH_SUFFIXES share/imgui REQUIRED)
  file(STRINGS "${AJP_IMGUI_DIR}/imgui.h" AJP_IMGUI_VERSION_LINE REGEX "^#define IMGUI_VERSION +\"" LIMIT_COUNT 1)
  if(NOT AJP_IMGUI_VERSION_LINE MATCHES "\"${AJP_IMGUI_VERSION_WANTED}\"")
    message(WARNING "Dear ImGui : ${AJP_IMGUI_DIR} est en ${AJP_IMGUI_VERSION_LINE}, le dépôt fixe ${AJP_IMGUI_VERSION_WANTED}")
  endif()
  message(STATUS "Dear ImGui : sources de ${AJP_IMGUI_DIR}")
  add_library(ajp_imgui STATIC
    "${AJP_IMGUI_DIR}/imgui.cpp" "${AJP_IMGUI_DIR}/imgui_draw.cpp"
    "${AJP_IMGUI_DIR}/imgui_tables.cpp" "${AJP_IMGUI_DIR}/imgui_widgets.cpp")
  # Aucun backend d'ImGui : le lecteur remplit ImGuiIO lui-même (app/play/ui.cpp) et dessine par libplacebo
  # (placebo_imgui.cpp).
  target_include_directories(ajp_imgui SYSTEM PUBLIC "${AJP_IMGUI_DIR}")
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

# TensorRT 11.3, seulement avec AJP_WITH_CUDA : en-têtes publics et libnvinfer. Sous Linux,
# AJP_TENSORRT_ROOT pointe un dossier include/ lib/ bin/ (sur le PC de référence ~/ajp-sdk/tensorrt :
# bibliothèques du paquet PyPI tensorrt-cu13-libs, en-têtes de NVIDIA/TensorRT, trtexec).
# libnvonnxparser et trtexec ne servent qu'aux tests, qui construisent leurs propres moteurs.
if(AJP_WITH_CUDA)
  set(AJP_TENSORRT_ROOT "" CACHE PATH "Racine de TensorRT 11 : include/, lib/ et bin/trtexec")
  find_path(AJP_TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${AJP_TENSORRT_ROOT}/include" REQUIRED)
  # La version des en-têtes : un TensorRT 10.x trouvé par ailleurs passerait la configuration et ne
  # casserait que plus loin (symbole absent, moteur refusé au chargement).
  file(STRINGS "${AJP_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" AJP_TRT_VERSION_LINES
       REGEX "^#define TRT_(MAJOR|MINOR)_ENTERPRISE [0-9]+")
  set(AJP_TRT_MAJOR "")
  set(AJP_TRT_MINOR "")
  foreach(line IN LISTS AJP_TRT_VERSION_LINES)
    if(line MATCHES "TRT_MAJOR_ENTERPRISE ([0-9]+)")
      set(AJP_TRT_MAJOR "${CMAKE_MATCH_1}")
    elseif(line MATCHES "TRT_MINOR_ENTERPRISE ([0-9]+)")
      set(AJP_TRT_MINOR "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(NOT AJP_TRT_MAJOR STREQUAL "11" OR NOT AJP_TRT_MINOR STREQUAL "3")
    message(FATAL_ERROR "TensorRT 11.3 exigé, les en-têtes de ${AJP_TENSORRT_INCLUDE_DIR} sont en "
                        "'${AJP_TRT_MAJOR}.${AJP_TRT_MINOR}' (AJP_TENSORRT_ROOT)")
  endif()
  message(STATUS "TensorRT ${AJP_TRT_MAJOR}.${AJP_TRT_MINOR} (${AJP_TENSORRT_INCLUDE_DIR})")
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
