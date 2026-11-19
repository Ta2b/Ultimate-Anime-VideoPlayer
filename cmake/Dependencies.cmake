# Toutes les dépendances tierces sont trouvées ici, une fois, et exposées sous
# des cibles kirari::dep::<nom>. Les modules ne font jamais de find_package.
#
# Deux sources possibles : le manifeste vcpkg (Windows), ou un préfixe construit par
# tools/deps/linux-from-source.sh (Linux, CI) ou macos-from-source.sh. Ce que le dépôt fixe lui-même y est aux
# mêmes références (tools/ci/check-deps-versions.sh le vérifie, y compris les "overrides" de vcpkg.json). Le reste ne l'est pas : sous Linux, dav1d,
# lcms2, shaderc, FriBidi, FreeType et HarfBuzz viennent de la distribution (Ubuntu 24.04 pour le paquet),
# sous Windows de la baseline de vcpkg, plus récente. Un écart mesuré entre les deux systèmes peut donc venir
# de ces bibliothèques (docs/audits/A-CONFIRMER.md, A27-04).
include_guard(GLOBAL)

function(kirari_alias_dep name target)
  add_library(kirari_dep_${name} INTERFACE)
  target_link_libraries(kirari_dep_${name} INTERFACE ${target})
  add_library(kirari::dep::${name} ALIAS kirari_dep_${name})
endfunction()

# toml++ en en-têtes seuls, partout : la bibliothèque compilée de vcpkg (Windows) n'exporte que
# l'API à exceptions (toml::ex), et kirari se construit avec TOML_EXCEPTIONS=0 (toml::noex). Ses
# en-têtes sont lus dans la copie corrigée de cmake/TomlFixes.cmake (<toml++-kirari/toml.hpp>), et
# src/core/toml_config.h complète sa configuration.
include("${CMAKE_CURRENT_LIST_DIR}/TomlFixes.cmake")
# Les minimums disent ce que le dépôt fixe (DEPS de tools/deps/linux-from-source.sh, overrides de vcpkg.json), pas
# « n'importe quelle version » (C27-03). tools/ci/check-deps-versions.sh recoupe ces nombres avec DEPS.
find_package(tomlplusplus 3.4 CONFIG REQUIRED)
get_target_property(kirari_toml_include tomlplusplus::tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
kirari_toml_fixed_include("${kirari_toml_include}" kirari_toml_fixed)
add_library(kirari_toml_headers INTERFACE)
target_include_directories(kirari_toml_headers SYSTEM INTERFACE "${kirari_toml_fixed}")
target_compile_definitions(kirari_toml_headers INTERFACE TOML_HEADER_ONLY=1
                                                      "TOML_CONFIG_HEADER=\"${CMAKE_CURRENT_LIST_DIR}/../src/core/toml_config.h\"")
kirari_alias_dep(toml kirari_toml_headers)

# zlib : le deflate des roues et des zip que le lecteur télécharge (core/inflate.h). Celui du système
# sous Linux et macOS, le port de vcpkg sous Windows.
find_package(ZLIB REQUIRED)
kirari_alias_dep(zlib ZLIB::ZLIB)

find_package(spdlog 1.17 CONFIG REQUIRED)
kirari_alias_dep(spdlog spdlog::spdlog)
target_compile_definitions(kirari_dep_spdlog INTERFACE SPDLOG_USE_STD_FORMAT)

# CUDA 13.4 (au moins ; la version exacte livrée est épinglée dans tools/release/versions.env), seulement avec KIRARI_WITH_CUDA (backend TensorRT et ses noyaux). Le runtime statique
# charge le pilote (libcuda) à l'exécution : sans pilote, un binaire démarre et refuse proprement.
if(KIRARI_WITH_CUDA)
  find_package(CUDAToolkit 13.4 REQUIRED)
  kirari_alias_dep(cudart CUDA::cudart_static)
  # NVTX3 (en-têtes seuls, kit CUDA) : plages de src/core/trace.h pour Nsight Systems.
  if(TARGET CUDA::nvtx3)
    get_target_property(KIRARI_NVTX_INCLUDES CUDA::nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
    if(KIRARI_NVTX_INCLUDES)
      include_directories(SYSTEM ${KIRARI_NVTX_INCLUDES})
      add_compile_definitions(KIRARI_WITH_NVTX)
    endif()
  endif()
endif()

if(NOT KIRARI_BUILD_PLAYER)
  return()
endif()

find_package(PkgConfig REQUIRED)

# Les bibliothèques d'image et de décodage que le dépôt ne fixe pas (dav1d, shaderc, lcms2, FreeType, HarfBuzz,
# FriBidi : celles de la distribution sous Linux, de la baseline de vcpkg sous Windows) : leurs versions à la
# construction entrent dans le rapport « Signaler un problème » (C27-04), pour qu'un écart de rendu ou de vitesse entre
# Windows et CachyOS se lise sans enquête.
set(kirari_lib_versions "")
foreach(kirari_lib IN ITEMS dav1d shaderc lcms2 freetype2 harfbuzz fribidi)
  pkg_check_modules(KIRARI_VERSION_${kirari_lib} QUIET ${kirari_lib})
  if(KIRARI_VERSION_${kirari_lib}_FOUND)
    list(APPEND kirari_lib_versions "${kirari_lib} ${KIRARI_VERSION_${kirari_lib}_VERSION}")
  endif()
endforeach()
list(JOIN kirari_lib_versions ", " KIRARI_SYSTEM_LIBS_TEXT)
message(STATUS "Bibliothèques du système : ${KIRARI_SYSTEM_LIBS_TEXT}")

# FFmpeg 9.0.2 (libavformat 63.1), LGPL partagée.
pkg_check_modules(KIRARI_FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=63.1 libavcodec>=63.1 libavfilter>=12.1 libavutil>=61.1 libswresample>=7.1 libswscale>=10.1)
kirari_alias_dep(ffmpeg PkgConfig::KIRARI_FFMPEG)

# Une dépendance que le dépôt corrige (libplacebo, libass) : son en-tête doit déclarer la fonction que le
# correctif ajoute. Sans ce contrôle, une bibliothèque officielle trouvée avant celle du préfixe passait la
# configuration et n'échouait qu'à l'édition de liens, loin de sa cause.
function(kirari_require_patched name include_dirs header symbol)
  find_file(KIRARI_${name}_HEADER NAMES ${header} HINTS ${include_dirs} NO_CACHE)
  if(KIRARI_${name}_HEADER)
    file(STRINGS "${KIRARI_${name}_HEADER}" found REGEX "${symbol}" LIMIT_COUNT 1)
  endif()
  if(NOT found)
    message(FATAL_ERROR
      "${name} : ${header} ne déclare pas ${symbol} (${KIRARI_${name}_HEADER}). Ce n'est pas la bibliothèque "
      "corrigée du dépôt : construire le préfixe avec tools/deps/linux-from-source.sh (ou "
      "macos-from-source.sh, ou les ports vcpkg de ports/) et le placer devant celui du système "
      "(KIRARI_DEPS_PREFIX, PKG_CONFIG_PATH).")
  endif()
endfunction()

# libplacebo v7.387.0 (overlay ports/libplacebo), avec le correctif de présentation.
pkg_check_modules(KIRARI_PLACEBO REQUIRED IMPORTED_TARGET libplacebo>=7.387)
kirari_require_patched(libplacebo "${KIRARI_PLACEBO_INCLUDE_DIRS}" libplacebo/vulkan.h pl_vulkan_swapchain_submit)
kirari_alias_dep(placebo PkgConfig::KIRARI_PLACEBO)

# libass 0.17.5 de l'amont plus le rendu sur plusieurs fils (overlay ports/libass, son correctif et son portfile).
pkg_check_modules(KIRARI_ASS REQUIRED IMPORTED_TARGET libass)
kirari_require_patched(libass "${KIRARI_ASS_INCLUDE_DIRS}" ass/ass.h ass_set_render_thread_count)
kirari_alias_dep(ass PkgConfig::KIRARI_ASS)

# fontconfig, facultatif : la police du bureau pour l'interface (app/play/ui.cpp).
pkg_check_modules(KIRARI_FONTCONFIG QUIET IMPORTED_TARGET fontconfig)
if(KIRARI_FONTCONFIG_FOUND)
  kirari_alias_dep(fontconfig PkgConfig::KIRARI_FONTCONFIG)
endif()

# libva, facultative (Linux) : choisir le nœud de rendu VA-API qui décode le codec
# (media/vaapi_device.cpp). Sans elle, VA-API prend l'appareil par défaut.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  pkg_check_modules(KIRARI_VA QUIET IMPORTED_TARGET libva)
  if(KIRARI_VA_FOUND)
    kirari_alias_dep(va PkgConfig::KIRARI_VA)
  endif()
endif()

# wp_presentation (--present=wayland, docs/adr/0011) : wayland-client et wayland-scanner, facultatifs ; sans
# eux le niveau Wayland n'est jamais offert.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  pkg_check_modules(KIRARI_WAYLAND QUIET IMPORTED_TARGET wayland-client)
  find_program(KIRARI_WAYLAND_SCANNER NAMES wayland-scanner)
  if(KIRARI_WAYLAND_FOUND AND KIRARI_WAYLAND_SCANNER)
    enable_language(C)  # le code du protocole produit par wayland-scanner
    kirari_alias_dep(wayland PkgConfig::KIRARI_WAYLAND)
  endif()
endif()

# SDL 3.4.20 au moins : le HDR de Windows en dépend (host_facts.cpp) ; vcpkg.json et DEPS la fixent à 3.4.20.
find_package(SDL3 3.4.20 CONFIG REQUIRED)
message(STATUS "SDL3 ${SDL3_VERSION}, FFmpeg (libavformat ${KIRARI_FFMPEG_libavformat_VERSION}, libavcodec ${KIRARI_FFMPEG_libavcodec_VERSION})")

kirari_alias_dep(sdl3 SDL3::SDL3)

find_package(VulkanHeaders CONFIG QUIET)
if(TARGET Vulkan::Headers)
  kirari_alias_dep(vulkan Vulkan::Headers)
else()
  find_package(Vulkan REQUIRED)
  kirari_alias_dep(vulkan Vulkan::Headers)
endif()

# Compilateur GLSL vers SPIR-V, pour les shaders compilés hors ligne.
# Sans SDK Vulkan (Windows, vcpkg) : le glslc du port shaderc, outil de l'hôte.
find_program(KIRARI_GLSLC NAMES glslc
  HINTS "$ENV{VULKAN_SDK}/bin" "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/shaderc" REQUIRED)

# Dear ImGui : sources installées par le script du préfixe (share/imgui), ou port vcpkg. Le préfixe passe d'abord
# (C27-03) : sans cela, un paquet imgui du système (CachyOS, Arch) qui installe un imguiConfig.cmake gagnait sur les
# sources du préfixe, avec une autre version et d'autres options de compilation (FreeType, backends).
set(KIRARI_IMGUI_VERSION_WANTED "1.93.0")  # DEPS de tools/deps/linux-from-source.sh et overrides de vcpkg.json
find_path(KIRARI_IMGUI_DIR imgui.cpp PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES share/imgui NO_DEFAULT_PATH)
if(NOT KIRARI_IMGUI_DIR)
  find_package(imgui CONFIG QUIET)
endif()
if(NOT KIRARI_IMGUI_DIR AND TARGET imgui::imgui)
  kirari_alias_dep(imgui imgui::imgui)
  # Le port avec sa fonction « freetype » (vcpkg.json) : le même chargeur de polices que sous Linux, pour les
  # polices CFF2 et variables que stb_truetype refuse.
  get_target_property(KIRARI_IMGUI_DEFINITIONS imgui::imgui INTERFACE_COMPILE_DEFINITIONS)
  find_path(KIRARI_IMGUI_FREETYPE_DIR imgui_freetype.h PATH_SUFFIXES misc/freetype)
  if("${KIRARI_IMGUI_DEFINITIONS}" MATCHES "IMGUI_ENABLE_FREETYPE" AND KIRARI_IMGUI_FREETYPE_DIR)
    target_include_directories(kirari_dep_imgui SYSTEM INTERFACE "${KIRARI_IMGUI_FREETYPE_DIR}")
    target_compile_definitions(kirari_dep_imgui INTERFACE KIRARI_WITH_FREETYPE=1)
  else()
    message(STATUS "ImGui sans FreeType : les polices CFF2 et variables seront refusées")
  endif()
else()
  find_path(KIRARI_IMGUI_DIR imgui.cpp PATH_SUFFIXES share/imgui REQUIRED)
  file(STRINGS "${KIRARI_IMGUI_DIR}/imgui.h" KIRARI_IMGUI_VERSION_LINE REGEX "^#define IMGUI_VERSION +\"" LIMIT_COUNT 1)
  if(NOT KIRARI_IMGUI_VERSION_LINE MATCHES "\"${KIRARI_IMGUI_VERSION_WANTED}\"")
    message(WARNING "Dear ImGui : ${KIRARI_IMGUI_DIR} est en ${KIRARI_IMGUI_VERSION_LINE}, le dépôt fixe ${KIRARI_IMGUI_VERSION_WANTED}")
  endif()
  message(STATUS "Dear ImGui : sources de ${KIRARI_IMGUI_DIR}")
  add_library(kirari_imgui STATIC
    "${KIRARI_IMGUI_DIR}/imgui.cpp" "${KIRARI_IMGUI_DIR}/imgui_draw.cpp"
    "${KIRARI_IMGUI_DIR}/imgui_tables.cpp" "${KIRARI_IMGUI_DIR}/imgui_widgets.cpp")
  # Aucun backend d'ImGui : le lecteur remplit ImGuiIO lui-même (app/play/ui.cpp) et dessine par libplacebo
  # (placebo_imgui.cpp).
  target_include_directories(kirari_imgui SYSTEM PUBLIC "${KIRARI_IMGUI_DIR}")
  # FreeType pour les polices : stb_truetype ne lit ni le CFF2 (Noto Sans CJK variable, installé par défaut sous Fedora) ni
  # les polices variables ; sans ce chargeur le japonais n'apparaît pas. Facultatif : sans FreeType ou sans les sources du
  # chargeur, ImGui garde stb_truetype.
  pkg_check_modules(KIRARI_FREETYPE QUIET IMPORTED_TARGET freetype2)
  if(KIRARI_FREETYPE_FOUND AND EXISTS "${KIRARI_IMGUI_DIR}/misc/freetype/imgui_freetype.cpp")
    target_sources(kirari_imgui PRIVATE "${KIRARI_IMGUI_DIR}/misc/freetype/imgui_freetype.cpp")
    target_include_directories(kirari_imgui SYSTEM PUBLIC "${KIRARI_IMGUI_DIR}/misc/freetype")
    target_link_libraries(kirari_imgui PUBLIC PkgConfig::KIRARI_FREETYPE)
    target_compile_definitions(kirari_imgui PUBLIC KIRARI_WITH_FREETYPE=1)
  endif()
  kirari_alias_dep(imgui kirari_imgui)
endif()

# TensorRT 11.3, seulement avec KIRARI_WITH_CUDA : en-têtes publics et libnvinfer. Sous Linux,
# KIRARI_TENSORRT_ROOT pointe un dossier include/ lib/ bin/ (sur le PC de référence ~/kirari-sdk/tensorrt :
# bibliothèques du paquet PyPI tensorrt-cu13-libs, en-têtes de NVIDIA/TensorRT, trtexec).
# libnvonnxparser et trtexec ne servent qu'aux tests, qui construisent leurs propres moteurs.
if(KIRARI_WITH_CUDA)
  set(KIRARI_TENSORRT_ROOT "" CACHE PATH "Racine de TensorRT 11 : include/, lib/ et bin/trtexec")
  find_path(KIRARI_TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${KIRARI_TENSORRT_ROOT}/include" REQUIRED)
  # La version des en-têtes : un TensorRT 10.x trouvé par ailleurs passerait la configuration et ne
  # casserait que plus loin (symbole absent, moteur refusé au chargement).
  file(STRINGS "${KIRARI_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" KIRARI_TRT_VERSION_LINES
       REGEX "^#define TRT_(MAJOR|MINOR)_ENTERPRISE [0-9]+")
  set(KIRARI_TRT_MAJOR "")
  set(KIRARI_TRT_MINOR "")
  foreach(line IN LISTS KIRARI_TRT_VERSION_LINES)
    if(line MATCHES "TRT_MAJOR_ENTERPRISE ([0-9]+)")
      set(KIRARI_TRT_MAJOR "${CMAKE_MATCH_1}")
    elseif(line MATCHES "TRT_MINOR_ENTERPRISE ([0-9]+)")
      set(KIRARI_TRT_MINOR "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(NOT KIRARI_TRT_MAJOR STREQUAL "11" OR NOT KIRARI_TRT_MINOR STREQUAL "3")
    message(FATAL_ERROR "TensorRT 11.3 exigé, les en-têtes de ${KIRARI_TENSORRT_INCLUDE_DIR} sont en "
                        "'${KIRARI_TRT_MAJOR}.${KIRARI_TRT_MINOR}' (KIRARI_TENSORRT_ROOT)")
  endif()
  message(STATUS "TensorRT ${KIRARI_TRT_MAJOR}.${KIRARI_TRT_MINOR} (${KIRARI_TENSORRT_INCLUDE_DIR})")
  # Sous Windows, les bibliothèques d'import portent la version (nvinfer_11.lib).
  find_library(KIRARI_NVINFER_LIBRARY NAMES nvinfer nvinfer_11 HINTS "${KIRARI_TENSORRT_ROOT}/lib" REQUIRED)
  find_library(KIRARI_NVONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_11 HINTS "${KIRARI_TENSORRT_ROOT}/lib"
               REQUIRED)
  find_program(KIRARI_TRTEXEC trtexec HINTS "${KIRARI_TENSORRT_ROOT}/bin")

  add_library(kirari_dep_tensorrt INTERFACE)
  target_include_directories(kirari_dep_tensorrt SYSTEM INTERFACE "${KIRARI_TENSORRT_INCLUDE_DIR}")
  target_link_libraries(kirari_dep_tensorrt INTERFACE "${KIRARI_NVINFER_LIBRARY}" kirari::dep::cudart)
  add_library(kirari::dep::tensorrt ALIAS kirari_dep_tensorrt)

  kirari_alias_dep(tensorrt_onnx "${KIRARI_NVONNXPARSER_LIBRARY}")
  target_link_libraries(kirari_dep_tensorrt_onnx INTERFACE kirari::dep::tensorrt)
endif()
