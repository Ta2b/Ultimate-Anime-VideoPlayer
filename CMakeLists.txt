cmake_minimum_required(VERSION 3.28)

project(ajp
  VERSION 0.1.0
  DESCRIPTION "Lecteur vidéo anime : upscaling, interpolation et HDR en temps réel"
  LANGUAGES CXX)

# CUDA n'entre qu'avec le backend TensorRT (M1 sur le PC de référence).
# Les architectures se passent toujours en entier : CMake prédéfinit
# CMAKE_CUDA_ARCHITECTURES pendant enable_language(), donc un repli
# « if(NOT DEFINED) » ne s'exécuterait jamais.
option(AJP_WITH_CUDA "Construire le backend TensorRT et les noyaux CUDA" OFF)
option(AJP_WITH_MIGRAPHX "Construire ajp_backend_migraphx (réseaux par MIGraphX, GPU AMD ; ROCm)" OFF)
option(AJP_WITH_OPENVINO "Construire ajp_backend_openvino (réseaux par OpenVINO, GPU Intel)" OFF)
option(AJP_WITH_COREML "Construire ajp_backend_coreml (réseaux par Core ML via ONNX Runtime, Mac Apple Silicon)" OFF)
option(AJP_BUILD_TESTS "Construire les tests unitaires" ON)
option(AJP_BUILD_PLAYER "Construire le lecteur ; OFF ne garde que core, kernels et leurs tests" ON)
# Avertissements en erreurs : ON dans les préréglages de développement et de CI (base, prefix), OFF
# par défaut, pour qu'un compilateur plus récent (avertissements nouveaux) construise encore.
option(AJP_WARNINGS_AS_ERRORS "Avertissements en erreurs" OFF)
option(AJP_TRANSITIVE_RPATH "RPATH valable pour les dépendances indirectes (préfixe Linux)" OFF)

if(AJP_WITH_CUDA)
  # Livraison : la liste du plan, celle du fork réduite aux RTX (Turing à Blackwell, plus le
  # PTX 120 pour les suivantes). Développement local sur la 4070 Ti : AJP_CUDA_ARCHITECTURES
  # vaut 89-real (preset linux-prefix-cuda), une seule cible, bien plus rapide à compiler.
  set(AJP_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
      CACHE STRING "Architectures CUDA, passées en entier à CMAKE_CUDA_ARCHITECTURES")
  set(CMAKE_CUDA_ARCHITECTURES "${AJP_CUDA_ARCHITECTURES}" CACHE STRING "Reprise de AJP_CUDA_ARCHITECTURES" FORCE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# windows.h fait de min et max des macros, qui cassent std::max et std::numeric_limits<T>::min
# dans tout fichier qui l'inclut, même par un en-tête tiers.
if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Les modules C++ sont interdits : pas de balayage, ni d'options .modmap dans
# compile_commands.json (clang-tidy ne sait pas les lire).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AjpModule)
include(Dependencies)

# Les bibliothèques d'un préfixe comme /opt/ajp-deps n'ont pas toutes de RUNPATH :
# libavcodec n'y trouverait pas libswresample. Un DT_RPATH, à la différence de
# DT_RUNPATH, vaut aussi pour les dépendances indirectes : chaque exécutable et
# bibliothèque partagée se lance alors sans LD_LIBRARY_PATH. Posé avant toute cible.
if(AJP_TRANSITIVE_RPATH AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_link_options("LINKER:--disable-new-dtags")
endif()

# La libc++ de Xcode 16 (LLVM 19) range std::jthread et <stop_token> parmi ses fonctions
# expérimentales : entièrement dans ses en-têtes, elles s'ouvrent par cette macro, sans
# -fexperimental-library ni libc++experimental.a. Inoffensive sur une libc++ plus récente.
if(APPLE)
  add_compile_definitions(_LIBCPP_ENABLE_EXPERIMENTAL)
endif()

# Ordre des feuilles vers la racine : une cible ne dépend que de ce qui précède.
add_subdirectory(src/core)
add_subdirectory(src/kernels)
if(AJP_BUILD_PLAYER)
  add_subdirectory(src/media)
  add_subdirectory(src/subs)
  add_subdirectory(src/vendor_hdr)
  add_subdirectory(src/gpu)
  add_subdirectory(src/prepare)
  add_subdirectory(src/audio)
  add_subdirectory(src/render)
  add_subdirectory(src/present)
  add_subdirectory(src/export)
  add_subdirectory(src/app)
endif()

if(AJP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
