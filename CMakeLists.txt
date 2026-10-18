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
option(AJP_BUILD_TESTS "Construire les tests unitaires" ON)
option(AJP_BUILD_PLAYER "Construire le lecteur ; OFF ne garde que core, kernels et leurs tests" ON)
option(AJP_WARNINGS_AS_ERRORS "Avertissements en erreurs" ON)

if(AJP_WITH_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
      CACHE STRING "Liste des RTX, reprise du fork" FORCE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Les modules C++ sont interdits : pas de balayage, ni d'options .modmap dans
# compile_commands.json (clang-tidy ne sait pas les lire).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AjpModule)
include(Dependencies)

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
