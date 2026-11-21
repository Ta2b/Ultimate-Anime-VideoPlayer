# kirari_backend_tensorrt.so : le backend TensorRT (gpu/backend.h), chargé à l'exécution.
# Il emporte prepare (constructeur trtexec) et sa propre copie de spdlog ; ses symboles
# restent cachés, sauf les trois points d'entrée.
add_library(kirari_backend_tensorrt_kernels STATIC deband_kernels.cu doubt_kernels.cu drba_fused_kernels.cu drba_tail_native.cu export_kernels.cu
  gmflow_attention_net.cu gmflow_backbone_net.cu gmflow_convex_kernels.cu gmflow_convex_net.cu gmflow_global_kernels.cu gmflow_local_kernels.cu
  gmflow_mlp_net.cu gmflow_scale_kernels.cu gmfss_conv_plan.cu gmfss_conv_tune.cu gmfss_kernels.cu gmfss_splat_indexed.cu
  gmfss_ifnet_native.cu gmfss_native_net.cu gmfss_pair_flow_glue.cu gmfss_pair_native_net.cu layer_kernels.cu rife_fused_kernels.cu rife_kernels.cu
  rife_native_bench.cu rife_native_finish.cu rife_native_glue.cu rife_native_net.cu rife_native_pick.cu
  rife_native_weights.cu scene_camera_kernels.cu scene_flow_kernels.cu scene_roundtrip_kernels.cu slice_kernels.cu stats_kernels.cu
  tfix_blend_kernels.cu tfix_fused_kernels.cu tfix_head.cu tfix_native_net.cu warp_kernels.cu warp_lib.cu)
target_include_directories(kirari_backend_tensorrt_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(kirari_backend_tensorrt_kernels PRIVATE kirari::dep::cudart)
target_sources(kirari_backend_tensorrt_kernels PRIVATE scene_glow_kernels.cu scene_glow_anchors.cu)
set_source_files_properties(scene_glow_kernels.cu scene_glow_anchors.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
# -lineinfo pour ncu ; ptxas donne les registres de chaque noyau au journal de build, et un
# débordement de registres en mémoire locale est une erreur (pixel_kernels.cuh) sous Linux, où le
# PC de référence et la CI le vérifient ; le code de l'appareil est le même sous Windows, où les
# en-têtes de MSVC que lit nvcc ne doivent pas arrêter la release.
# Les avertissements du compilateur hôte : GCC et Clang sous Linux, MSVC sous Windows.
target_compile_options(kirari_backend_tensorrt_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ptxas-options=--verbose,--warn-on-spills>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/W3>)
set_target_properties(kirari_backend_tensorrt_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CUTLASS FP8 reste activé par défaut sous Linux avec ses en-têtes intégrés. Windows le construit à la demande,
# avec le préprocesseur conforme de MSVC. L'option de paquet Windows reste à OFF (docs/span-fp8-cutlass.md).
set(KIRARI_CUTLASS_INCLUDE "${PROJECT_SOURCE_DIR}/third_party/cutlass/include")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${KIRARI_CUTLASS_INCLUDE}/cutlass/cutlass.h")
  set(kirari_span_fp8_default ON)
else()
  set(kirari_span_fp8_default OFF)
endif()
option(KIRARI_WITH_SPAN_FP8 "Construire les noyaux CUTLASS du SPAN entier en FP8 (sm 8.9)" ${kirari_span_fp8_default})
# AUTO suit le choix FP8 à chaque configuration, y compris lorsque la CI le reconfigure dans le même build.
# ON et OFF permettent d'isoler le FP16 natif et le FP8. Le paquet Windows garde ses deux voies à OFF.
set(KIRARI_WITH_SPAN_FP16 AUTO CACHE STRING "Noyaux SPAN FP16 natifs : AUTO, ON ou OFF")
set_property(CACHE KIRARI_WITH_SPAN_FP16 PROPERTY STRINGS AUTO ON OFF)
if(KIRARI_WITH_SPAN_FP16 STREQUAL "AUTO")
  set(kirari_span_fp16 ${KIRARI_WITH_SPAN_FP8})
elseif(KIRARI_WITH_SPAN_FP16 STREQUAL "ON" OR KIRARI_WITH_SPAN_FP16 STREQUAL "OFF")
  set(kirari_span_fp16 ${KIRARI_WITH_SPAN_FP16})
else()
  message(FATAL_ERROR "KIRARI_WITH_SPAN_FP16 attend AUTO, ON ou OFF")
endif()

# Les cibles SPAN spécialisées gardent Ada et Blackwell grand public, avec le PTX de 120 pour le chargement
# sur une carte plus récente. Les capacités à l'exécution restent celles de SpanFp8Net.
function(kirari_span_kernel_target target)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PRIVATE kirari::gpu_span_fp8 kirari::dep::cudart)
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "89-real;120" POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcudafe=--diag_suppress=177
      -Xcudafe=--diag_suppress=550 --ptxas-options=--verbose,--warn-on-spills>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra>
    # CCCL de CUDA 13 exige le préprocesseur conforme de MSVC.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>)
endfunction()

add_library(kirari_backend_span_net STATIC)
target_include_directories(kirari_backend_span_net PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(kirari_backend_span_net PRIVATE kirari::gpu_span_fp8 kirari::dep::cudart)
set_target_properties(kirari_backend_span_net PROPERTIES POSITION_INDEPENDENT_CODE ON)
kirari_set_warnings(kirari_backend_span_net)
if(KIRARI_WITH_SPAN_FP8 OR kirari_span_fp16)
  target_sources(kirari_backend_span_net PRIVATE span_fp8_net.cpp)
  target_compile_definitions(kirari_backend_span_net PRIVATE
    KIRARI_SPAN_FP8_KERNELS=$<BOOL:${KIRARI_WITH_SPAN_FP8}> KIRARI_SPAN_FP16_KERNELS=$<BOOL:${kirari_span_fp16}>)
else()
  target_sources(kirari_backend_span_net PRIVATE span_fp8_net_none.cpp)
endif()
# La référence sert seulement à la dérivation FP8. nvcc 13.4 s'y arrête avec MSVC ; sa version vide la remplace.
if(NOT KIRARI_WITH_SPAN_FP8 OR MSVC)
  target_sources(kirari_backend_span_net PRIVATE span_fp8_ref_none.cpp)
endif()

if(KIRARI_WITH_SPAN_FP8)
  add_library(kirari_backend_span_fp8_kernels STATIC span_fp8_net.cu span_fp8_net48.cu)
  if(NOT MSVC)
    target_sources(kirari_backend_span_fp8_kernels PRIVATE span_fp8_ref.cu)
  endif()
  target_include_directories(kirari_backend_span_fp8_kernels SYSTEM PRIVATE "${KIRARI_CUTLASS_INCLUDE}")
  kirari_span_kernel_target(kirari_backend_span_fp8_kernels)
  target_link_libraries(kirari_backend_span_net PRIVATE kirari_backend_span_fp8_kernels)
endif()
if(kirari_span_fp16)
  add_library(kirari_backend_span_fp16_kernels STATIC span_native16.cu)
  kirari_span_kernel_target(kirari_backend_span_fp16_kernels)
  target_link_libraries(kirari_backend_span_net PRIVATE kirari_backend_span_fp16_kernels)
endif()

add_library(kirari_backend_tensorrt MODULE
  drba_fused_engine.cpp drba_fused_native.cpp entry.cpp fused_graphs.cpp gmflow_pair_flow_net.cpp gmflow_transformer_net.cpp gmfss_fused_engine.cpp gmfss_fused_trial.cpp gmfss_pair_engine.cpp gmfss_pair_trial.cpp gmfss_trial.cpp gpu_timing.cpp host_upload.cpp ring_slot.cpp rife_fused_engine.cpp rife_fused_native.cpp
  rife_native_mix.cpp rife_native_trial.cpp
  ingest_chroma.cpp nvof_session.cpp scene_camera.cpp scene_flow.cpp scene_roundtrip.cpp shared_activations.cpp span_fp8_check.cpp span_fp8_engine.cpp span_fp8_tiles.cpp tensorrt_backend.cpp tensorrt_doubt.cpp tensorrt_export.cpp tensorrt_kernels.cpp
  tensorrt_gmfss.cpp tensorrt_gmfss_fused.cpp tensorrt_gmfss_pair.cpp tensorrt_layers.cpp tensorrt_lanes.cpp tensorrt_drba_fused.cpp tensorrt_rife.cpp tensorrt_rife_fused.cpp tensorrt_scene_flow.cpp tensorrt_span_derive.cpp
  span_timing.cpp tensorrt_span_fp8.cpp tensorrt_stats.cpp tensorrt_textures.cpp tensorrt_tfix_fused.cpp tensorrt_warp.cpp tfix_fused_engine.cpp tfix_fused_native.cpp
  trt_engine.cpp)
target_include_directories(kirari_backend_tensorrt PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_sources(kirari_backend_tensorrt PRIVATE scene_glow.cpp)
# En-têtes BSD de NVIDIA Optical Flow (nvof_session.cpp) : types et signatures seulement, la bibliothèque vient
# du pilote ; aucun fichier de NVIDIA n'est livré.
target_include_directories(kirari_backend_tensorrt SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nvidia-optical-flow")
target_link_libraries(kirari_backend_tensorrt PRIVATE
  kirari::gpu kirari::gpu_span_fp8 kirari::cuda_fused_keys kirari::cuda_native_mix kirari::cuda_rife_mix kirari::prepare kirari::core_log kirari::dep::spdlog kirari_backend_tensorrt_kernels kirari_backend_span_net
  kirari::dep::tensorrt CUDA::cuda_driver ${CMAKE_DL_LIBS})
# Le trtexec du paquet est celui de la configuration : son SHA-256 entre dans le module, qui refuse
# tout autre fichier posé à côté de lui (P1.21). Le repli sur le trtexec de la machine est réservé
# aux builds de développement.
set(kirari_trtexec_sha256 "")
set(kirari_trtexec_path "")
if(KIRARI_TRTEXEC AND EXISTS "${KIRARI_TRTEXEC}")
  set(kirari_trtexec_path "${KIRARI_TRTEXEC}")
  set(kirari_trtexec_packaged "${KIRARI_TRTEXEC}")
  # Un trtexec remplacé sur place (TensorRT mis à jour dans le même dossier) refait la configuration : sinon le
  # module garderait l'ancienne empreinte et refuserait le trtexec livré (C01-08).
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${KIRARI_TRTEXEC}")
  # Sous Linux, le paquet range trtexec sur $ORIGIN (ses bibliothèques de NVIDIA sont à côté de lui) : la copie
  # rangée est faite ici, une fois, et c'est son SHA-256 qui entre dans le module ; package-linux.sh livre cette
  # copie telle quelle (KIRARI_TRTEXEC_PACKAGED). Patcher après coup changeait l'empreinte (0.10.0).
  find_program(KIRARI_PATCHELF patchelf)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND KIRARI_PATCHELF)
    set(kirari_trtexec_packaged "${CMAKE_BINARY_DIR}/trtexec-paquet/trtexec")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/trtexec-paquet")
    file(COPY_FILE "${KIRARI_TRTEXEC}" "${kirari_trtexec_packaged}")
    execute_process(COMMAND "${KIRARI_PATCHELF}" --set-rpath "$ORIGIN" "${kirari_trtexec_packaged}"
                    RESULT_VARIABLE kirari_patchelf_result)
    if(NOT kirari_patchelf_result EQUAL 0)
      message(FATAL_ERROR "patchelf n'a pas rangé ${kirari_trtexec_packaged} sur $ORIGIN")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "patchelf manquant : le trtexec livré ne trouvera pas ses bibliothèques à côté de lui")
  endif()
  set(KIRARI_TRTEXEC_PACKAGED "${kirari_trtexec_packaged}" CACHE INTERNAL "trtexec livré, celui dont le module garde l'empreinte")
  file(SHA256 "${kirari_trtexec_packaged}" kirari_trtexec_sha256)
endif()
target_compile_definitions(kirari_backend_tensorrt PRIVATE KIRARI_TRTEXEC_SHA256="${kirari_trtexec_sha256}"
  KIRARI_TRTEXEC_PATH="${kirari_trtexec_path}" $<$<CONFIG:Release>:KIRARI_TRTEXEC_NO_FALLBACK>)
# L'empreinte des sources du RIFE natif (noyaux, réseau, essai) entre dans l'identité du verdict gardé de son
# essai (rife_native_verdict.h) : des noyaux changés refont l'essai. Une source modifiée refait la configuration.
file(GLOB kirari_rife_native_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rife_*"
  "${CMAKE_CURRENT_SOURCE_DIR}/warp_lib*")
list(SORT kirari_rife_native_files)
set(kirari_rife_native_digest "")
foreach(file ${kirari_rife_native_files})
  file(SHA256 "${file}" kirari_file_digest)
  string(APPEND kirari_rife_native_digest "${kirari_file_digest}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
endforeach()
string(SHA256 kirari_rife_native_digest "${kirari_rife_native_digest}")
string(SUBSTRING "${kirari_rife_native_digest}" 0 16 kirari_rife_native_digest)
set_property(SOURCE tensorrt_rife_fused.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS KIRARI_RIFE_NATIVE_SOURCES="${kirari_rife_native_digest}")
set_target_properties(kirari_backend_tensorrt PROPERTIES
  PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(kirari_backend_tensorrt PRIVATE LINKER:--exclude-libs,ALL LINKER:--no-undefined)
endif()
kirari_set_warnings(kirari_backend_tensorrt)

# Les bibliothèques statiques liées dans un objet partagé doivent être compilées en PIC. prepare
# et les modules qu'il tire sont déclarés après gpu : la propriété se pose à la fin du projet.
function(kirari_backend_tensorrt_pic)
  foreach(lib kirari_prepare kirari_core kirari_core_log kirari_kernels kirari_gpu_span_fp8 kirari_cuda_fused_keys
          kirari_cuda_native_mix kirari_cuda_rife_mix)
    if(TARGET ${lib})
      set_property(TARGET ${lib} PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL kirari_backend_tensorrt_pic)

# Outil de contrôle du SPAN FP8 : tools/span/derive_span_fp8.py y passe les images de validation et chronomètre.
if(KIRARI_WITH_SPAN_FP8)
  add_executable(kirari_span_fp8_run span_fp8_run.cpp)
  target_include_directories(kirari_span_fp8_run PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(kirari_span_fp8_run PRIVATE kirari::gpu_span_fp8 kirari::gpu_span_fp8_onnx kirari_backend_span_net
    kirari::dep::cudart)
  kirari_set_warnings(kirari_span_fp8_run)
  # Le banc des formes de tuile (span_tile_bench.cu) : des formes que le lecteur ne prend pas, dont certaines
  # débordent de registres ; ptxas le dit (--warn-on-spills) sans arrêter la construction.
  add_executable(kirari_span_tile_bench span_tile_bench.cu)
  target_include_directories(kirari_span_tile_bench PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(kirari_span_tile_bench SYSTEM PRIVATE "${KIRARI_CUTLASS_INCLUDE}")
  target_link_libraries(kirari_span_tile_bench PRIVATE kirari::dep::cudart)
  set_target_properties(kirari_span_tile_bench PROPERTIES CUDA_ARCHITECTURES "89-real;120")
  target_compile_options(kirari_span_tile_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcudafe=--diag_suppress=177
      -Xcudafe=--diag_suppress=550 --ptxas-options=--warn-on-spills>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>)
endif()
