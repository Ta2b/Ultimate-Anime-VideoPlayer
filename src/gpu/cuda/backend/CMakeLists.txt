# kirari_backend_tensorrt.so : le backend TensorRT (gpu/backend.h), chargé à l'exécution.
# Il emporte prepare (constructeur trtexec) et sa propre copie de spdlog ; ses symboles
# restent cachés, sauf les trois points d'entrée.
add_library(kirari_backend_tensorrt_kernels STATIC deband_kernels.cu doubt_kernels.cu drba_fused_kernels.cu export_kernels.cu
  gmflow_backbone_net.cu gmfss_kernels.cu gmfss_native_net.cu gmfss_pair_native_net.cu layer_kernels.cu rife_fused_kernels.cu
  rife_kernels.cu rife_native_bench.cu rife_native_glue.cu rife_native_net.cu rife_native_pick.cu rife_native_weights.cu
  scene_camera_kernels.cu scene_flow_kernels.cu slice_kernels.cu stats_kernels.cu
  tfix_blend_kernels.cu tfix_fused_kernels.cu warp_kernels.cu warp_lib.cu)
target_include_directories(kirari_backend_tensorrt_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(kirari_backend_tensorrt_kernels PRIVATE kirari::dep::cudart)
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

# Le SPAN entier en FP8 par noyaux CUTLASS (docs/span-fp8-cutlass.md). CUTLASS est vendorisé (third_party/cutlass,
# BSD-3-Clause, en-têtes seulement) ; les noyaux n'existent que pour sm 8.9, et sont éteints sans CUDA, sans les
# en-têtes, ou par défaut hors Linux. Windows (12.4) : -DKIRARI_WITH_SPAN_FP8=ON les construit avec MSVC
# (/Zc:preprocessor ci-dessous ; KIRARI_WITH_SPAN_FP8=ON pour package-windows.ps1, entrée span_fp8 de
# release-windows.yml) ; le défaut y reste OFF tant que PSNR et gain ne sont pas mesurés sur la RTX à ±3 % de
# Linux (docs/span-fp8-cutlass.md). Éteints, une version vide (span_fp8_net_none.cpp) du même
# contrat refuse toute demande et la chaîne reste sur TensorRT.
set(KIRARI_CUTLASS_INCLUDE "${PROJECT_SOURCE_DIR}/third_party/cutlass/include")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${KIRARI_CUTLASS_INCLUDE}/cutlass/cutlass.h")
  set(kirari_span_fp8_default ON)
else()
  set(kirari_span_fp8_default OFF)
endif()
option(KIRARI_WITH_SPAN_FP8 "Construire les noyaux CUTLASS du SPAN entier en FP8 (sm 8.9)" ${kirari_span_fp8_default})
if(KIRARI_WITH_SPAN_FP8)
  # La référence de la dérivation FP8 (span_fp8_ref.cu) arrête nvcc 13.4 sous MSVC (erreur interne) : sous
  # Windows, sa version vide la remplace, et seuls les noyaux du SPAN sont construits.
  if(MSVC)
    set(kirari_span_fp8_ref span_fp8_ref_none.cpp)
  else()
    set(kirari_span_fp8_ref span_fp8_ref.cu)
  endif()
  # Les options de nvcc de la bibliothèque des noyaux CUTLASS ci-dessous.
  function(kirari_span_cutlass_target target architectures)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    # SYSTEM : les avertissements de CUTLASS ne sont pas les nôtres.
    target_include_directories(${target} SYSTEM PRIVATE "${KIRARI_CUTLASS_INCLUDE}")
    target_link_libraries(${target} PRIVATE kirari::gpu_span_fp8 kirari::dep::cudart)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${architectures}" POSITION_INDEPENDENT_CODE ON)
    # Comme les autres noyaux du backend : un débordement de registres de net_kernel est une erreur sous Linux
    # (B02-03) ; les diag_suppress de CUTLASS restent.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcudafe=--diag_suppress=177
        -Xcudafe=--diag_suppress=550 --ptxas-options=--verbose,--warn-on-spills>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra>
      # CCCL de CUDA 13 exige le préprocesseur conforme de MSVC.
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>)
  endfunction()
  # Une seule bibliothèque pour les noyaux FP8 (CUTLASS) et FP16 natifs (span_native16.cu, sans CUTLASS), aux
  # mêmes architectures. sm 89 (Ada) et sm 120 (Blackwell
  # grand public, étape 10.2) : les mma FP8 m16n8k32 n'existent pas avant 89, et la liste
  # KIRARI_CUDA_ARCHITECTURES n'y change rien. Le PTX de 120 (code réel et virtuel) garde le module chargeable sur
  # une carte plus récente, où supported_device refuse de lancer les noyaux. CUDA 12.8 au moins.
  add_library(kirari_backend_span_fp8_kernels STATIC span_fp8_net.cu span_fp8_net48.cu span_native16.cu
    ${kirari_span_fp8_ref})
  kirari_span_cutlass_target(kirari_backend_span_fp8_kernels "89-real;120")
  set(kirari_span_fp8_net kirari_backend_span_fp8_kernels)
else()
  add_library(kirari_backend_span_fp8_none STATIC span_fp8_net_none.cpp span_fp8_ref_none.cpp)
  target_include_directories(kirari_backend_span_fp8_none PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(kirari_backend_span_fp8_none PRIVATE kirari::gpu_span_fp8 kirari::dep::cudart)
  set_target_properties(kirari_backend_span_fp8_none PROPERTIES POSITION_INDEPENDENT_CODE ON)
  kirari_set_warnings(kirari_backend_span_fp8_none)
  set(kirari_span_fp8_net kirari_backend_span_fp8_none)
endif()

add_library(kirari_backend_tensorrt MODULE
  drba_fused_engine.cpp entry.cpp fused_graphs.cpp gmfss_fused_engine.cpp gmfss_fused_trial.cpp gmfss_pair_engine.cpp gmfss_pair_trial.cpp gpu_timing.cpp host_upload.cpp ring_slot.cpp rife_fused_engine.cpp rife_fused_native.cpp
  rife_native_mix.cpp rife_native_trial.cpp
  ingest_chroma.cpp nvof_session.cpp scene_camera.cpp scene_flow.cpp shared_activations.cpp span_fp8_check.cpp span_fp8_engine.cpp span_fp8_tiles.cpp tensorrt_backend.cpp tensorrt_doubt.cpp tensorrt_export.cpp tensorrt_kernels.cpp
  tensorrt_gmfss.cpp tensorrt_gmfss_fused.cpp tensorrt_gmfss_pair.cpp tensorrt_layers.cpp tensorrt_lanes.cpp tensorrt_drba_fused.cpp tensorrt_rife.cpp tensorrt_rife_fused.cpp tensorrt_scene_flow.cpp tensorrt_span_derive.cpp
  span_timing.cpp tensorrt_span_fp8.cpp tensorrt_stats.cpp tensorrt_textures.cpp tensorrt_tfix_fused.cpp tensorrt_warp.cpp tfix_fused_engine.cpp
  trt_engine.cpp)
target_include_directories(kirari_backend_tensorrt PRIVATE "${PROJECT_SOURCE_DIR}/src")
# En-têtes BSD de NVIDIA Optical Flow (nvof_session.cpp) : types et signatures seulement, la bibliothèque vient
# du pilote ; aucun fichier de NVIDIA n'est livré.
target_include_directories(kirari_backend_tensorrt SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nvidia-optical-flow")
target_link_libraries(kirari_backend_tensorrt PRIVATE
  kirari::gpu kirari::gpu_span_fp8 kirari::cuda_fused_keys kirari::cuda_rife_mix kirari::prepare kirari::core_log kirari::dep::spdlog kirari_backend_tensorrt_kernels ${kirari_span_fp8_net}
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
          kirari_cuda_rife_mix)
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
  target_link_libraries(kirari_span_fp8_run PRIVATE kirari::gpu_span_fp8 kirari::gpu_span_fp8_onnx kirari_backend_span_fp8_kernels
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
