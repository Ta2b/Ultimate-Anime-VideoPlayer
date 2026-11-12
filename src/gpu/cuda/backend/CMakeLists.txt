# ajp_backend_tensorrt.so : le backend TensorRT (gpu/backend.h), chargé à l'exécution.
# Il emporte prepare (constructeur trtexec) et sa propre copie de spdlog ; ses symboles
# restent cachés, sauf les trois points d'entrée.
add_library(ajp_backend_tensorrt_kernels STATIC deband_kernels.cu drba_fused_kernels.cu export_kernels.cu
  rife_fused_kernels.cu rife_kernels.cu slice_kernels.cu stats_kernels.cu warp_kernels.cu)
target_include_directories(ajp_backend_tensorrt_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_tensorrt_kernels PRIVATE ajp::dep::cudart)
# -lineinfo pour ncu ; ptxas donne les registres de chaque noyau au journal de build, et un
# débordement de registres en mémoire locale est une erreur (pixel_kernels.cuh) sous Linux, où le
# PC de référence et la CI le vérifient ; le code de l'appareil est le même sous Windows, où les
# en-têtes de MSVC que lit nvcc ne doivent pas arrêter la release.
# Les avertissements du compilateur hôte : GCC et Clang sous Linux, MSVC sous Windows.
target_compile_options(ajp_backend_tensorrt_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ptxas-options=--verbose,--warn-on-spills>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/W3>)
set_target_properties(ajp_backend_tensorrt_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Le SPAN entier en FP8 par noyaux CUTLASS (docs/span-fp8-cutlass.md). CUTLASS est vendorisé (third_party/cutlass,
# BSD-3-Clause, en-têtes seulement) ; les noyaux n'existent que pour sm 8.9, et sont éteints sans CUDA, sans les
# en-têtes, ou par défaut hors Linux et Windows. Windows (12.4) : construits avec MSVC (/Zc:preprocessor
# ci-dessous), allumés par défaut depuis le 12 novembre 2026 : le reçu v2 et sa porte, refaits sur la carte,
# décident seuls si le paquet sert ; PSNR et gain restent à relever sur la RTX à ±3 % de Linux
# (docs/span-fp8-cutlass.md). Éteints, une version vide (span_fp8_net_none.cpp) du même
# contrat refuse toute demande et la chaîne reste sur TensorRT.
set(AJP_CUTLASS_INCLUDE "${PROJECT_SOURCE_DIR}/third_party/cutlass/include")
if(CMAKE_SYSTEM_NAME MATCHES "Linux|Windows" AND EXISTS "${AJP_CUTLASS_INCLUDE}/cutlass/cutlass.h")
  set(ajp_span_fp8_default ON)
else()
  set(ajp_span_fp8_default OFF)
endif()
option(AJP_WITH_SPAN_FP8 "Construire les noyaux CUTLASS du SPAN entier en FP8 (sm 8.9)" ${ajp_span_fp8_default})
if(AJP_WITH_SPAN_FP8)
  add_library(ajp_backend_span_fp8_kernels STATIC span_fp8_net.cu span_fp8_ref.cu)
  target_include_directories(ajp_backend_span_fp8_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
  # SYSTEM : les avertissements de CUTLASS ne sont pas les nôtres.
  target_include_directories(ajp_backend_span_fp8_kernels SYSTEM PRIVATE "${AJP_CUTLASS_INCLUDE}")
  target_link_libraries(ajp_backend_span_fp8_kernels PRIVATE ajp::gpu_span_fp8 ajp::dep::cudart)
  # sm 89 (Ada) et sm 120 (Blackwell grand public, étape 10.2) : les mma FP8 m16n8k32 n'existent pas avant 89,
  # et la liste AJP_CUDA_ARCHITECTURES n'y change rien. Le PTX de 120 (code réel et virtuel) garde le module
  # chargeable sur une carte plus récente, où supported_device refuse de lancer les noyaux. CUDA 12.8 au moins.
  set_target_properties(ajp_backend_span_fp8_kernels PROPERTIES
    CUDA_ARCHITECTURES "89-real;120" POSITION_INDEPENDENT_CODE ON)
  # Comme les autres noyaux du backend : un débordement de registres de net_kernel est une erreur sous Linux
  # (B02-03) ; les diag_suppress de CUTLASS restent.
  target_compile_options(ajp_backend_span_fp8_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcudafe=--diag_suppress=177
      -Xcudafe=--diag_suppress=550 --ptxas-options=--verbose,--warn-on-spills>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra>
    # CCCL de CUDA 13 exige le préprocesseur conforme de MSVC.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>)
  set(ajp_span_fp8_net ajp_backend_span_fp8_kernels)
else()
  add_library(ajp_backend_span_fp8_none STATIC span_fp8_net_none.cpp)
  target_include_directories(ajp_backend_span_fp8_none PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(ajp_backend_span_fp8_none PRIVATE ajp::gpu_span_fp8 ajp::dep::cudart)
  set_target_properties(ajp_backend_span_fp8_none PROPERTIES POSITION_INDEPENDENT_CODE ON)
  ajp_set_warnings(ajp_backend_span_fp8_none)
  set(ajp_span_fp8_net ajp_backend_span_fp8_none)
endif()

add_library(ajp_backend_tensorrt MODULE
  drba_fused_engine.cpp entry.cpp fused_graphs.cpp gpu_timing.cpp host_upload.cpp ring_slot.cpp rife_fused_engine.cpp
  ingest_chroma.cpp shared_activations.cpp span_fp8_engine.cpp tensorrt_backend.cpp tensorrt_export.cpp tensorrt_kernels.cpp
  tensorrt_lanes.cpp tensorrt_drba_fused.cpp tensorrt_rife.cpp tensorrt_rife_fused.cpp tensorrt_span_derive.cpp
  span_timing.cpp tensorrt_span_fp8.cpp tensorrt_stats.cpp tensorrt_textures.cpp tensorrt_warp.cpp trt_engine.cpp)
target_include_directories(ajp_backend_tensorrt PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_tensorrt PRIVATE
  ajp::gpu ajp::gpu_span_fp8 ajp::cuda_fused_keys ajp::prepare ajp::core_log ajp::dep::spdlog ajp_backend_tensorrt_kernels ${ajp_span_fp8_net}
  ajp::dep::tensorrt CUDA::cuda_driver ${CMAKE_DL_LIBS})
# Le trtexec du paquet est celui de la configuration : son SHA-256 entre dans le module, qui refuse
# tout autre fichier posé à côté de lui (P1.21). Le repli sur le trtexec de la machine est réservé
# aux builds de développement.
set(ajp_trtexec_sha256 "")
set(ajp_trtexec_path "")
if(AJP_TRTEXEC AND EXISTS "${AJP_TRTEXEC}")
  set(ajp_trtexec_path "${AJP_TRTEXEC}")
  set(ajp_trtexec_packaged "${AJP_TRTEXEC}")
  # Un trtexec remplacé sur place (TensorRT mis à jour dans le même dossier) refait la configuration : sinon le
  # module garderait l'ancienne empreinte et refuserait le trtexec livré (C01-08).
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${AJP_TRTEXEC}")
  # Sous Linux, le paquet range trtexec sur $ORIGIN (ses bibliothèques de NVIDIA sont à côté de lui) : la copie
  # rangée est faite ici, une fois, et c'est son SHA-256 qui entre dans le module ; package-linux.sh livre cette
  # copie telle quelle (AJP_TRTEXEC_PACKAGED). Patcher après coup changeait l'empreinte (0.10.0).
  find_program(AJP_PATCHELF patchelf)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND AJP_PATCHELF)
    set(ajp_trtexec_packaged "${CMAKE_BINARY_DIR}/trtexec-paquet/trtexec")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/trtexec-paquet")
    file(COPY_FILE "${AJP_TRTEXEC}" "${ajp_trtexec_packaged}")
    execute_process(COMMAND "${AJP_PATCHELF}" --set-rpath "$ORIGIN" "${ajp_trtexec_packaged}"
                    RESULT_VARIABLE ajp_patchelf_result)
    if(NOT ajp_patchelf_result EQUAL 0)
      message(FATAL_ERROR "patchelf n'a pas rangé ${ajp_trtexec_packaged} sur $ORIGIN")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "patchelf manquant : le trtexec livré ne trouvera pas ses bibliothèques à côté de lui")
  endif()
  set(AJP_TRTEXEC_PACKAGED "${ajp_trtexec_packaged}" CACHE INTERNAL "trtexec livré, celui dont le module garde l'empreinte")
  file(SHA256 "${ajp_trtexec_packaged}" ajp_trtexec_sha256)
endif()
target_compile_definitions(ajp_backend_tensorrt PRIVATE AJP_TRTEXEC_SHA256="${ajp_trtexec_sha256}"
  AJP_TRTEXEC_PATH="${ajp_trtexec_path}" $<$<CONFIG:Release>:AJP_TRTEXEC_NO_FALLBACK>)
set_target_properties(ajp_backend_tensorrt PROPERTIES
  PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_tensorrt PRIVATE LINKER:--exclude-libs,ALL LINKER:--no-undefined)
endif()
ajp_set_warnings(ajp_backend_tensorrt)

# Les bibliothèques statiques liées dans un objet partagé doivent être compilées en PIC. prepare
# et les modules qu'il tire sont déclarés après gpu : la propriété se pose à la fin du projet.
function(ajp_backend_tensorrt_pic)
  foreach(lib ajp_prepare ajp_core ajp_core_log ajp_kernels ajp_gpu_span_fp8 ajp_cuda_fused_keys)
    if(TARGET ${lib})
      set_property(TARGET ${lib} PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL ajp_backend_tensorrt_pic)

# Outil de contrôle du SPAN FP8 : tools/span/derive_span_fp8.py y passe les images de validation et chronomètre.
if(AJP_WITH_SPAN_FP8)
  add_executable(ajp_span_fp8_run span_fp8_run.cpp)
  target_include_directories(ajp_span_fp8_run PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(ajp_span_fp8_run PRIVATE ajp::gpu_span_fp8 ajp_backend_span_fp8_kernels ajp::dep::cudart)
  ajp_set_warnings(ajp_span_fp8_run)
endif()
