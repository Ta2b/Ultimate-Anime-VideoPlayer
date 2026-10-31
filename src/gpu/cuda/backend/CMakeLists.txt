# ajp_backend_tensorrt.so : le backend TensorRT (gpu/backend.h), chargé à l'exécution.
# Il emporte prepare (constructeur trtexec) et sa propre copie de spdlog ; ses symboles
# restent cachés, sauf les trois points d'entrée.
add_library(ajp_backend_tensorrt_kernels STATIC drba_fused_kernels.cu export_kernels.cu gmfss_kernels.cu hdr_kernels.cu
  rife_fused_kernels.cu rife_kernels.cu slice_kernels.cu warp_kernels.cu)
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
# en-têtes, ou hors Linux (Windows : jamais construit ni mesuré, voir le document). Éteints, une version vide (span_fp8_net_none.cpp) du même
# contrat refuse toute demande et la chaîne reste sur TensorRT.
set(AJP_CUTLASS_INCLUDE "${PROJECT_SOURCE_DIR}/third_party/cutlass/include")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${AJP_CUTLASS_INCLUDE}/cutlass/cutlass.h")
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
  # sm 89 seulement : les mma FP8 n'existent pas avant, et la liste AJP_CUDA_ARCHITECTURES n'y change rien. Le
  # PTX (89 = code réel et virtuel) garde le module chargeable sur une carte plus récente, où supported_device
  # refuse de lancer les noyaux.
  set_target_properties(ajp_backend_span_fp8_kernels PROPERTIES
    CUDA_ARCHITECTURES 89 POSITION_INDEPENDENT_CODE ON)
  target_compile_options(ajp_backend_span_fp8_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcudafe=--diag_suppress=177
      -Xcudafe=--diag_suppress=550 --ptxas-options=--verbose>
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
  drba_fused_engine.cpp entry.cpp gpu_timing.cpp host_upload.cpp ring_slot.cpp rife_fused_engine.cpp span_fp8_engine.cpp tensorrt_backend.cpp tensorrt_export.cpp
  tensorrt_gmfss.cpp tensorrt_kernels.cpp tensorrt_lanes.cpp tensorrt_drba_fused.cpp tensorrt_rife.cpp tensorrt_rife_fused.cpp tensorrt_span_derive.cpp tensorrt_span_fp8.cpp
  tensorrt_textures.cpp tensorrt_warp.cpp tensorrt_true_hdr.cpp trt_engine.cpp)
target_include_directories(ajp_backend_tensorrt PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_tensorrt PRIVATE
  ajp::gpu ajp::gpu_span_fp8 ajp::prepare ajp_backend_tensorrt_kernels ${ajp_span_fp8_net} ajp::dep::tensorrt CUDA::cuda_driver ${CMAKE_DL_LIBS})
target_compile_definitions(ajp_backend_tensorrt PRIVATE AJP_TRTEXEC_PATH="${AJP_TRTEXEC}")
set_target_properties(ajp_backend_tensorrt PROPERTIES
  PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_tensorrt PRIVATE LINKER:--exclude-libs,ALL LINKER:--no-undefined)
endif()
ajp_set_warnings(ajp_backend_tensorrt)

# Les bibliothèques statiques liées dans un objet partagé doivent être compilées en PIC. prepare
# et les modules qu'il tire sont déclarés après gpu : la propriété se pose à la fin du projet.
function(ajp_backend_tensorrt_pic)
  foreach(lib ajp_prepare ajp_core ajp_core_log ajp_kernels ajp_vendor_hdr ajp_gpu_span_fp8)
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
