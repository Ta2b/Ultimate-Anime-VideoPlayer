# kernels : spécification des noyaux (spec.h) et référence CPU FP32 (ref/), vérité des ports
# CUDA, GLSL et Metal. La référence sert aux tests et aux goldens, mais une partie de ref/ est du code produit,
# lié aux binaires livrés (C50-05) : le demi-flottant de ref/half.h (gpu_ir, Vulkan, TensorRT : arrondir comme
# les noyaux), l'analyse d'image de ref/frame_stats.h (la pré-analyse des plans, shot_probe.cpp, et les mesures
# tirées des comptes du noyau), la carte de la scène de ref/scene_flow.h (scene_motion_of, lue par l'étage gpu)
# et, de ref/resample.h, compute_weights et Weights (les tables de chroma de
# l'ingestion TensorRT, gpu/cuda/backend/ingest_chroma.cpp, de l'export et de l'ingestion Vulkan). frame_stats
# appelle ref/transfer.cpp. Modifier ces fonctions modifie le lecteur ; le reste de ref/ (color, detect, rife,
# rife_fused, warp) n'est lu que par les tests, mais fixe le contrat que les ports doivent suivre.
# Feuille du graphe : ne dépend de rien.
kirari_add_module(kernels
  SOURCES
    glsl.cpp
    ref/color.cpp
    ref/deband.cpp
    ref/detect.cpp
    ref/frame_stats.cpp
    ref/gmfss.cpp
    ref/half.cpp
    ref/motion_doubt.cpp
    ref/resample.cpp
    ref/rife.cpp
    ref/rife_fused.cpp
    ref/scene_flow.cpp
    ref/transfer.cpp
    ref/warp.cpp
    ref/warp_pan.cpp)

# La référence ne contracte jamais a × b + c en FMA : ses arrondis sont ceux écrits dans spec.h.
# PRIVATE : les formules inline de spec/ s'évaluent dans le fichier qui les appelle, et chaque
# en-tête de spec/ y ouvre sa propre région sans contraction (spec/fp_off_begin.h), sans imposer
# l'option au reste du lecteur (P5.22). C++ seul : nvcc refuse l'option, et les ports CUDA suivent
# leurs propres réglages. MSVC ne contracte pas en /fp:precise.
if(NOT MSVC)
  target_compile_options(kirari_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# En-tête GLSL des constantes de spec.h, pour les shaders du moteur Vulkan :
#   #include "kirari_kernels_spec.glsl", avec le dossier KIRARI_KERNELS_GLSL_DIR en -I de glslc.
add_executable(kirari_kernels_glsl glsl_main.cpp)
target_link_libraries(kirari_kernels_glsl PRIVATE kirari::kernels)
kirari_set_warnings(kirari_kernels_glsl)

set(KIRARI_KERNELS_GLSL_DIR "${CMAKE_CURRENT_BINARY_DIR}/glsl"
    CACHE INTERNAL "Dossier de l'en-tête GLSL généré depuis kernels/spec.h")
add_custom_command(
  OUTPUT "${KIRARI_KERNELS_GLSL_DIR}/kirari_kernels_spec.glsl"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KIRARI_KERNELS_GLSL_DIR}"
  COMMAND kirari_kernels_glsl "${KIRARI_KERNELS_GLSL_DIR}/kirari_kernels_spec.glsl"
  DEPENDS kirari_kernels_glsl
  COMMENT "Constantes de kernels/spec.h pour GLSL"
  VERBATIM)
add_custom_target(kirari_kernels_glsl_header ALL DEPENDS "${KIRARI_KERNELS_GLSL_DIR}/kirari_kernels_spec.glsl")
