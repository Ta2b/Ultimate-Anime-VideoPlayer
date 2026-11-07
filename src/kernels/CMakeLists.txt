# kernels : spécification des noyaux (spec.h) et référence CPU FP32 (ref/), vérité des ports
# CUDA, GLSL et Metal. La référence sert aux tests et aux goldens ; seuls le demi-flottant de ref/half.h (gpu_ir,
# qui l'utilise pour arrondir comme les noyaux) et l'analyse d'image de ref/frame_stats.h (la pré-analyse des
# plans, et les mesures tirées des comptes du noyau) sont liés à un binaire livré.
# Feuille du graphe : ne dépend de rien.
ajp_add_module(kernels
  SOURCES
    glsl.cpp
    ref/color.cpp
    ref/detect.cpp
    ref/frame_stats.cpp
    ref/half.cpp
    ref/resample.cpp
    ref/gmfss.cpp
    ref/rife.cpp
    ref/rife_fused.cpp
    ref/transfer.cpp
    ref/warp.cpp)

# La référence ne contracte jamais a × b + c en FMA : ses arrondis sont ceux écrits dans spec.h.
# PRIVATE : les formules inline de spec/ s'évaluent dans le fichier qui les appelle, et chaque
# en-tête de spec/ y ouvre sa propre région sans contraction (spec/fp_off_begin.h), sans imposer
# l'option au reste du lecteur (P5.22). C++ seul : nvcc refuse l'option, et les ports CUDA suivent
# leurs propres réglages. MSVC ne contracte pas en /fp:precise.
if(NOT MSVC)
  target_compile_options(ajp_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# En-tête GLSL des constantes de spec.h, pour les shaders du moteur Vulkan :
#   #include "ajp_kernels_spec.glsl", avec le dossier AJP_KERNELS_GLSL_DIR en -I de glslc.
add_executable(ajp_kernels_glsl glsl_main.cpp)
target_link_libraries(ajp_kernels_glsl PRIVATE ajp::kernels)
ajp_set_warnings(ajp_kernels_glsl)

set(AJP_KERNELS_GLSL_DIR "${CMAKE_CURRENT_BINARY_DIR}/glsl"
    CACHE INTERNAL "Dossier de l'en-tête GLSL généré depuis kernels/spec.h")
add_custom_command(
  OUTPUT "${AJP_KERNELS_GLSL_DIR}/ajp_kernels_spec.glsl"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${AJP_KERNELS_GLSL_DIR}"
  COMMAND ajp_kernels_glsl "${AJP_KERNELS_GLSL_DIR}/ajp_kernels_spec.glsl"
  DEPENDS ajp_kernels_glsl
  COMMENT "Constantes de kernels/spec.h pour GLSL"
  VERBATIM)
add_custom_target(ajp_kernels_glsl_header ALL DEPENDS "${AJP_KERNELS_GLSL_DIR}/ajp_kernels_spec.glsl")
