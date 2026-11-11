# Le grade HDR : décisions de grade_params (sans GPU) et hooks rendus par libplacebo ([gpu], sautés
# sans appareil Vulkan, sous la couche de validation de tests/support).
ajp_add_unit_tests(render SOURCES cube_lut_test.cpp shaders_test.cpp display_model_test.cpp grade_params_test.cpp grade_gpu_test.cpp grade_gpu_anime_test.cpp grade_chain_gpu_test.cpp output_tune_test.cpp faces_test.cpp emitters_test.cpp sharpen_gpu_test.cpp
  DEPS ajp::dep::placebo ajp::dep::vulkan ajp::kernels ajp::core_os
  MAIN ajp_test_vulkan)
# Les LUT livrées (data/luts), lues par les tests de la chaîne complète du grade.
target_compile_definitions(ajp_render_tests PRIVATE AJP_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
# Les visages (A4) et la référence des émetteurs (A5) n'ont pas d'appelant dans le lecteur (B41-05) : leur code
# vit hors de la bibliothèque render, compilé avec leurs tests seulement. tools/grade/emitters en garde la
# version numpy.
target_sources(ajp_render_tests PRIVATE "${CMAKE_SOURCE_DIR}/src/render/faces.cpp"
                                        "${CMAKE_SOURCE_DIR}/src/render/emitters.cpp")
