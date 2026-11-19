# Contrats de gpu : interface des backends et noyaux de la chaîne, sans GPU (conv_stream_test : le calendrier
# de la forme en flux des convolutions natives, et le noyau rejoué sur l'hôte en FP16).
kirari_add_unit_tests(gpu
  SOURCES graph_book_test.cpp backend_test.cpp conv_stream_test.cpp coreml_compiled_test.cpp cuda_fused_keys_test.cpp
    lanes_test.cpp native_mix_test.cpp vt_pixels_test.cpp gpu_profile_test.cpp guarded_relay_test.cpp
    rife_mix_choice_test.cpp rife_native_verdict_test.cpp
  DEPS kirari::media kirari::coreml_pixels kirari::cuda_fused_keys kirari::cuda_native_mix kirari::cuda_rife_mix
    kirari::kernels)
# A21-04 : guarded_relay_test.cpp lit les en-têtes de la frontière.
target_compile_definitions(kirari_gpu_tests PRIVATE KIRARI_GPU_SOURCE_DIR="${PROJECT_SOURCE_DIR}/src/gpu")

# P5.19 : l'empreinte des en-têtes de la frontière suit une virtuelle ajoutée, pas un commentaire.
add_test(NAME "gpu: l'empreinte de la frontière change avec une virtuelle, pas avec un commentaire"
         COMMAND ${CMAKE_COMMAND} -DSOURCE=${PROJECT_SOURCE_DIR} -DWORK=${CMAKE_CURRENT_BINARY_DIR}/boundary_hash
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/boundary_hash_test.cmake)
