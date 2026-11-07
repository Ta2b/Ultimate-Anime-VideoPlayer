# Contrats de gpu : interface des backends et noyaux de la chaîne, sans GPU.
ajp_add_unit_tests(gpu
  SOURCES graph_book_test.cpp backend_test.cpp coreml_compiled_test.cpp lanes_test.cpp vt_pixels_test.cpp gpu_profile_test.cpp
  DEPS ajp::media ajp::coreml_pixels)

# P5.19 : l'empreinte des en-têtes de la frontière suit une virtuelle ajoutée, pas un commentaire.
add_test(NAME "gpu: l'empreinte de la frontière change avec une virtuelle, pas avec un commentaire"
         COMMAND ${CMAKE_COMMAND} -DSOURCE=${PROJECT_SOURCE_DIR} -DWORK=${CMAKE_CURRENT_BINARY_DIR}/boundary_hash
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/boundary_hash_test.cmake)
