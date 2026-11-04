# Contrats de gpu : interface des backends et noyaux de la chaîne, sans GPU.
ajp_add_unit_tests(gpu
  SOURCES graph_book_test.cpp backend_test.cpp coreml_compiled_test.cpp lanes_test.cpp vt_pixels_test.cpp gpu_profile_test.cpp
  DEPS ajp::media ajp::coreml_pixels)
