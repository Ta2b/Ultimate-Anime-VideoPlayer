# Contrats de gpu : interface des backends et noyaux de la chaîne, sans GPU.
ajp_add_unit_tests(gpu
  SOURCES backend_test.cpp lanes_test.cpp vt_pixels_test.cpp
  DEPS ajp::media ajp::coreml_pixels)
