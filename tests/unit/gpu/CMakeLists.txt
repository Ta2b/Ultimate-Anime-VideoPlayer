# Contrats de gpu : interface des backends et noyaux de la chaîne, sans GPU.
ajp_add_unit_tests(gpu
  SOURCES backend_test.cpp
  DEPS ajp::media)
