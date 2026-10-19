# Le grade HDR : décisions de grade_params (sans GPU) et hooks rendus par libplacebo ([gpu], sautés
# sans appareil Vulkan).
ajp_add_unit_tests(render SOURCES grade_params_test.cpp grade_gpu_test.cpp
  DEPS ajp::dep::placebo ajp::dep::vulkan)
