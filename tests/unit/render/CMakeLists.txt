# Le grade HDR : décisions de grade_params (sans GPU) et hooks rendus par libplacebo ([gpu], sautés
# sans appareil Vulkan, sous la couche de validation de tests/support).
ajp_add_unit_tests(render SOURCES adaptive_rules_test.cpp cube_lut_test.cpp curve_test.cpp display_model_test.cpp grade_params_test.cpp grade_gpu_test.cpp grade_fused_test.cpp grade_gpu_local_test.cpp grade_gpu_anime_test.cpp grade_golden_test.cpp grade_ring_ref_test.cpp scene_hdr_test.cpp
  DEPS ajp::dep::placebo ajp::dep::vulkan ajp::kernels
  MAIN ajp_test_vulkan)
