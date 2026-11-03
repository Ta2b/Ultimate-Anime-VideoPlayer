# Noyaux CUTLASS du SPAN FP8 seuls (étape 3.1 du plan NVIDIA) : aucune ligne au-delà de M lue ni écrite, aux tailles
# dont la dernière tuile de 128 lignes déborde. Construit avec AJP_WITH_SPAN_FP8 ; sans GPU sm 8.9, chaque test se
# SKIP.
add_executable(ajp_span_fp8_kernel_tests span_fp8_kernel_test.cpp)
target_include_directories(ajp_span_fp8_kernel_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_span_fp8_kernel_tests PRIVATE
  ajp::gpu_span_fp8 ajp_backend_span_fp8_kernels ajp::dep::cudart Catch2::Catch2WithMain)
ajp_set_warnings(ajp_span_fp8_kernel_tests)
catch_discover_tests(ajp_span_fp8_kernel_tests TEST_PREFIX "span_fp8_kernels: " DISCOVERY_MODE PRE_TEST)
