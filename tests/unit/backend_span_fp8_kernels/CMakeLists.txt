# Noyaux CUTLASS du SPAN FP8 seuls (étape 3.1 du plan NVIDIA) : aucune ligne au-delà de M lue ni écrite, aux tailles
# dont la dernière tuile de 128 lignes déborde ; l'épilogue de to_pixel écrit l'anneau au bit près de write_rgba16
# (étape 3.4a) ; un modèle à 48 canaux calculé en 48 suit le calcul en 64 (kKernelsVersion 4). Construit avec
# AJP_WITH_SPAN_FP8 ; sans GPU sm 8.9, chaque test se SKIP.
add_executable(ajp_span_fp8_kernel_tests span_fp8_kernel_test.cpp span_fp8_48_test.cpp)
target_include_directories(ajp_span_fp8_kernel_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_span_fp8_kernel_tests PRIVATE
  ajp::gpu_span_fp8 ajp_backend_span_fp8_kernels ajp_backend_tensorrt_kernels ajp::dep::cudart
  Catch2::Catch2WithMain)
ajp_set_warnings(ajp_span_fp8_kernel_tests)
catch_discover_tests(ajp_span_fp8_kernel_tests TEST_PREFIX "span_fp8_kernels: " DISCOVERY_MODE PRE_TEST)
