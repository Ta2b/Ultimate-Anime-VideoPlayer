# Le backend se charge par dlopen, comme le lecteur le charge : le test ne le lie pas.
# AJP_SPAN_MODEL : le modèle SPAN d'AnimeJaNai installé sur le poste, lu en place (jamais
# copié dans le dépôt) ; absent, le test qui l'exécute se SKIP.
set(AJP_SPAN_MODEL "$ENV{HOME}/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
    CACHE FILEPATH "Modèle SPAN ×2 d'AnimeJaNai pour le test du backend TensorRT")
# AJP_SPAN_FP8_PACKAGE, AJP_SPAN_FP8_MODEL : le paquet dérivé par tools/span/derive_span_fp8.py et le SPAN dont il
# vient, pour le test caché [.span_fp8] (docs/span-fp8-cutlass.md) ; vides, il se SKIP.
set(AJP_SPAN_FP8_PACKAGE "" CACHE FILEPATH "Paquet .span8 du SPAN FP8 (test caché [.span_fp8])")
set(AJP_SPAN_FP8_MODEL "" CACHE FILEPATH "SPAN d'origine du paquet (test caché [.span_fp8])")
set(AJP_SPAN_FP8_IMAGE "" CACHE FILEPATH "Image PPM d'anime pour le test caché [.span_fp8] (sinon un dessin synthétique)")
add_executable(ajp_backend_tensorrt_tests backend_fixture.cpp backend_motion_test.cpp backend_test.cpp export_test.cpp gmfss_test.cpp
  span_fp8_test.cpp)
target_link_libraries(ajp_backend_tensorrt_tests PRIVATE
  ajp::gpu ajp::gpu_span_fp8 ajp::kernels ajp::prepare ajp::dep::cudart ${CMAKE_DL_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(ajp_backend_tensorrt_tests PRIVATE
  AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>"
  AJP_TEST_CACHE_DIR="${CMAKE_CURRENT_BINARY_DIR}/engines"
  AJP_SPAN_MODEL="${AJP_SPAN_MODEL}"
  AJP_SPAN_FP8_PACKAGE="${AJP_SPAN_FP8_PACKAGE}"
  AJP_SPAN_FP8_MODEL="${AJP_SPAN_FP8_MODEL}"
  AJP_SPAN_FP8_IMAGE="${AJP_SPAN_FP8_IMAGE}")
add_dependencies(ajp_backend_tensorrt_tests ajp_backend_tensorrt)
ajp_set_warnings(ajp_backend_tensorrt_tests)
catch_discover_tests(ajp_backend_tensorrt_tests TEST_PREFIX "backend_tensorrt: " DISCOVERY_MODE PRE_TEST)
