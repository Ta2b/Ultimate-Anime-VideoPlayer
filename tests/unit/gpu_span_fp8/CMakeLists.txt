# Test caché [.span_compat] : le paquet que tools/span/derive_span_fp8.py a écrit d'un SPAN, et ce SPAN ; la lecture
# native des poids du second doit retrouver, octet pour octet, les poids du premier. Vides, il se SKIP.
set(AJP_SPAN_FP8_MODEL "" CACHE FILEPATH "SPAN d'origine du paquet (tests cachés du SPAN FP8)")
set(AJP_SPAN_FP8_PYTHON_PACKAGE "" CACHE FILEPATH "Paquet .span8 écrit par derive_span_fp8.py (test caché [.span_compat])")
ajp_add_unit_tests(gpu_span_fp8 SOURCES derive_test.cpp metrics_test.cpp onnx_weights_test.cpp package_test.cpp
  DEPS ajp::gpu_span_fp8_onnx)
target_compile_definitions(ajp_gpu_span_fp8_tests PRIVATE
  AJP_SPAN_FP8_MODEL="${AJP_SPAN_FP8_MODEL}" AJP_SPAN_FP8_PYTHON_PACKAGE="${AJP_SPAN_FP8_PYTHON_PACKAGE}")
