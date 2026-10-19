# Les tests [cuda] construisent leurs moteurs depuis des ONNX générés (onnx_builder.h de gpu_ir)
# et se comparent à l'interpréteur gpu_ir ; sans appareil CUDA, ils sautent avec leur raison.
ajp_add_unit_tests(cuda_trt
  SOURCES plan_builder.cpp refusal_test.cpp runtime_test.cpp trtexec_test.cpp types_test.cpp
  DEPS ajp::gpu_ir ajp::prepare ajp::dep::tensorrt_onnx ajp::dep::spdlog)
target_include_directories(ajp_cuda_trt_tests PRIVATE "${PROJECT_SOURCE_DIR}/tests/unit/gpu_ir")
if(AJP_TRTEXEC)
  target_compile_definitions(ajp_cuda_trt_tests PRIVATE AJP_TRTEXEC="${AJP_TRTEXEC}")
endif()
