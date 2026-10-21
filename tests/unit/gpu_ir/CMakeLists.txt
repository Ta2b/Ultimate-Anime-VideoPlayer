ajp_add_unit_tests(gpu_ir
  SOURCES
    fuzz_test.cpp import_test.cpp memory_test.cpp onnx_reader_test.cpp ops_conv_test.cpp
    ops_sample_test.cpp ops_tensor_test.cpp passes_test.cpp property_test.cpp protobuf_test.cpp
    robustness_test.cpp
  DEPS ajp::kernels)
