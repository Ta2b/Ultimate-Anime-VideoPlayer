ajp_add_unit_tests(gpu_ir
  SOURCES
    fuzz_test.cpp half_test.cpp import_test.cpp memory_test.cpp onnx_reader_test.cpp ops_conv_test.cpp
    ops_sample_test.cpp ops_tensor_test.cpp passes_test.cpp protobuf_test.cpp)
