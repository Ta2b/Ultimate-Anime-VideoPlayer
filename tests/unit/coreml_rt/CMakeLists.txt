# Le runtime Core ML sur un Mac, sans GPU (Core ML calcule alors sur le CPU) : un petit réseau
# construit, chargé et exécuté en direct, comparé à l'interpréteur de gpu_ir et à ONNX Runtime.
ajp_add_unit_tests(coreml_rt SOURCES coreml_rt_test.cpp DEPS ajp::gpu_ir ajp::kernels)
