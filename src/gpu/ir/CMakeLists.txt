# gpu_ir : import ONNX, représentation interne, passes et interpréteur CPU FP32.
# Base du moteur Vulkan maison et du futur générateur Metal ; ne touche à aucun GPU.
# Le demi-flottant est celui de kernels (kernels/ref/half.h) : une seule implémentation.
kirari_add_module(gpu_ir
  SOURCES
    describe.cpp fp8_weights.cpp graph.cpp import.cpp import_attrs.cpp interpreter.cpp memory_plan.cpp
    onnx_fp8.cpp onnx_io_type.cpp onnx_reader.cpp onnx_tensor.cpp onnx_unshuffle.cpp op_elementwise.cpp op_pixel.cpp op_sample.cpp op_shape.cpp op_slice.cpp
    op_util.cpp op_window.cpp ops.cpp pass_cast.cpp pass_fold.cpp pass_grid.cpp pass_fuse.cpp pass_layout.cpp pass_pixel.cpp
    pass_quant.cpp pass_transpose.cpp protobuf.cpp shape.cpp
  DEPS kirari::core_types
  PRIVATE_DEPS kirari::core_log kirari::kernels)

# L'interpréteur sert de référence : son arrondi ne dépend pas du compilateur.
if(NOT MSVC)
  target_compile_options(kirari_gpu_ir PRIVATE -ffp-contract=off)
endif()

