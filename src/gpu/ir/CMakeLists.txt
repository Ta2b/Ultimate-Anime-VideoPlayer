# gpu_ir : import ONNX, représentation interne, passes et interpréteur CPU FP32.
# Base du moteur Vulkan maison et du futur générateur Metal ; ne touche à aucun GPU.
# Le demi-flottant est celui de kernels (kernels/ref/half.h) : une seule implémentation.
ajp_add_module(gpu_ir
  SOURCES
    describe.cpp graph.cpp import.cpp import_attrs.cpp interpreter.cpp memory_plan.cpp
    onnx_reader.cpp op_elementwise.cpp op_pixel.cpp op_sample.cpp op_shape.cpp op_slice.cpp
    op_util.cpp op_window.cpp ops.cpp pass_cast.cpp pass_fold.cpp pass_fuse.cpp pass_layout.cpp pass_pixel.cpp
    pass_transpose.cpp protobuf.cpp shape.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::core_log ajp::kernels)

# L'interpréteur sert de référence : son arrondi ne dépend pas du compilateur.
if(NOT MSVC)
  target_compile_options(ajp_gpu_ir PRIVATE -ffp-contract=off)
endif()

# Inspection d'un modèle : opérateurs, formes et mémoire après passes.
add_executable(ajp_onnx_info onnx_info_main.cpp)
target_link_libraries(ajp_onnx_info PRIVATE ajp::gpu_ir ajp::kernels)
ajp_set_warnings(ajp_onnx_info)
