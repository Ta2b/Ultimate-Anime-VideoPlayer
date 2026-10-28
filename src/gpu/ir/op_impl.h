// Règles de forme et calculs de chaque famille d'opérations (interne au module).
// Une famille par fichier op_*.cpp ; la table de ops.cpp les relie aux OpKind.
#pragma once

#include <span>

#include "gpu/ir/ops.h"

namespace ajp::gpu::ir::detail {

#define AJP_IR_OP(fn)                                         \
  Inferred infer_##fn(const Node&, std::span<const Operand>); \
  Evaluated eval_##fn(const Node&, std::span<const Value* const>, std::span<const OutputInfo>)

// op_elementwise.cpp
AJP_IR_OP(binary);      // Add, Sub, Mul, Div, Mod, PRelu
AJP_IR_OP(unary);       // Relu, LeakyRelu, Sigmoid, Tanh, Clip, Reciprocal, Identity, Cast
AJP_IR_OP(fake_quant);  // FakeQuantE4M3

// op_window.cpp
AJP_IR_OP(conv);
AJP_IR_OP(conv_transpose);
AJP_IR_OP(average_pool);

// op_shape.cpp
AJP_IR_OP(shape);
AJP_IR_OP(gather);
AJP_IR_OP(reshape);  // Reshape, Unsqueeze, Squeeze
AJP_IR_OP(transpose);
AJP_IR_OP(range);

// op_slice.cpp
AJP_IR_OP(concat);
AJP_IR_OP(split);
AJP_IR_OP(slice);
AJP_IR_OP(pad);

// op_pixel.cpp
AJP_IR_OP(depth_to_space);
AJP_IR_OP(space_to_depth);
AJP_IR_OP(reduce_mean);

// op_sample.cpp
AJP_IR_OP(resize);
AJP_IR_OP(grid_sample);

#undef AJP_IR_OP

}  // namespace ajp::gpu::ir::detail
