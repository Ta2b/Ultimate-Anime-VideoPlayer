// Règles de forme et calculs de chaque famille d'opérations (interne au module).
// Une famille par fichier op_*.cpp ; la table de ops.cpp les relie aux OpKind.
#pragma once

#include <span>

#include "gpu/ir/ops.h"

namespace kirari::gpu::ir::detail {

#define KIRARI_IR_OP(fn)                                      \
  Inferred infer_##fn(const Node&, std::span<const Operand>); \
  Evaluated eval_##fn(const Node&, std::span<const Value* const>, std::span<const OutputInfo>)

// op_elementwise.cpp
KIRARI_IR_OP(binary);      // Add, Sub, Mul, Div, Mod, PRelu
KIRARI_IR_OP(unary);       // Relu, LeakyRelu, Sigmoid, Tanh, Clip, Reciprocal, Identity, Cast
KIRARI_IR_OP(fake_quant);  // FakeQuantE4M3, FakeQuantInt8

// op_window.cpp
KIRARI_IR_OP(conv);
KIRARI_IR_OP(conv_transpose);
KIRARI_IR_OP(average_pool);

// op_shape.cpp
KIRARI_IR_OP(shape);
KIRARI_IR_OP(gather);
KIRARI_IR_OP(reshape);  // Reshape, Unsqueeze, Squeeze
KIRARI_IR_OP(transpose);
KIRARI_IR_OP(range);

// op_slice.cpp
KIRARI_IR_OP(concat);
KIRARI_IR_OP(split);
KIRARI_IR_OP(slice);
KIRARI_IR_OP(pad);

// op_pixel.cpp
KIRARI_IR_OP(depth_to_space);
KIRARI_IR_OP(space_to_depth);
KIRARI_IR_OP(reduce_mean);

// op_sample.cpp
KIRARI_IR_OP(resize);
KIRARI_IR_OP(grid_sample);

#undef KIRARI_IR_OP

}  // namespace kirari::gpu::ir::detail
