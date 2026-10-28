// Traduction des attributs ONNX en attributs typés de l'IR (interne à l'import).
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string_view>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/ir/onnx_model.h"

namespace ajp::gpu::ir::detail {

struct ImportedAttrs {
  Attrs attrs;
  // Anciens opsets : axes ou split passés en attribut deviennent l'entrée 1,
  // pour que l'IR ne connaisse que la forme récente.
  std::optional<Value> input_from_attr;
};

std::expected<ImportedAttrs, Refusal> import_attrs(OpKind op, const onnx::Node& node, std::int64_t opset);

std::expected<DataType, Refusal> data_type_of(std::int32_t onnx_type, std::string_view where);
std::expected<Value, Refusal> tensor_value(const onnx::Tensor& tensor);

// Valeur d'un nœud Constant (value, value_float(s), value_int(s)).
std::expected<Value, Refusal> constant_value(const onnx::Node& node);

}  // namespace ajp::gpu::ir::detail
