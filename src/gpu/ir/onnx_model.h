// Sous-ensemble d'un ModelProto ONNX, tel que lu par read_onnx().
//
// Ces structures recopient le fichier sans l'interpréter : les opérateurs, les
// opsets et les attributs restent des chaînes et des nombres ONNX. L'import
// (import.h) les traduit ensuite en représentation interne.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::ir::onnx {

// Valeurs de TensorProto.DataType lues par le décodeur ; les autres sont refusées.
inline constexpr std::int32_t kFloat = 1;
inline constexpr std::int32_t kInt32 = 6;
inline constexpr std::int32_t kInt64 = 7;
inline constexpr std::int32_t kFloat16 = 10;

// Un TensorProto décodé. Les flottants (FLOAT, FLOAT16) vont dans floats, les
// entiers (INT64, INT32) dans ints ; la conversion depuis FLOAT16 est exacte.
struct Tensor {
  std::string name;
  std::int32_t data_type = 0;
  std::vector<std::int64_t> dims;
  std::vector<float> floats;
  std::vector<std::int64_t> ints;
};

// Valeurs de AttributeProto.AttributeType.
enum class AttributeType : std::uint8_t {
  Undefined = 0,
  Float = 1,
  Int = 2,
  String = 3,
  Tensor = 4,
  Graph = 5,
  Floats = 6,
  Ints = 7,
  Strings = 8,
  Tensors = 9,
  Graphs = 10,
  SparseTensor = 11,
  SparseTensors = 12,
  TypeProto = 13,
  TypeProtos = 14,
};

struct Attribute {
  std::string name;
  AttributeType type = AttributeType::Undefined;
  float f = 0.0F;
  std::int64_t i = 0;
  std::string s;
  std::vector<Tensor> tensors;  // t pour Tensor, tensors pour Tensors
  std::vector<float> floats;
  std::vector<std::int64_t> ints;
  std::vector<std::string> strings;
};

struct Node {
  std::string name;
  std::string op_type;
  std::string domain;               // vide ou "ai.onnx" : domaine par défaut
  std::vector<std::string> inputs;  // "" : entrée facultative absente
  std::vector<std::string> outputs;
  std::vector<Attribute> attributes;
};

// Forme déclarée d'une entrée ou d'une sortie. Une dimension symbolique vaut -1
// et garde son nom dans symbols ; has_shape est faux si le rang est inconnu.
struct ValueInfo {
  std::string name;
  std::int32_t elem_type = 0;
  bool has_shape = false;
  std::vector<std::int64_t> dims;
  std::vector<std::string> symbols;
};

struct Graph {
  std::string name;
  std::vector<Node> nodes;
  std::vector<Tensor> initializers;
  std::vector<ValueInfo> inputs;
  std::vector<ValueInfo> outputs;
};

struct OpsetId {
  std::string domain;
  std::int64_t version = 0;
};

struct Model {
  std::int64_t ir_version = 0;
  std::string producer_name;
  std::string producer_version;
  std::vector<OpsetId> opsets;
  Graph graph;
};

// Décode un ModelProto. Refuse les octets malformés (OnnxMalformed), les poids
// externes (OnnxExternalData) et les types hors du sous-ensemble (OnnxUnsupported).
std::expected<Model, Refusal> read_onnx(std::span<const std::byte> bytes);

}  // namespace ajp::gpu::ir::onnx
