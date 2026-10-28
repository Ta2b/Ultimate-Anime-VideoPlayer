#include "prepare/onnx_blend.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "prepare/protobuf_fields.h"

namespace ajp::prepare {
namespace {

// Numéros des champs d'onnx.proto.
constexpr std::uint32_t kModelIrVersion = 1;
constexpr std::uint32_t kModelProducerName = 2;
constexpr std::uint32_t kModelProducerVersion = 3;
constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kModelOpset = 8;
constexpr std::uint32_t kOpsetVersion = 2;
constexpr std::uint32_t kGraphNode = 1;
constexpr std::uint32_t kGraphName = 2;
constexpr std::uint32_t kGraphInitializer = 5;
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kNodeInput = 1;
constexpr std::uint32_t kNodeOutput = 2;
constexpr std::uint32_t kNodeOpType = 4;
constexpr std::uint32_t kNodeAttribute = 5;
constexpr std::uint32_t kAttributeName = 1;
constexpr std::uint32_t kAttributeInt = 3;
constexpr std::uint32_t kAttributeType = 20;
constexpr std::uint32_t kAttributeTypeInt = 2;
constexpr std::uint32_t kTensorDims = 1;
constexpr std::uint32_t kTensorDataType = 2;
constexpr std::uint32_t kTensorName = 8;
constexpr std::uint32_t kTensorRawData = 9;
constexpr std::uint32_t kValueName = 1;
constexpr std::uint32_t kValueType = 2;
constexpr std::uint32_t kTypeTensor = 1;
constexpr std::uint32_t kTensorElemType = 1;
constexpr std::uint32_t kTensorShape = 2;
constexpr std::uint32_t kShapeDim = 1;
constexpr std::uint32_t kDimValue = 1;
constexpr std::uint32_t kDimParam = 2;
constexpr std::uint64_t kInt64 = 7;
constexpr std::uint64_t kFloat16 = 10;
// Split prend ses tailles en entrée depuis l'opset 13 ; IR 7 est celle de cet opset.
constexpr std::uint64_t kIrVersion = 7;
constexpr std::uint64_t kOpset = 13;

class Message {
 public:
  Message& varint(std::uint32_t number, std::uint64_t value) {
    protobuf::put_varint(out_, std::uint64_t{number} << 3U);
    protobuf::put_varint(out_, value);
    return *this;
  }
  Message& bytes(std::uint32_t number, std::span<const std::byte> data) {
    protobuf::put_delimited(out_, number, data);
    return *this;
  }
  Message& text(std::uint32_t number, std::string_view value) {
    return bytes(number, std::as_bytes(std::span(value)));
  }
  Message& message(std::uint32_t number, const Message& inner) { return bytes(number, inner.out_); }
  std::vector<std::byte> take() && { return std::move(out_); }

 private:
  std::vector<std::byte> out_;
};

// 1 × planes × height × width en fp16, height et width nommés.
Message planes_value(std::string_view name, std::uint64_t planes) {
  Message shape;
  shape.message(kShapeDim, Message().varint(kDimValue, 1))
      .message(kShapeDim, Message().varint(kDimValue, planes))
      .message(kShapeDim, Message().text(kDimParam, "height"))
      .message(kShapeDim, Message().text(kDimParam, "width"));
  Message tensor;
  tensor.varint(kTensorElemType, kFloat16).message(kTensorShape, shape);
  Message value;
  value.text(kValueName, name).message(kValueType, Message().message(kTypeTensor, tensor));
  return value;
}

Message node(std::string_view op, std::span<const std::string_view> inputs,
             std::span<const std::string_view> outputs) {
  Message n;
  for (const std::string_view in : inputs) n.text(kNodeInput, in);
  for (const std::string_view out : outputs) n.text(kNodeOutput, out);
  n.text(kNodeOpType, op);
  return n;
}

}  // namespace

std::vector<std::byte> blend_model(std::string_view producer_name, std::string_view producer_version) {
  // Les tailles du découpage : 3 plans de a, 3 de b, 1 de t, en int64 petit-boutiste.
  std::array<std::byte, 24> sizes{};
  for (std::size_t i = 0; i < 3; ++i) sizes[i * 8] = static_cast<std::byte>(i < 2 ? 3 : 1);
  Message split_sizes;
  split_sizes.varint(kTensorDims, 3)
      .varint(kTensorDataType, kInt64)
      .text(kTensorName, "split")
      .bytes(kTensorRawData, sizes);

  Message split = node("Split", std::array<std::string_view, 2>{"input", "split"},
                       std::array<std::string_view, 3>{"a", "b", "t"});
  split.message(kNodeAttribute, Message()
                                    .text(kAttributeName, "axis")
                                    .varint(kAttributeInt, 1)
                                    .varint(kAttributeType, kAttributeTypeInt));
  Message graph;
  graph.message(kGraphNode, split)
      .message(kGraphNode, node("Sub", std::array<std::string_view, 2>{"b", "a"},
                                std::array<std::string_view, 1>{"step"}))
      .message(kGraphNode, node("Mul", std::array<std::string_view, 2>{"step", "t"},
                                std::array<std::string_view, 1>{"part"}))
      .message(kGraphNode, node("Add", std::array<std::string_view, 2>{"a", "part"},
                                std::array<std::string_view, 1>{"output"}))
      .text(kGraphName, "blend")
      .message(kGraphInitializer, split_sizes)
      .message(kGraphInput, planes_value("input", 7))
      .message(kGraphOutput, planes_value("output", 3));
  Message model;
  model.varint(kModelIrVersion, kIrVersion)
      .text(kModelProducerName, producer_name)
      .text(kModelProducerVersion, producer_version)
      .message(kModelGraph, graph)
      .message(kModelOpset, Message().varint(kOpsetVersion, kOpset));
  return std::move(model).take();
}

}  // namespace ajp::prepare
