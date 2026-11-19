#include "prepare/gmflow_onnx_view.h"

#include <algorithm>

namespace kirari::prepare {

namespace onnx = gpu::ir::onnx;

OnnxView::OnnxView(const onnx::Model& model) {
  for (const onnx::Tensor& tensor : model.graph.initializers) constants_.emplace(tensor.name, &tensor);
  for (const onnx::Node& node : model.graph.nodes) {
    nodes_.emplace(node.name, &node);
    for (const std::string& out : node.outputs) producers_.emplace(out, &node);
    for (const std::string& in : node.inputs) {
      if (!in.empty()) readers_[in].push_back(&node);
    }
    if (node.op_type != "Constant" || node.outputs.size() != 1) continue;
    for (const onnx::Attribute& a : node.attributes) {
      if (a.name == "value" && a.tensors.size() == 1) constants_.emplace(node.outputs[0], a.tensors.data());
    }
  }
}

const onnx::Node* OnnxView::node(std::string_view name, std::string_view op) const {
  const auto found = nodes_.find(name);
  return found != nodes_.end() && found->second->op_type == op ? found->second : nullptr;
}

const onnx::Node* OnnxView::producer(std::string_view tensor) const {
  const auto found = producers_.find(tensor);
  return found != producers_.end() ? found->second : nullptr;
}

std::vector<const onnx::Node*> OnnxView::readers(std::string_view tensor) const {
  const auto found = readers_.find(tensor);
  return found != readers_.end() ? found->second : std::vector<const onnx::Node*>{};
}

const onnx::Tensor* OnnxView::constant(std::string_view tensor) const {
  for (int hops = 0; hops < 8; ++hops) {
    if (const auto found = constants_.find(tensor); found != constants_.end()) return found->second;
    const onnx::Node* p = producer(tensor);
    if (p == nullptr || p->op_type != "Identity" || p->inputs.size() != 1) return nullptr;
    tensor = p->inputs[0];
  }
  return nullptr;
}

std::optional<std::int64_t> OnnxView::integer(std::string_view tensor) const {
  const onnx::Tensor* t = constant(base(tensor));
  if (t == nullptr || t->ints.size() != 1) return std::nullopt;
  return t->ints[0];
}

std::optional<float> OnnxView::scalar(std::string_view tensor) const {
  const onnx::Tensor* t = constant(base(tensor));
  if (t == nullptr || t->floats.size() != 1) return std::nullopt;
  return t->floats[0];
}

std::string_view OnnxView::base(std::string_view tensor) const {
  for (int hops = 0; hops < 16; ++hops) {
    const onnx::Node* p = producer(tensor);
    if (p == nullptr || p->inputs.empty() ||
        (p->op_type != "Cast" && p->op_type != "Unsqueeze" && p->op_type != "Identity"))
      return tensor;
    tensor = p->inputs[0];
  }
  return tensor;
}

const onnx::Node* OnnxView::made_by(std::string_view tensor, std::string_view op) const {
  const onnx::Node* p = producer(base(tensor));
  return p != nullptr && p->op_type == op ? p : nullptr;
}

const onnx::Attribute* onnx_attribute(const onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &onnx::Attribute::name);
  return found != node.attributes.end() ? &*found : nullptr;
}

}  // namespace kirari::prepare
