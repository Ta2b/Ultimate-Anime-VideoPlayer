// Ce que les découpes de RIFE et de DistilDRBA partagent pour lire un graphe : son index par nom de nœud et
// par tenseur, la lecture d'un attribut texte et le nom d'un nœud de bloc. Interne à prepare/ (C04-09).
#pragma once

#include <format>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "prepare/onnx_split.h"

namespace ajp::prepare::split {

// Le graphe indexé par nom de nœud et par tenseur produit.
class Index {
 public:
  explicit Index(const OnnxGraph& graph) : graph_(graph) {
    for (const OnnxFullNode& node : graph.nodes) {
      by_name_.emplace(node.name, &node);
      for (const std::string& output : node.outputs) producer_.emplace(output, &node);
      for (const std::string& input : node.inputs) ++readers_[input];
    }
  }
  // Le nœud de ce nom, s'il fait cet opérateur quand il est donné.
  const OnnxFullNode* named(const std::string& name, std::string_view op = {}) const {
    const auto found = by_name_.find(name);
    return found != by_name_.end() && (op.empty() || found->second->op_type == op) ? found->second : nullptr;
  }
  const OnnxFullNode* producer(const std::string& tensor) const {
    const auto found = producer_.find(tensor);
    return found != producer_.end() ? found->second : nullptr;
  }
  // Le tenseur d'avant les Cast qui le produisent : tools/rife/convert_rife_fp16.py entoure Resize et
  // GridSample de Cast FP16 ↔ FP32, que les noyaux n'ont pas à refaire.
  const std::string& uncast(const std::string& tensor) const {
    const std::string* at = &tensor;
    for (const OnnxFullNode* node = producer(*at);
         node != nullptr && node->op_type == "Cast" && !node->inputs.empty(); node = producer(*at))
      at = &node->inputs[0];
    return *at;
  }
  // Le producteur du tenseur, Cast passés.
  const OnnxFullNode* source(const std::string& tensor) const { return producer(uncast(tensor)); }
  // Nombre de nœuds qui lisent le tenseur.
  int readers(const std::string& tensor) const {
    const auto found = readers_.find(tensor);
    return found != readers_.end() ? found->second : 0;
  }
  // Le tenseur et tout ce dont il dépend, de proche en proche jusqu'aux entrées du graphe.
  std::set<std::string, std::less<>> ancestors(const std::string& tensor) const {
    std::set<std::string, std::less<>> seen;
    std::vector<std::string> pending{tensor};
    while (!pending.empty()) {
      std::string current = std::move(pending.back());
      pending.pop_back();
      if (!seen.insert(current).second) continue;
      const OnnxFullNode* node = producer(current);
      if (node == nullptr) continue;
      for (const std::string& input : node->inputs) {
        if (!input.empty()) pending.push_back(input);
      }
    }
    return seen;
  }
  // Les valeurs du tenseur constant, vides s'il n'en est pas un.
  const std::vector<double>& constant(const std::string& tensor) const {
    static const std::vector<double> none;
    const auto found = graph_.constants.find(tensor);
    return found != graph_.constants.end() ? found->second : none;
  }

 private:
  const OnnxGraph& graph_;
  std::map<std::string, const OnnxFullNode*, std::less<>> by_name_;
  std::map<std::string, const OnnxFullNode*, std::less<>> producer_;
  std::map<std::string, int, std::less<>> readers_;
};

inline bool text_is(const OnnxFullNode& node, std::string_view attribute, std::string_view value) {
  const OnnxAttribute* found = node.attribute(attribute);
  return found != nullptr && found->text && *found->text == value;
}

inline std::string block_name(int b, std::string_view tail) {
  return std::format("/block{}/{}", b, tail);
}

}  // namespace ajp::prepare::split
