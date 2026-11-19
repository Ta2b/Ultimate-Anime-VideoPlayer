// Un ONNX décodé vu comme un graphe à parcourir, pour les lectures de GMFlow natif (docs/gmflow-natif.md) :
// les nœuds par nom, le producteur et les lecteurs de chaque tenseur, les constantes (initialiseurs et
// sorties de Constant, à travers Identity). Les tenseurs du modèle doivent vivre plus longtemps que la vue.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/ir/onnx_model.h"

namespace kirari::prepare {

class OnnxView {
 public:
  explicit OnnxView(const gpu::ir::onnx::Model& model);

  // Le nœud name s'il existe et fait op, sinon nul.
  const gpu::ir::onnx::Node* node(std::string_view name, std::string_view op) const;
  // Le nœud qui écrit tensor (nul : entrée du graphe ou constante).
  const gpu::ir::onnx::Node* producer(std::string_view tensor) const;
  // Les nœuds qui lisent tensor, dans l'ordre du graphe.
  std::vector<const gpu::ir::onnx::Node*> readers(std::string_view tensor) const;
  // La constante tensor (initialiseur ou sortie d'un Constant), à travers les Identity ; sinon nul.
  const gpu::ir::onnx::Tensor* constant(std::string_view tensor) const;
  // La constante entière ou flottante d'une seule valeur, s'il y en a une.
  std::optional<std::int64_t> integer(std::string_view tensor) const;
  std::optional<float> scalar(std::string_view tensor) const;
  // tensor sans les Cast, Unsqueeze et Identity qui l'écrivent : le premier tenseur calculé autrement.
  std::string_view base(std::string_view tensor) const;
  // Le producteur de base(tensor) s'il fait op, sinon nul.
  const gpu::ir::onnx::Node* made_by(std::string_view tensor, std::string_view op) const;

 private:
  std::map<std::string, const gpu::ir::onnx::Tensor*, std::less<>> constants_;
  std::map<std::string, const gpu::ir::onnx::Node*, std::less<>> nodes_;
  std::map<std::string, const gpu::ir::onnx::Node*, std::less<>> producers_;
  std::map<std::string, std::vector<const gpu::ir::onnx::Node*>, std::less<>> readers_;
};

// L'attribut name du nœud, ou nul.
const gpu::ir::onnx::Attribute* onnx_attribute(const gpu::ir::onnx::Node& node, std::string_view name);

}  // namespace kirari::prepare
