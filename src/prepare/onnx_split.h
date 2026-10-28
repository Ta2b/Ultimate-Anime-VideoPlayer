// Lire le graphe d'un ONNX de plus près (attributs, constantes) et en extraire un morceau, au niveau du
// format protobuf, sans dépendance onnx : de quoi couper RIFE en six petits modèles (rife_split.h). Les
// nœuds et les poids du morceau sont recopiés octet pour octet.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

// Un attribut de nœud, ses valeurs entières ou textuelles seulement (les flottants et les tenseurs ne servent
// pas aux vérifications).
struct OnnxAttribute {
  std::string name;
  std::optional<std::int64_t> integer;
  std::optional<std::string> text;
};

struct OnnxFullNode {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<OnnxAttribute> attributes;
  // Le nœud Constant : les valeurs de son tenseur, en double (flottants 32 ou 16 bits) ; vide sinon.
  std::vector<double> constant;

  const OnnxAttribute* attribute(std::string_view attribute_name) const;
};

struct OnnxGraph {
  std::vector<OnnxFullNode> nodes;
  // Valeurs flottantes des tenseurs constants du graphe (nœuds Constant et initialiseurs de moins de
  // kMaxConstantElements valeurs), par nom de tenseur.
  std::map<std::string, std::vector<double>, std::less<>> constants;
  std::vector<std::string> initializers;  // noms de tous les initialiseurs
  bool has_external_data = false;         // un initialiseur garde ses poids dans un autre fichier
  bool has_sparse_initializers = false;
  bool has_subgraphs = false;  // un attribut de nœud porte un graphe (If, Loop…)
};

inline constexpr std::size_t kMaxConstantElements = 16;

// Le graphe principal. Refus : ModelUnreadable (protobuf malformé, pas de graphe).
std::expected<OnnxGraph, Refusal> read_onnx_graph(std::span<const std::byte> model);

// Un tenseur à la frontière d'un morceau : nom, type ONNX (10 : FLOAT16) et forme.
struct SplitTensor {
  std::string name;
  std::int32_t elem_type = 10;
  std::vector<std::int64_t> dims;  // vide : forme laissée à TensorRT (sorties)
};

// Le modèle réduit aux nœuds qui mènent des inputs aux outputs : entrées du graphe = inputs (formes fixes),
// sorties = outputs, nœuds Constant et initialiseurs repris tant qu'ils servent. Les autres champs du modèle
// (version, jeux d'opérateurs) restent. Refus : ModelUnreadable, FieldMissing (un output n'est produit par
// aucun nœud, ou un nœud du morceau lit un tenseur qui n'est ni input, ni constante, ni produit),
// FormatUnsupported (poids externes, initialiseurs creux, graphes imbriqués : non repris).
std::expected<std::vector<std::byte>, Refusal> extract_subgraph(std::span<const std::byte> model,
                                                                std::span<const SplitTensor> inputs,
                                                                std::span<const SplitTensor> outputs,
                                                                std::string_view graph_name);

}  // namespace ajp::prepare
