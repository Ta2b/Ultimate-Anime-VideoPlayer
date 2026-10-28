// Couper un DistilDRBA ONNX (v2 lite, export de tools/drba/export_distil_drba.py converti en FP16) en quatre
// petits modèles (DistilDRBA fusionné, kernels/spec/drba_fused.h) : l'encodeur, en lot de trois images, et un
// modèle par bloc, des convolutions à basse résolution que TensorRT construit en quatre moteurs ; tout le
// reste du graphe (réductions d'images, choix du côté de la paire, Resize, DepthToSpace, flot, grille,
// GridSample, Concat, mélange) est refait par des noyaux CUDA. La découpe ne s'applique qu'à un graphe dont
// toute la structure est reconnue : nœuds et câblage de la colle, échelles des Resize, constantes des Mul et
// des Slice, attributs de Resize, de GridSample et de DepthToSpace. Sinon un refus dit ce qui manque et la
// lecture garde le DistilDRBA entier.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "kernels/spec/drba_fused.h"

namespace ajp::prepare {

struct DrbaSplit {
  kernels::DrbaFusedPlan plan;
  std::string encode_input;   // une image réduite de t : 1 × 3 × H/t × W/t (le lot de 3 en prend trois)
  std::string encode_output;  // son encodage : 1 × 16 × H/t × W/t
  std::array<std::string, kernels::kDrbaFusedBlocks> block_input;   // X_k
  std::array<std::string, kernels::kDrbaFusedBlocks> block_output;  // Y_k
  std::int32_t elem_type = 10;                                      // type ONNX des frontières : FLOAT16
};

// Un nœud de la colle que la reconnaissance exige, avec ses entrées exactes (« » : tenseur sans contrôle).
struct DrbaExpectedNode {
  std::string name;
  std::string op;
  std::vector<std::string> inputs;
};
// Une constante exigée : l'entrée `input` du nœud `node` vaut `values`.
struct DrbaExpectedConstant {
  std::string node;
  std::size_t input = 0;
  std::vector<double> values;
};
struct DrbaExpectedWiring {
  std::vector<DrbaExpectedNode> nodes;
  std::vector<DrbaExpectedConstant> constants;
};
// Le câblage de la colle du graphe de référence pour les réductions t = 4 ou 8 (les tests en fabriquent un
// graphe).
DrbaExpectedWiring drba_expected_wiring(int tail);

// Reconnaît le graphe. Refus : ModelUnreadable, InputMismatch (entrées qui ne sont pas « pair01 » et « pair12
// » en 1 × 7 × H × W FP16), FormatUnsupported (poids externes, graphes imbriqués), FieldMissing (structure
// inconnue).
std::expected<DrbaSplit, Refusal> analyze_drba(std::span<const std::byte> model);

// Un des morceaux, un modèle ONNX à forme d'entrée fixe.
struct DrbaPiece {
  std::string name;  // « encode », « bloc0 »…
  std::vector<std::byte> model;
};

// « encode » pour 0, « bloc<k-1> » pour k ≥ 1.
std::string drba_piece_name(int index);

// Les quatre morceaux pour une taille remplie (drba_fused_size_valid). Refus : ShapeInvalid pour une autre
// taille, et ceux de l'extraction.
std::expected<std::vector<DrbaPiece>, Refusal> split_drba(std::span<const std::byte> model,
                                                          const DrbaSplit& split, int height, int width);

// Ce que le backend lit pour construire et charger DistilDRBA fusionné : un petit fichier texte (« .drbafused
// ») à côté des quatre modèles qu'il nomme, dans l'ordre : encodeur, blocs 0 à 2.
inline constexpr std::string_view kDrbaFusedExtension = ".drbafused";

struct DrbaFusedManifest {
  int height = 0;
  int width = 0;
  kernels::DrbaFusedPlan plan;
  std::vector<std::string> pieces;  // 4 noms de fichier, sans dossier
};

std::string format_drba_manifest(const DrbaFusedManifest& manifest);
// Refus : FieldMissing, FieldOutOfRange (champ absent, inconnu, ou découpage/taille que les noyaux ne
// prennent pas).
std::expected<DrbaFusedManifest, Refusal> parse_drba_manifest(std::string_view text);

// Le fichier d'un chemin : « .drbafused » ?
bool is_drba_fused_manifest(const std::filesystem::path& path);

// Version de la dérivation : elle entre dans le nom des fichiers, une règle de coupe changée en refait
// d'autres.
inline constexpr std::string_view kDrbaFusedDerivation = "fusion1";

// Les quatre morceaux de drba_onnx pour une taille remplie et leur manifeste, écrits dans dir (créé s'il
// manque) sous des noms qui portent la version de la dérivation, la taille et l'empreinte du modèle ; ce qui
// s'y trouve déjà est gardé (écriture à côté puis renommage : jamais un fichier à moitié écrit). Rend le
// chemin du manifeste. Refus : ModelUnreadable, ceux d'analyze_drba et split_drba, FileUnwritable.
std::expected<std::filesystem::path, Refusal> derive_drba_fused(const std::filesystem::path& drba_onnx,
                                                                const std::filesystem::path& dir, int height,
                                                                int width);

}  // namespace ajp::prepare
