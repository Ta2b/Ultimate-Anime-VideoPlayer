// Couper le graphe « paire » de GMFSS Fortuna (kernels/spec/gmfss.h, « GMFSS fusionné ») en deux modèles :
// les caractéristiques d'une seule image (FeatureNet, un réseau de convolutions qui ne lit ni la seconde
// image ni le flot) et tout le reste (flot de GMFlow dans les deux sens, MetricNet, images réduites de
// moitié). Les caractéristiques d'une image servent à deux paires de suite (côté b de l'une, côté a de la
// suivante) : le lecteur les garde au lieu de les refaire, un FeatureNet de moins par paire. Le graphe «
// fusion » reste tel quel. La découpe ne s'applique qu'à un graphe dont la structure est reconnue : noms et
// constantes des Slice de « frames », sorties d'un export de version 2, mêmes poids aux deux côtés de
// FeatureNet. Sinon un refus dit ce qui manque et la lecture garde la paire entière.
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

namespace ajp::prepare {

// Les deux morceaux, des modèles ONNX à formes d'entrée fixes.
struct GmfssPieces {
  std::vector<std::byte> features;  // une image RGB 1 × 3 × H × W -> kGmfssFeatureOutputs
  std::vector<std::byte> flow;      // « frames » -> motion{k}_{l} (6), half{k} (2)
};

// Les tenseurs aux frontières des morceaux, tels que le graphe d'origine les nomme (ils restent ceux des
// morceaux). Les sorties du morceau des caractéristiques sont celles du côté a de la paire, dans l'ordre des
// niveaux 1 à 3.
inline constexpr std::string_view kGmfssFeatureInput = "/Slice_output_0";  // l'image a de « frames »
inline constexpr std::array<std::string_view, 3> kGmfssFeatureOutputs = {"feature0_1", "source0_2",
                                                                         "source0_3"};

// Reconnaît la paire. Refus : ModelUnreadable, InputMismatch (entrée qui n'est pas « frames » en 1 × 6 × H ×
// W FP16), FormatUnsupported (poids externes, graphes imbriqués), FieldMissing (structure inconnue, export
// d'une autre version que 2).
std::expected<void, Refusal> analyze_gmfss(std::span<const std::byte> pair);

// Les deux morceaux pour des plans remplis de width × height. Refus : ShapeInvalid pour une taille qui n'est
// pas un multiple de 8, ceux d'analyze_gmfss et de l'extraction.
std::expected<GmfssPieces, Refusal> split_gmfss(std::span<const std::byte> pair, int height, int width);

// Version de la dérivation : elle entre dans le nom des fichiers, une règle de coupe changée en refait
// d'autres.
inline constexpr std::string_view kGmfssFusedDerivation = "fusion1";

struct GmfssFusedFiles {
  std::filesystem::path features;
  std::filesystem::path flow;
};

// Les deux morceaux de pair_onnx pour des plans remplis de width × height, écrits dans dir (créé s'il manque)
// sous des noms qui portent la version de la dérivation, la taille et l'empreinte du modèle ; ce qui s'y
// trouve déjà est gardé (écriture à côté puis renommage : jamais un fichier à moitié écrit). Refus :
// ModelUnreadable, ceux d'analyze_gmfss et split_gmfss, FileUnwritable.
std::expected<GmfssFusedFiles, Refusal> derive_gmfss_fused(const std::filesystem::path& pair_onnx,
                                                           const std::filesystem::path& dir, int height,
                                                           int width);

}  // namespace ajp::prepare
