// Couper un RIFE ONNX en six petits modèles (RIFE fusionné, kernels/spec/rife_fused.h) : l'encodeur et un
// modèle par bloc, des convolutions à basse résolution que TensorRT construit en six moteurs ; tout le
// reste du graphe (DepthToSpace, Resize, flot, grille, GridSample, Concat, mélange) est refait par des
// noyaux CUDA. La découpe ne s'applique qu'à un graphe dont toute la structure est reconnue : noms des
// tenseurs aux frontières, opérateurs, échelles des Resize, constantes des Mul et des Split, attributs de
// Resize et de GridSample. Sinon un refus dit ce qui manque et la lecture garde le RIFE entier.
// Aujourd'hui : RIFE 4.25 (5 blocs) et 4.22 lite (4 blocs), exports de vs-mlrt à 11 plans, FP16.
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
#include "kernels/spec/rife_fused.h"

namespace ajp::prepare {

struct RifeSplit {
  kernels::RifeFusedPlan plan;
  std::string encode_input;                                        // 2 × 3 × H × W : les images A et B
  std::string encode_output;                                       // 2 × 4 × H × W : leurs encodages
  std::array<std::string, kernels::kFusedMaxBlocks> block_input;   // X_k
  std::array<std::string, kernels::kFusedMaxBlocks> block_output;  // Y_k
  std::int32_t elem_type = 10;                                     // type ONNX des frontières : FLOAT16
};

// Reconnaît le graphe. Refus : ModelUnreadable, InputMismatch (entrée qui n'est pas 1 × 11 × H × W en
// FP16), FormatUnsupported (poids externes, graphes imbriqués), FieldMissing (structure inconnue).
std::expected<RifeSplit, Refusal> analyze_rife(std::span<const std::byte> model);

// Un des morceaux, un modèle ONNX à forme d'entrée fixe.
struct RifePiece {
  std::string name;  // « encode », « bloc0 »…
  std::vector<std::byte> model;
};

// « encode » pour 0, « bloc<k-1> » pour k ≥ 1.
std::string rife_piece_name(int index);

// Les 1 + blocs morceaux pour une taille remplie (hauteur et largeur que le découpage prend :
// rife_fused_size_valid). Refus : ShapeInvalid pour une autre taille, et ceux de l'extraction.
std::expected<std::vector<RifePiece>, Refusal> split_rife(std::span<const std::byte> model,
                                                          const RifeSplit& split, int height, int width);

// Ce que le backend lit pour construire et charger RIFE fusionné : un petit fichier texte (« .rifefused ») à
// côté des six modèles qu'il nomme. Il porte la taille remplie, le découpage, ce que la chaîne sort (l'image,
// ou le flot et le masque de la variante mouvement) et les fichiers des morceaux, dans l'ordre : encodeur,
// blocs 0 à N-1.
inline constexpr std::string_view kRifeFusedExtension = ".rifefused";

enum class RifeFusedOutput : std::uint8_t { Image, Motion };

struct RifeFusedManifest {
  RifeFusedOutput output = RifeFusedOutput::Image;
  int height = 0;
  int width = 0;
  kernels::RifeFusedPlan plan;
  std::vector<std::string> pieces;  // 1 + blocs noms de fichier, sans dossier
};

std::string format_manifest(const RifeFusedManifest& manifest);
// Refus : FieldMissing, FieldOutOfRange (champ absent, inconnu, ou découpage/taille que les noyaux ne
// prennent pas).
std::expected<RifeFusedManifest, Refusal> parse_manifest(std::string_view text);

// Le fichier d'un chemin : « .rifefused » ?
bool is_rife_fused_manifest(const std::filesystem::path& path);

// Version de la dérivation : elle entre dans le nom des fichiers, une règle de coupe changée en refait
// d'autres.
inline constexpr std::string_view kRifeFusedDerivation = "fusion1";

// Les six morceaux de rife_onnx pour une taille remplie et leur manifeste, écrits dans dir (créé s'il manque)
// sous des noms qui portent la version de la dérivation, la taille et l'empreinte du modèle ; ce qui s'y
// trouve déjà est gardé (écriture à côté puis renommage : jamais un fichier à moitié écrit). Rend le chemin
// du manifeste. Refus : ModelUnreadable, ceux d'analyze_rife et split_rife, FileUnwritable.
std::expected<std::filesystem::path, Refusal> derive_rife_fused(const std::filesystem::path& rife_onnx,
                                                                const std::filesystem::path& dir, int height,
                                                                int width, RifeFusedOutput output);

}  // namespace ajp::prepare
