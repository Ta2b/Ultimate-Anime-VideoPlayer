// Couper TemporalFix en petits modèles (TemporalFix fusionné, kernels/spec/tfix_fused.h) : l'encodeur (une
// image), les trois blocs du flot et la tête de confiance (lots de six appuis), des convolutions que TensorRT
// construit en cinq moteurs ; tout le reste du graphe (flous, Resize, grilles, GridSample, Concat, mélange)
// est refait par les noyaux CUDA. La découpe ne s'applique qu'à un graphe dont la structure est reconnue :
// noms des tenseurs aux frontières, Concat d'entrée, temps des appuis, constantes du mélange, attributs des
// Resize et des GridSample. Sinon un refus dit ce qui manque, et la lecture garde le graphe entier.
// Aujourd'hui : s1 v1.1, s2 v1 et s3 v1 de pifroggi/vs_temporalfix (exports FP16, opset 18).
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace kirari::prepare {

// Les deux moteurs fusionnés : l'encodeur (une image ingérée → son tampon gardé) et la fenêtre.
enum class TfixRole : std::uint8_t { Encoder, Window };
std::string_view role_name(TfixRole role);  // « encoder », « window »

// Les tenseurs aux frontières des morceaux.
struct TfixSplit {
  std::string encode_input = "/Reshape_1_output_0";
  std::string encode_output = "/encode/Resize_output_0";
  // La dernière convolution de l'encodeur, que son Resize agrandit ×2 : là s'arrêtent les noyaux natifs.
  std::string encode_last = "/encode/proj/Conv_output_0";
  std::vector<std::string> block_input;   // X0, X1, X2
  std::vector<std::string> block_output;  // la dernière convolution de chaque bloc, avant son Resize
  std::string conf_input = "/conf_head/Concat_output_0";
  std::string conf_output = "/conf_head/net/net.2/Conv_output_0";
};

// Reconnaît le graphe. Refus : ModelUnreadable, FormatUnsupported (poids externes, graphes imbriqués),
// FieldMissing (structure inconnue).
std::expected<TfixSplit, Refusal> analyze_tfix(std::span<const std::byte> model);

struct TfixPiece {
  std::string name;  // « encode », « bloc0 », « bloc1 », « bloc2 », « conf »
  std::vector<std::byte> model;
};

// Les morceaux d'un rôle pour un réseau de height × width (tfix_size_valid) : l'encodeur seul, ou les trois
// blocs et la tête. Refus : ShapeInvalid pour une autre taille, et ceux de l'extraction.
std::expected<std::vector<TfixPiece>, Refusal> split_tfix(std::span<const std::byte> model,
                                                          const TfixSplit& split, TfixRole role, int height,
                                                          int width);

// Le manifeste (« .tfixfused ») que le backend lit pour construire et charger un moteur fusionné : le rôle,
// la taille du réseau, la demi-taille et les fichiers des morceaux, dans l'ordre de split_tfix.
inline constexpr std::string_view kTfixFusedExtension = ".tfixfused";

struct TfixFusedManifest {
  TfixRole role = TfixRole::Window;
  int height = 0;  // le réseau
  int width = 0;
  bool half = false;
  std::vector<std::string> pieces;
  std::vector<std::uint64_t> sizes;
  std::vector<std::string> digests;
};

// Les morceaux d'un rôle : 1 (encodeur) ou 4 (fenêtre).
std::size_t tfix_piece_count(TfixRole role);

std::string format_tfix_manifest(const TfixFusedManifest& manifest);
// Refus : FieldOutOfRange (champ absent, inconnu, taille que les noyaux ne prennent pas, morceaux en nombre).
std::expected<TfixFusedManifest, Refusal> parse_tfix_manifest(std::string_view text);

bool is_tfix_fused_manifest(const std::filesystem::path& path);

// Version de la dérivation : elle entre dans le nom des fichiers.
inline constexpr std::string_view kTfixFusedDerivation = "fusion1";

// Les morceaux d'un rôle et leur manifeste, écrits dans dir sous des noms qui portent la dérivation, le rôle,
// la taille, la demi-taille et l'empreinte du modèle ; ce qui s'y trouve déjà et reste intact est gardé.
// Rend le chemin du manifeste. Refus : ModelUnreadable, ceux d'analyze_tfix et split_tfix, FileUnwritable.
std::expected<std::filesystem::path, Refusal> derive_tfix_fused(const std::filesystem::path& onnx,
                                                                const std::filesystem::path& dir,
                                                                TfixRole role, int height, int width,
                                                                bool half);

}  // namespace kirari::prepare
