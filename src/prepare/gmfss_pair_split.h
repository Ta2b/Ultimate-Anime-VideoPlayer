// La paire de GMFSS Fortuna Union coupée pour les noyaux natifs (--gmfss-native, docs/gmfss-natif.md) :
// GMFlow et ce qui prépare l'entrée du MetricNet (images réduites, backwarp, distances, cohérence des flots)
// extraits en un modèle que TensorRT construit (le « flot », FP16 ou FP8 selon la paire), le FeatureNet et
// les convolutions du MetricNet refaits par les noyaux CUDA (gmfss_pair_native.h).
//
// Le flot rend, à la place des quatorze sorties de la paire : l'entrée de la convolution metric_in (14 plans
// à la taille de la fusion), le flot de chaque côté à chaque niveau (les deux premiers plans de
// motion{k}_{l}, déjà réduits et mis à l'échelle) et half0, half1. Le natif écrit feature{k}_1, source{k}_2,
// source{k}_3, et le troisième plan de chaque motion{k}_{l} (la métrique, tanh × 10, réduite comme dans la
// paire).
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace kirari::prepare {

// Les tenseurs de la paire où passe la coupe, relevés dans son graphe : l'entrée de la convolution metric_in
// (le Concat de ses 14 plans), et la première entrée (le flot) du Concat qui écrit motion{k}_{l}, rangée
// k · 3 + l - 1. Des Q/DQ (paire FP8) devant ces tenseurs sont remontés.
inline constexpr int kGmfssPairFlows = 6;
struct GmfssPairCut {
  std::string metric_input;
  std::array<std::string, kGmfssPairFlows> flows;
};

// Refus : ModelUnreadable, FieldMissing (pas de metric_in, ou un motion{k}_{l} qui n'est pas un Concat de
// deux tenseurs, ou un nom de tenseur que le manifeste ne peut pas écrire).
std::expected<GmfssPairCut, Refusal> gmfss_pair_cut(std::span<const std::byte> pair);

// Le flot d'une paire dont la fusion est de height × width (frames : 1 × 6 × 2·height × 2·width) : sorties
// metric_input, les six flots, half0, half1. Refus : ceux d'extract_subgraph.
std::expected<std::vector<std::byte>, Refusal> gmfss_flow_piece(std::span<const std::byte> pair,
                                                                const GmfssPairCut& cut, int height,
                                                                int width);

// Le manifeste (« .gmfsspair ») que le backend lit : la taille de la fusion, ses deux fichiers dans le
// dossier du manifeste (la paire FP16 dont on lit les poids, puis le flot), leurs tailles et leurs SHA-256,
// et les noms des sorties du flot.
inline constexpr std::string_view kGmfssPairExtension = ".gmfsspair";

struct GmfssPairManifest {
  int height = 0;
  int width = 0;
  std::vector<std::string> pieces;  // la paire des poids, puis le flot
  std::vector<std::uint64_t> sizes;
  std::vector<std::string> digests;
  GmfssPairCut cut;
};

std::string format_gmfss_pair_manifest(const GmfssPairManifest& manifest);
// Refus : FieldOutOfRange (champ absent ou inconnu, taille hors des multiples de 32, morceaux ou flots en
// nombre).
std::expected<GmfssPairManifest, Refusal> parse_gmfss_pair_manifest(std::string_view text);

bool is_gmfss_pair_manifest(const std::filesystem::path& path);

// Version de la dérivation : elle entre dans le nom des fichiers.
inline constexpr std::string_view kGmfssPairDerivation = "natif1";

// Le flot de la paire pair (la paire FP16, ou FP8 : celle dont TensorRT construit GMFlow) et le manifeste,
// écrits à côté d'elle ; weights est la paire FP16 dont le natif lit les poids (pair elle-même quand elle est
// en FP16), dans le même dossier. Ce qui s'y trouve déjà et reste intact est gardé. Rend le chemin du
// manifeste. Refus : ShapeInvalid (taille), ModelUnreadable, ceux de gmfss_pair_cut et de gmfss_flow_piece,
// FileUnwritable.
std::expected<std::filesystem::path, Refusal> derive_gmfss_pair(const std::filesystem::path& pair,
                                                                const std::filesystem::path& weights,
                                                                int height, int width);

}  // namespace kirari::prepare
