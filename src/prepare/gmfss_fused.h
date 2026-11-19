// La fusion de GMFSS Fortuna Union coupée pour les noyaux natifs (--gmfss-native, docs/gmfss-natif.md) : son
// IFNet (RIFE 4.26 : half0, half1, time → « rife ») extrait en un petit modèle que TensorRT construit, le
// GridNet refait par les noyaux CUDA avec les poids de la fusion entière. La fusion entière reste construite
// : elle est la référence de l'essai au chargement et le repli quand le natif est écarté.
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

// Le manifeste (« .gmfssfused ») que le backend lit pour construire et charger la fusion native : la taille
// de la fusion (la moitié des plans remplis), puis ses deux fichiers dans le dossier du manifeste, la fusion
// entière et l'IFNet, avec leurs tailles et leurs SHA-256.
inline constexpr std::string_view kGmfssFusedExtension = ".gmfssfused";

struct GmfssFusedManifest {
  int height = 0;
  int width = 0;
  std::vector<std::string> pieces;  // la fusion entière, puis l'IFNet
  std::vector<std::uint64_t> sizes;
  std::vector<std::string> digests;
};

// Le nom de la sortie de l'IFNet dans la fusion exportée par tools/gmfss/export_gmfss.py.
inline constexpr std::string_view kGmfssRifeTensor = "rife";

std::string format_gmfss_manifest(const GmfssFusedManifest& manifest);
// Refus : FieldOutOfRange (champ absent ou inconnu, taille hors des multiples de 32, morceaux en nombre).
std::expected<GmfssFusedManifest, Refusal> parse_gmfss_manifest(std::string_view text);

bool is_gmfss_fused_manifest(const std::filesystem::path& path);

// L'IFNet d'une fusion de height × width : half0 et half1 (1 × 3 × h × w), time (1 × 1 × 1 × 1) vers « rife
// ». Refus : ceux d'extract_subgraph.
std::expected<std::vector<std::byte>, Refusal> gmfss_ifnet(std::span<const std::byte> fusion, int height,
                                                           int width);

// Version de la dérivation : elle entre dans le nom des fichiers.
inline constexpr std::string_view kGmfssFusedDerivation = "natif1";

// L'IFNet et le manifeste de la fusion fusion (à forme fixe height × width), écrits à côté d'elle ; ce qui
// s'y trouve déjà et reste intact est gardé. Rend le chemin du manifeste. Refus : ModelUnreadable, ceux de
// gmfss_ifnet, FileUnwritable.
std::expected<std::filesystem::path, Refusal> derive_gmfss_fused(const std::filesystem::path& fusion,
                                                                 int height, int width);

// Le morceau piece du dossier dir (un manifeste « .gmfssfused » ou « .gmfsspair »), relu s'il a encore
// l'empreinte digest : le backend y lit les poids des réseaux natifs. Refus : ModelUnreadable (changé ou
// illisible), ceux de read_onnx_bytes.
std::expected<std::vector<std::byte>, Refusal> read_gmfss_piece(const std::filesystem::path& dir,
                                                                const std::string& piece,
                                                                const std::string& digest);

}  // namespace kirari::prepare
