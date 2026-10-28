// GMFSS Fortuna (kernels/spec/gmfss.h) : ses deux graphes ONNX, exportés par
// tools/gmfss/export_gmfss.py et rangés dans un dossier des modèles RIFE, puis leurs variantes aux
// tailles fixées pour la taille remplie de la lecture (derived_model.h), que TensorRT construit
// sans profil.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/precision.h"
#include "core/types.h"
#include "prepare/onnx_dims.h"

namespace ajp::app::play {

// Les deux modèles de vsgmfss_fortuna : base, et union, dont la fusion mêle aussi l'image de
// RIFE (IFNet, dans le graphe « fusion », qui reçoit alors l'instant « time »).
enum class GmfssVariant : std::uint8_t { Base, Union };

inline constexpr std::string_view kGmfssPairFile = "gmfss_fortuna_base_pair.onnx";
inline constexpr std::string_view kGmfssFusionFile = "gmfss_fortuna_base_fusion.onnx";
inline constexpr std::string_view kGmfssUnionPairFile = "gmfss_fortuna_union_pair.onnx";
inline constexpr std::string_view kGmfssUnionFusionFile = "gmfss_fortuna_union_fusion.onnx";
// GMFlow à l'échelle 0,5 (export_gmfss.py --scale 0.5) : le flot à un quart de la taille de travail,
// 3,5 fois moins cher pour la paire (8,3 ms contre 29,4 à 720p sur une 4070 Ti) et la même qualité
// mesurée (21,2 dB sur 19 triplets de Re:Zero) ; plans remplis au multiple de 128. Préférés quand
// ils sont là.
inline constexpr std::string_view kGmfssHalfPairFile = "gmfss_fortuna_base_s05_pair.onnx";
inline constexpr std::string_view kGmfssHalfFusionFile = "gmfss_fortuna_base_s05_fusion.onnx";
inline constexpr std::string_view kGmfssUnionHalfPairFile = "gmfss_fortuna_union_s05_pair.onnx";
inline constexpr std::string_view kGmfssUnionHalfFusionFile = "gmfss_fortuna_union_s05_fusion.onnx";
inline constexpr int kGmfssHalfAlign = 128;
// Mode qualité : la paire d'un flot à demi-échelle dont GMFlow est calculé en FP32 (export_gmfss.py
// --scale 0.5 --precision flownet32). Son graphe « fusion » est celui du mode performance : seul le
// flot change, la fusion n'est pas écrite deux fois. Mesuré sur une 4070 Ti à 1280x768 : l'écart au
// GMFSS d'origine est divisé par 4, pour +7 ms par paire avec base et +10 ms avec union.
inline constexpr std::string_view kGmfssHalfQualityPairFile = "gmfss_fortuna_base_s05q_pair.onnx";
inline constexpr std::string_view kGmfssUnionHalfQualityPairFile = "gmfss_fortuna_union_s05q_pair.onnx";

// Cran Performance : la paire d'un flot à demi-échelle dont GMFlow est quantifié en FP8 (E4M3, nœuds
// Q/DQ), avec la fusion FP16 du cran Équilibre. Calcul en FP8 sur les cartes qui l'accélèrent (RTX 40
// et après) ; absente ou refusée par le moteur, la paire FP16 prend sa place.
inline constexpr std::string_view kGmfssHalfFp8PairFile = "gmfss_fortuna_base_s05fp8_pair.onnx";
inline constexpr std::string_view kGmfssUnionHalfFp8PairFile = "gmfss_fortuna_union_s05fp8_pair.onnx";

struct GmfssFiles {
  std::string_view pair;
  std::string_view fusion;
};
// tier : la paire du cran Performance (FP8) ou Qualité (FP32), flot à demi-échelle seulement, avec la
// fusion du cran Équilibre ; Équilibre (FP16) est la paire de base.
constexpr GmfssFiles gmfss_files(GmfssVariant variant, bool half_flow = false,
                                 PrecisionTier tier = PrecisionTier::Balanced) {
  const bool union_model = variant == GmfssVariant::Union;
  if (half_flow && tier == PrecisionTier::Quality) {
    return union_model ? GmfssFiles{kGmfssUnionHalfQualityPairFile, kGmfssUnionHalfFusionFile}
                       : GmfssFiles{kGmfssHalfQualityPairFile, kGmfssHalfFusionFile};
  }
  if (half_flow && tier == PrecisionTier::Performance) {
    return union_model ? GmfssFiles{kGmfssUnionHalfFp8PairFile, kGmfssUnionHalfFusionFile}
                       : GmfssFiles{kGmfssHalfFp8PairFile, kGmfssHalfFusionFile};
  }
  if (half_flow) {
    return union_model ? GmfssFiles{kGmfssUnionHalfPairFile, kGmfssUnionHalfFusionFile}
                       : GmfssFiles{kGmfssHalfPairFile, kGmfssHalfFusionFile};
  }
  return union_model ? GmfssFiles{kGmfssUnionPairFile, kGmfssUnionFusionFile}
                     : GmfssFiles{kGmfssPairFile, kGmfssFusionFile};
}

// Version du format des graphes, écrite par tools/gmfss/gmfss_graph.py (EXPORT_VERSION) dans la métadonnée
// « ajp.gmfss.export » de l'ONNX. 2 : la paire sort feature{k}_1 (64 canaux) et half{k} (RGB) séparés, sans
// le Concat 3 + 64 canaux de source{k}_1 de la version 1 (0,7 ms de moins par paire à 1280x768).
inline constexpr int kGmfssExportVersion = 2;
inline constexpr std::string_view kGmfssExportKey = "ajp.gmfss.export";

// Ce que le lecteur fait d'un graphe selon son export :
//  - Current : de la version courante ;
//  - Legacy : d'une version antérieure dont la lecture reste exacte (source{k}_1 en 67 canaux), lue telle
//    quelle en attendant le ré-export ;
//  - Stale : périmé, jamais utilisé : le résultat ou le coût de sa conversion est connu pour avoir changé
//    (la fenêtre locale de GMFlow lue par un GridSample de 159 millions de valeurs, 2 x 7,2 ms par paire à
//    1280x768, avant l'export par décalages entiers), ou d'un export plus récent que le lecteur.
// Le lecteur ne lance pas export_gmfss.py (Python, torch) : un graphe périmé se remplace en ré-exportant
// (tools/gmfss/export_gmfss.py) ou en téléchargeant la release des modèles d'interpolation.
enum class GmfssExport : std::uint8_t { Current, Legacy, Stale };

struct GmfssGraphState {
  GmfssExport state = GmfssExport::Current;
  int version = kGmfssExportVersion;  // 1 : sans métadonnée
  std::string reason;                 // Legacy et Stale : pourquoi, pour le journal
};

// L'état de la paire file, lu dans sa métadonnée « ajp.gmfss.export » (sans charger le graphe) ; sans elle,
// le graphe est lu une fois pour y chercher le GridSample de GMFlow (le résultat est gardé tant que le
// fichier ne change pas). Un fichier illisible n'est pas jugé ici : Current, la construction le refusera.
GmfssGraphState gmfss_graph_state(const std::filesystem::path& file);

struct GmfssModels {
  std::filesystem::path pair;
  std::filesystem::path fusion;
  int align = 64;  // multiple des plans remplis : 128 pour le flot à l'échelle 0,5
  PrecisionTier tier = PrecisionTier::Balanced;  // le cran de la paire rendue (FP8, FP16, FP32)
};

// Une paire périmée (GmfssExport::Stale) est journalisée (« graphe GMFSS périmé, ré-export ») puis écartée :
// le dossier ou l'essai suivant est cherché. Les deux graphes de variant du premier de ces dossiers qui les a
// tous les deux. Le flot à demi-échelle
// passe toujours avant le flot entier (prefer_half) : à qualité égale ou meilleure il coûte moitié moins
// (mesuré sur une 4070 Ti, 1280x768) ; le flot entier ne sert que de repli quand les graphes _s05
// manquent. tier demande d'abord la paire FP8 (Performance) ou FP32 (Qualité) du flot à demi-échelle ;
// sans elle, la paire FP16 prend sa place (GmfssModels::tier dit laquelle est rendue).
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs,
                                      GmfssVariant variant = GmfssVariant::Base, bool prefer_half = true,
                                      PrecisionTier tier = PrecisionTier::Balanced);

// Le cran tier (Performance ou Qualité) est-il installé : sa paire et la fusion s05 de variant dans un même
// dossier. Équilibre l'est dès que les graphes s05 le sont.
bool gmfss_tier_available(std::span<const std::filesystem::path> dirs, GmfssVariant variant,
                          PrecisionTier tier);

// Les crans installés de variant ; fp8_engine : le moteur calcule le FP8.
TierSupport gmfss_tiers(std::span<const std::filesystem::path> dirs, GmfssVariant variant, bool fp8_engine);

// Les dimensions nommées des deux graphes pour des plans remplis de padded_width × padded_height
// (multiples de 64) : « H », « W », puis « H/2 » à « W/8 ».
std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height);

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir.
// Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace ajp::app::play
