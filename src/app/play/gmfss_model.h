// GMFSS Fortuna Union (kernels/spec/gmfss.h) : ses deux graphes ONNX, exportés par
// tools/gmfss/export_gmfss.py (poids de la roue vsgmfss_fortuna 1.0.0, MIT ; flot entier) et rangés dans le
// dossier de --gmfss-dir, un dossier des modèles RIFE ou son voisin « gmfss », puis leurs variantes aux
// tailles fixées pour les plans remplis de la lecture (derived_model.h), que TensorRT construit sans profil.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace kirari::app::play {

inline constexpr std::string_view kGmfssPairFile = "gmfss_fortuna_union_pair.onnx";
inline constexpr std::string_view kGmfssFusionFile = "gmfss_fortuna_union_fusion.onnx";
// La paire au demi-flot (export_gmfss.py --scale 0.5) : mêmes entrées et sorties, GMFlow sur les images
// réduites de moitié ; la même fusion la suit (PlaySettings::gmfss_half_flow).
inline constexpr std::string_view kGmfssHalfPairFile = "gmfss_fortuna_union_s05_pair.onnx";

struct GmfssModels {
  std::filesystem::path pair;  // flot entier trouvé, ou la paire choisie par find_usable_gmfss
  std::filesystem::path fusion;
  std::filesystem::path half_pair;  // vide sans graphe au demi-flot dans ce dossier
};

// Plafond de cadence Auto de GMFSS seul, avant la limite de l'écran ou du réglage.
int gmfss_frame_limit(double source_fps, bool half);

// Où chercher les graphes : chaque dossier, puis son voisin « gmfss » (~/kirari/models/gmfss à côté de
// ~/kirari/models/rife, le --out que propose tools/gmfss/export_gmfss.py), sans doublon.
std::vector<std::filesystem::path> gmfss_places(std::span<const std::filesystem::path> dirs);

// Le premier dossier avec la fusion et au moins une paire, d'après les noms des fichiers.
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs);

// Les deux graphes à construire : le flot demandé s'il est là, sinon celui présent.
GmfssModels gmfss_chosen(const GmfssModels& found, bool half);

// Les sorties de la paire que l'étage lit au niveau 1 (format 2 d'export_gmfss.py, métadonnée
// « ajp.gmfss.export ») : un export d'avant (format 1, source{k}_1 de 67 canaux) ne les a pas.
inline constexpr std::array<std::string_view, 4> kGmfssPairLevel1{"feature0_1", "feature1_1", "half0",
                                                                  "half1"};

// La paire a-t-elle les sorties de kGmfssPairLevel1 ? Refus : ceux de prepare::read_onnx_bytes et
// prepare::read_onnx_graph, GraphInvalid (export périmé : le dire et le réexporter).
std::expected<void, Refusal> gmfss_pair_current(const std::filesystem::path& pair);

// gmfss_chosen, la paire vérifiée par gmfss_pair_current : une paire périmée cède la place à l'autre paire du
// dossier si celle-ci est au format 2, dans un sens comme dans l'autre (dropped dit pourquoi, vide sinon).
// Refus : celui de la paire choisie, quand aucune paire du dossier n'est au format 2.
std::expected<GmfssModels, Refusal> gmfss_usable(const GmfssModels& found, bool half, std::string& dropped);

// Les mêmes dossiers, avec repli vers les suivants si les deux paires du premier export sont périmées.
std::expected<GmfssModels, Refusal> find_usable_gmfss(std::span<const std::filesystem::path> dirs, bool half,
                                                      std::string& dropped);

struct GmfssAvailability {
  bool available = false;
  bool half_flow = false;
};

// Modèles à proposer au menu, dans le même ordre de recherche que le chargement de la paire.
GmfssAvailability gmfss_availability(std::span<const std::filesystem::path> dirs);

// Le menu revérifie l'empreinte des fichiers à son ouverture. Le chargement des moteurs contrôle toujours
// la paire qu'il a choisie en lisant l'ONNX, indépendamment de cet état du menu.
class GmfssMenuProbe {
 public:
  GmfssAvailability read(std::span<const std::filesystem::path> dirs);

 private:
  struct File {
    std::filesystem::path path;
    std::uintmax_t size;
    std::filesystem::file_time_type modified;
    friend bool operator==(const File&, const File&) = default;
  };
  std::vector<File> files_;
  GmfssAvailability available_;
};

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir ; les tailles
// d'une autre vidéo s'en vont. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace kirari::app::play
