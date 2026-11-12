// Les fichiers ONNX d'une lecture : le modèle SPAN et le modèle RIFE. L'option de la ligne de
// commande l'emporte, puis le choix du menu (un nom de fichier à côté du modèle par défaut),
// puis le modèle par défaut ; le dossier du modèle par défaut donne la liste du menu.
#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/first_run.h"
#include "app/play/options.h"
#include "app/play/precision.h"
#include "app/play/settings.h"
#include "core/types.h"

namespace ajp::app::play {

struct ModelChoice {
  std::filesystem::path path;
  std::vector<std::string> available;  // fichiers du dossier proposés au menu, triés
  std::string default_name;            // celui qu'un nom vide désigne
};

// Les modèles SPAN que le menu propose (v0.6 : plus de modèle « legacy » du dossier d'AnimeJaNai, comme le
// Compact SD 1.x de l'ancien lecteur) : AnimeJaNai HD V3.1, Performance ou Balanced, avec ou sans Sharp1,
// et ArtCNN (P6.7), bien plus léger, pour les petites cartes : C4F32 (32 canaux) et C4F16 (16 canaux, environ
// quatre fois moins de calcul, pour les iGPU les plus faibles). --model lit encore tout autre fichier, pour
// les essais ; un réglage de play.toml qui nomme un autre modèle retombe sur le défaut.
inline constexpr std::string_view kLightSpanModel = "2x_ArtCNN_C4F32_rgb_fp16.onnx";
inline constexpr std::string_view kLightestSpanModel = "2x_ArtCNN_C4F16_rgb_fp16.onnx";
inline constexpr std::array<std::string_view, 6> kSpanCatalogue{
    "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
    kLightSpanModel,
    kLightestSpanModel};

// SPAN : --model, sinon settings.model s'il est au catalogue, sinon le modèle par défaut ; le menu propose le
// catalogue présent dans le dossier. bundled : les modèles livrés avec le lecteur (default_model_candidates),
// cherchés en dernier.
std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings,
                                                 const std::filesystem::path& bundled = {});
// Les versions de RIFE que le menu propose, de la plus fine à la plus légère : 4.26 (Équilibré), 4.25
// (réglages avancés, l'ancien Équilibré : 7,1 ms par passe à 720p sur une 4070 Ti), 4.22 lite (Rapide)
// et 4.25 lite (Très rapide, deux fois moins de calcul que 4.22 lite et 2,9 dB de moins au banc,
// docs/rife-candidats.md). Les exports ONNX de vs-mlrt à 11 plans d'entrée, en FP16 : 4.26, 4.25 et 4.22 lite
// tels que convertis par animejanai-inference (tools/release/fetch-models.sh), 4.25 lite par
// tools/rife/convert_rife_fp16.py. Chez animejanai, la grille est en FP32 et une partie des Resize lit du
// FP16. Perte mesurée (docs/banc-qualite.md, B09-06) : 0,9 dB contre la vraie image en 720p, 1,2 dB en 1024
// lignes, face au même graphe tout en FP32 ; ces Resize passés en FP32 n'en rendent rien (47,56 dB des deux
// côtés), la perte est ailleurs. --rife-model lit encore tout autre fichier, pour les essais.
inline constexpr std::array<std::string_view, 4> kRifeCatalogue{
    "rife_v4.26.onnx", "rife_v4.25.onnx", "rife_v4.22_lite.onnx", "rife_v4.25_lite.onnx"};

// Le modèle par défaut, le premier présent dans cet ordre : 4.26 sur une RTX, 4.25 s'il manque ; 4.22 lite,
// plus légère, sur les autres cartes (AMD, Intel, GTX), 4.25 lite si elle manque.
inline constexpr std::array<std::string_view, 4> kRifeForRtx{"rife_v4.26.onnx", "rife_v4.25.onnx",
                                                             "rife_v4.22_lite.onnx", "rife_v4.25_lite.onnx"};
inline constexpr std::array<std::string_view, 4> kRifeForOthers{
    "rife_v4.22_lite.onnx", "rife_v4.25_lite.onnx", "rife_v4.26.onnx", "rife_v4.25.onnx"};
std::span<const std::string_view> rife_preference(const GpuFacts& gpu);

// RIFE : --rife-model, sinon settings.rife_model s'il est au catalogue, sinon le premier de
// preference présent. Un choix absent du dossier est signalé au journal et le défaut le remplace.
std::expected<ModelChoice, Refusal> choose_rife_model(
    const PlayOptions& options, const std::filesystem::path& home, const PlaySettings& settings,
    std::span<const std::string_view> preference = kRifeForRtx, const std::filesystem::path& bundled = {});
// Le graphe du cran tier à côté de model : « rife_v4.25_fp8.onnx » (Performance), « rife_v4.25_fp32.onnx »
// (Qualité). model lui-même pour Équilibre ou si le fichier du cran manque.
std::filesystem::path tier_variant(const std::filesystem::path& model, PrecisionTier tier);
// L'inverse : le cran du graphe chargé, lu à son nom (« _fp8 », « _fp32 ») ; Équilibre pour le modèle livré
// ou absent. C'est le cran rendu, celui qui dit si la chaîne fusionnée s'applique (A31-06).
PrecisionTier model_tier(const std::optional<std::filesystem::path>& model);
// Les crans dont le graphe est à côté de model ; fp8_engine : le moteur calcule le FP8.
TierSupport sibling_tiers(const std::filesystem::path& model, bool fp8_engine);
struct RifeEntry {
  std::string file;
  bool present = false;  // dans le dossier du modèle par défaut
  friend bool operator==(const RifeEntry&, const RifeEntry&) = default;
};
// La liste du menu : le catalogue seul, présent ou non ; les autres fichiers du dossier (les
// modèles d'AnimeJaNai pour mpv) n'y entrent pas.
std::vector<RifeEntry> rife_menu(const std::vector<std::string>& available);

}  // namespace ajp::app::play
