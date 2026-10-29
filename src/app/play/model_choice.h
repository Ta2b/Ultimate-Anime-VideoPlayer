// Les fichiers ONNX d'une lecture : le modèle SPAN et le modèle RIFE. L'option de la ligne de
// commande l'emporte, puis le choix du menu (un nom de fichier à côté du modèle par défaut),
// puis le modèle par défaut ; le dossier du modèle par défaut donne la liste du menu.
#pragma once

#include <array>
#include <expected>
#include <filesystem>
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

// Le SPAN des vieux animes en SD : AniSD AC (Sirosky, Upscale-Hub, CC BY-NC 4.0), un SPAN à 48 canaux et 6
// blocs entraîné sur des DVD abîmés (compression, dotcrawl, bruit), converti en FP16 par
// tools/span/convert_anisd_fp16.py (docs/modele-sd.md). Ce n'est pas un modèle d'AnimeJaNai : AnimeJaNai ne
// publie en SD que des Compact, que le lecteur ne lance pas.
inline constexpr std::string_view kSpanSdModel = "2x_AniSD_AC_G6i2b_SPAN_190K_fp16.onnx";
// En dessous de cette hauteur (lignes) la source est du SD, DVD compris (576 lignes en PAL) : la config
// automatique y conseille kSpanSdModel.
inline constexpr int kSdMaxHeight = 576;

// Les modèles SPAN que le menu propose (v0.6 : plus de modèle « legacy » du dossier d'AnimeJaNai, comme le
// Compact SD 1.x de l'ancien lecteur) : AnimeJaNai HD V3.1, Performance ou Balanced, avec ou sans Sharp1, et
// AniSD AC pour le SD. --model lit encore tout autre fichier, pour les essais ; un réglage de play.toml qui
// nomme un autre modèle retombe sur le défaut.
inline constexpr std::array<std::string_view, 5> kSpanCatalogue{
    "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx",
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx", kSpanSdModel};

// Le modèle que la config automatique conseille pour une source de cette hauteur : kSpanSdModel en SD, « »
// (le modèle par défaut, HD) sinon ou si la hauteur est inconnue. Absent du dossier, le choix retombe sur le
// modèle HD (choose_model).
std::string_view recommended_span_model(int source_height);
// Les noyaux FP8 CUTLASS (cran Performance, « Préparer le mode Performance ») ne prennent que le SPAN b8f64 à
// 64 canaux d'AnimeJaNai ; AniSD (48 canaux, 6 blocs, entrée à 3 canaux sans dégroupage) n'a pas ce cran.
bool span_fp8_supported(std::string_view model_file);

// SPAN : --model, sinon settings.model s'il est au catalogue, sinon le modèle par défaut ; le menu propose le
// catalogue présent dans le dossier. bundled : les modèles livrés avec le lecteur (default_model_candidates),
// cherchés en dernier.
std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings,
                                                 const std::filesystem::path& bundled = {});
// Les deux versions de RIFE que le menu propose, de la plus fine à la plus légère : 4.25 (7,1 ms par passe à
// 720p sur une 4070 Ti) et 4.22 lite, plus légère (retours de Tatoub, v0.5.1 : « catalogue réduit à 4.22 lite
// et 4.25 »). Les exports ONNX de vs-mlrt à 11 plans d'entrée, convertis en FP16 à îlots FP32
// (tools/rife/convert_rife_fp16.py), sous ces noms. --rife-model lit encore tout autre fichier, pour les
// essais.
inline constexpr std::array<std::string_view, 2> kRifeCatalogue{"rife_v4.25.onnx", "rife_v4.22_lite.onnx"};

// Le modèle par défaut, le premier présent dans cet ordre : 4.25 sur une RTX ; 4.22 lite, plus légère, sur
// les autres cartes (AMD, Intel, GTX).
inline constexpr std::array<std::string_view, 2> kRifeForRtx{"rife_v4.25.onnx", "rife_v4.22_lite.onnx"};
inline constexpr std::array<std::string_view, 2> kRifeForOthers{"rife_v4.22_lite.onnx", "rife_v4.25.onnx"};
std::span<const std::string_view> rife_preference(const GpuFacts& gpu);

// RIFE : --rife-model, sinon settings.rife_model s'il est au catalogue, sinon le premier de
// preference présent. Un choix absent du dossier est signalé au journal et le défaut le remplace.
std::expected<ModelChoice, Refusal> choose_rife_model(
    const PlayOptions& options, const std::filesystem::path& home, const PlaySettings& settings,
    std::span<const std::string_view> preference = kRifeForRtx, const std::filesystem::path& bundled = {});
// Le graphe du cran tier à côté de model : « rife_v4.25_fp8.onnx » (Performance), « rife_v4.25_fp32.onnx »
// (Qualité). model lui-même pour Équilibre ou si le fichier du cran manque.
std::filesystem::path tier_variant(const std::filesystem::path& model, PrecisionTier tier);
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
