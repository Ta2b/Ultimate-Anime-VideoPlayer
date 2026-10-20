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
#include "app/play/settings.h"
#include "core/types.h"

namespace ajp::app::play {

struct ModelChoice {
  std::filesystem::path path;
  std::vector<std::string> available;  // fichiers du dossier proposés au menu, triés
  std::string default_name;            // celui qu'un nom vide désigne
};

// SPAN : --model, sinon settings.model, sinon le modèle par défaut ; le menu propose les 2x_*.onnx.
// bundled : les modèles livrés avec le lecteur (default_model_candidates), cherchés en dernier.
std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings,
                                                 const std::filesystem::path& bundled = {});
// Les seules versions de RIFE du lecteur : 4.25, la plus conseillée pour l'animation (7,1 ms par
// passe à 720p sur une 4070 Ti), et 4.22 lite, la plus légère. Les exports ONNX de vs-mlrt à 11
// plans d'entrée, convertis en FP16 à îlots FP32 (tools/rife/convert_rife_fp16.py), sous ces noms.
// --rife-model lit encore tout autre fichier, pour les essais.
inline constexpr std::array<std::string_view, 2> kRifeCatalogue{"rife_v4.25.onnx", "rife_v4.22_lite.onnx"};

// Le modèle par défaut, le premier présent dans cet ordre : 4.25 sur une RTX ; 4.22 lite, plus
// légère, sur les autres cartes (AMD, Intel, GTX).
inline constexpr std::array<std::string_view, 2> kRifeForRtx{"rife_v4.25.onnx", "rife_v4.22_lite.onnx"};
inline constexpr std::array<std::string_view, 2> kRifeForOthers{"rife_v4.22_lite.onnx", "rife_v4.25.onnx"};
std::span<const std::string_view> rife_preference(const GpuFacts& gpu);

// RIFE : --rife-model, sinon settings.rife_model s'il est au catalogue, sinon le premier de
// preference présent. Un choix absent du dossier est signalé au journal et le défaut le remplace.
std::expected<ModelChoice, Refusal> choose_rife_model(
    const PlayOptions& options, const std::filesystem::path& home, const PlaySettings& settings,
    std::span<const std::string_view> preference = kRifeForRtx, const std::filesystem::path& bundled = {});
struct RifeEntry {
  std::string file;
  bool present = false;  // dans le dossier du modèle par défaut
  friend bool operator==(const RifeEntry&, const RifeEntry&) = default;
};
// La liste du menu : le catalogue seul, présent ou non ; les autres fichiers du dossier (les
// modèles d'AnimeJaNai pour mpv) n'y entrent pas.
std::vector<RifeEntry> rife_menu(const std::vector<std::string>& available);

}  // namespace ajp::app::play
