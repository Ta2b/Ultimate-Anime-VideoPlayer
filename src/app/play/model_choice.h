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
std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings);
// Les versions de RIFE que le menu propose toujours, de la plus conseillée pour l'animation à la
// plus légère (avis du développeur de RIFE et de la communauté SVP, et nos mesures : 4.25 coûte
// 7,1 ms par passe à 720p sur une 4070 Ti, 4.26 7,8 ms). Les exports ONNX de vs-mlrt à 11 plans
// d'entrée, convertis en FP16 à îlots FP32 (tools/rife/convert_rife_fp16.py), sous ces noms. Les
// v2 (7 plans) sont le même réseau aux mêmes poids : même image, 3 % plus rapides en 1080p
// seulement ; ils restent lisibles mais ne sont pas proposés. 4.26 garde les déformations les plus
// faibles, et l'ancien lecteur mpv l'utilise.
inline constexpr std::array<std::string_view, 5> kRifeCatalogue{
    "rife_v4.25.onnx", "rife_v4.25_lite.onnx", "rife_v4.22.onnx", "rife_v4.22_lite.onnx", "rife_v4.26.onnx"};

// Le modèle par défaut, le premier présent dans cet ordre : 4.25 sur une RTX ; 4.22 lite, plus
// légère, sur les autres cartes (AMD, Intel, GTX).
inline constexpr std::array<std::string_view, 5> kRifeForRtx{
    "rife_v4.25.onnx", "rife_v4.26.onnx", "rife_v4.25_lite.onnx", "rife_v4.22.onnx", "rife_v4.22_lite.onnx"};
inline constexpr std::array<std::string_view, 5> kRifeForOthers{
    "rife_v4.22_lite.onnx", "rife_v4.25_lite.onnx", "rife_v4.22.onnx", "rife_v4.25.onnx", "rife_v4.26.onnx"};
std::span<const std::string_view> rife_preference(const GpuFacts& gpu);

// RIFE : --rife-model, sinon settings.rife_model, sinon le premier de preference présent ; le
// menu propose les rife_v*.onnx du dossier. Un choix absent du dossier est signalé au journal et
// le défaut le remplace.
std::expected<ModelChoice, Refusal> choose_rife_model(
    const PlayOptions& options, const std::filesystem::path& home, const PlaySettings& settings,
    std::span<const std::string_view> preference = kRifeForRtx);
struct RifeEntry {
  std::string file;
  bool present = false;  // dans le dossier du modèle par défaut
  friend bool operator==(const RifeEntry&, const RifeEntry&) = default;
};
// La liste du menu : le catalogue, présent ou non, puis les autres fichiers du dossier dans
// l'ordre d'available.
std::vector<RifeEntry> rife_menu(const std::vector<std::string>& available);

}  // namespace ajp::app::play
