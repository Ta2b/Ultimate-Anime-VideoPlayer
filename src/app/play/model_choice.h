// Les fichiers ONNX d'une lecture : le modèle SPAN et le modèle RIFE. L'option de la ligne de
// commande l'emporte, puis le choix du menu (un nom de fichier à côté du modèle par défaut),
// puis le modèle par défaut ; le dossier du modèle par défaut donne la liste du menu.
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

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
// RIFE : --rife-model, sinon settings.rife_model, sinon rife_v4.26.onnx ; le menu propose les
// rife_v*.onnx (4.26 heavy, lite, versions plus anciennes). Un choix absent du dossier est
// signalé au journal et le défaut le remplace.
std::expected<ModelChoice, Refusal> choose_rife_model(const PlayOptions& options,
                                                      const std::filesystem::path& home,
                                                      const PlaySettings& settings);

}  // namespace ajp::app::play
