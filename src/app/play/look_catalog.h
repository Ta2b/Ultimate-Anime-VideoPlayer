// Les looks proposés au menu Couleurs : ceux du lecteur (render/looks.h) et les .cube que l'utilisateur
// dépose dans le dossier « luts » de ses données.
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "app/play/settings.h"
#include "core/types.h"
#include "render/cube_lut.h"

namespace ajp::app::play {

struct LookEntry {
  std::string key;    // ce que play.toml garde : « maison:vif » ou le nom du fichier
  std::string label;  // ce que le menu montre
};

// Les looks du lecteur d'abord, puis les .cube du dossier (triés par nom) ; un dossier absent n'est pas une
// erreur.
std::vector<LookEntry> list_looks(const std::filesystem::path& luts_dir);

// La LUT d'une clé (look du lecteur ou fichier du dossier). Refus nommé si le fichier manque ou ne se lit
// pas.
std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir);

// Les noms des LUT de la pile dans l'ordre des emplacements (le premier, puis ceux de look_layers, au plus
// render::kLookSlots) et leur intensité.
std::vector<LookLayer> look_slots(const PlaySettings& s);

// Vrai si les deux réglages ont la même pile de fichiers (les intensités ne comptent pas).
bool same_look_files(const PlaySettings& a, const PlaySettings& b);

}  // namespace ajp::app::play
