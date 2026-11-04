// Les looks proposés au menu Couleurs : ceux du lecteur (render/looks.h) et les .cube que l'utilisateur
// dépose dans le dossier « luts » de ses données.
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
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
// Le dossier des LUT livrées avec le lecteur (luts/ à côté de ses modèles), lu après celui de l'utilisateur.
std::filesystem::path shipped_luts_dir();

std::vector<LookEntry> list_looks(const std::filesystem::path& luts_dir);

// La LUT d'une clé (look du lecteur ou fichier du dossier). Refus nommé si le fichier manque ou ne se lit
// pas.
std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir);

// Les noms des LUT de la pile dans l'ordre des emplacements (le premier, puis ceux de look_layers, au plus
// render::kLookSlots) et leur intensité.
// Sous --hdr-contract-v2 (8.4, set_contract_v2_looks), la pile par défaut, intensités comprises, devient le
// seul kContractV2Look ; une pile choisie par l'utilisateur reste la sienne. play.toml n'en sait rien.
// Depuis le 04/11, la pile est le look HDR anime seul (anime_look, les curseurs look_contrast et
// look_highlight) ; --no-looks : rien ; --hdr-contract-v2 : kContractV2Look. stack_slots : la pile de
// play.toml d'avant (look, look_layers), que le lecteur ne lit plus.
std::vector<LookLayer> look_slots(const PlaySettings& s);
std::vector<LookLayer> stack_slots(const PlaySettings& s);
// Le look HDR anime : la LUT de base, 3/7 de 7 × (1 + contraste/100) LUT de contraste, 1,5 × (1 + éclats/100)
// LUT d'éclats, « skin fix », puis les 4/7 restants du contraste ; une part fractionnaire est la dernière LUT
// à intensité partielle. À 0 et 0 : la pile de Tatoub du 04/11. Composé en une LUT de 65 points, gardé.
inline constexpr std::string_view kAnimeLookPrefix = "hdr-anime:";
inline constexpr std::string_view kAnimeLookBase = "HDR grading base ver.9.0 33 point.cube";
inline constexpr std::string_view kAnimeLookContrast = "HDR-contrast-Enhancement-L.cube";
inline constexpr std::string_view kAnimeLookHighlight = "Highlight-Enhancement-L.cube";
inline constexpr std::string_view kAnimeLookSkin = "skin fix.cube";
std::string anime_look_key(int contrast, int highlight);
std::vector<double> anime_look_mixes(double units);  // 2,5 → 1, 1, 0,5
std::expected<render::CubeLut, Refusal> anime_look(int contrast, int highlight,
                                                   const std::filesystem::path& luts_dir);
inline constexpr std::string_view kContractV2Look = "ver9-couleur.cube";
void set_contract_v2_looks(bool on);
// --no-looks (essai du tri du 04/11) : aucune LUT, quelle que soit la pile. play.toml n'en sait rien.
void set_no_looks(bool on);

// Vrai si les deux réglages ont la même pile de fichiers (les intensités ne comptent pas).
bool same_look_files(const PlaySettings& a, const PlaySettings& b);

}  // namespace ajp::app::play
