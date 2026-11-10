// Les LUT du grade HDR anime : lues dans le dossier « luts » de l'utilisateur, sinon dans celui livré avec le
// lecteur, puis composées selon les curseurs de contraste et d'éclats (docs/luts.md).
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

// Le dossier des LUT livrées avec le lecteur (luts/ à côté de ses modèles), lu après celui de l'utilisateur.
std::filesystem::path shipped_luts_dir();

// La LUT d'un fichier du dossier de l'utilisateur, sinon de celui livré. Refus nommé si le fichier manque ou
// ne se lit pas.
std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir);

// Les fichiers de la pile dans l'ordre des emplacements et leur intensité : le look HDR anime seul (les
// curseurs look_contrast et look_highlight). La pile de play.toml d'avant la 0.9 (look, look_layers) n'est
// plus lue.
std::vector<LookLayer> look_slots(const PlaySettings& s);
// Le look HDR anime : la LUT de base, 3/7 de 7 × (1 + contraste/100) LUT de contraste, 1,5 × (1 + éclats/100)
// LUT d'éclats, « skin fix », puis les 4/7 restants du contraste ; une part fractionnaire est la dernière LUT
// à intensité partielle. À 0 et 0 : la pile par défaut, 11 LUT ; aux curseurs à 100 : 19
// (render::kLookSlots en tient 20). Le shader du look les parcourt une à une (A41-01).
inline constexpr std::string_view kAnimeLookBase = "HDR grading base ver.9.0 33 point.cube";
inline constexpr std::string_view kAnimeLookContrast = "HDR-contrast-Enhancement-L.cube";
inline constexpr std::string_view kAnimeLookHighlight = "Highlight-Enhancement-L.cube";
inline constexpr std::string_view kAnimeLookSkin = "skin fix.cube";
std::vector<double> anime_look_mixes(double units);  // 2,5 → 1, 1, 0,5
std::vector<LookLayer> anime_look_layers(int contrast, int highlight);

// Vrai si les deux réglages ont la même pile de fichiers (les intensités ne comptent pas).
bool same_look_files(const PlaySettings& a, const PlaySettings& b);

}  // namespace ajp::app::play
