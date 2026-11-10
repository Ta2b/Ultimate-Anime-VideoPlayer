#include "app/play/look_catalog.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>

#include <SDL3/SDL_filesystem.h>

#include "app/play/i18n.h"
#include "core/dir_entries.h"
#include "core/log.h"
#include "core/text.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

std::filesystem::path shipped_luts_dir() {
  // Les LUT livrées avec le lecteur : luts/ à côté de SDL_GetBasePath (Contents/Resources dans AJP.app),
  // comme models/.
  const char* base = SDL_GetBasePath();
  return base == nullptr ? std::filesystem::path() : std::filesystem::path(base) / "luts";
}

std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir) {
  // Un nom de fichier seul, en .cube : ni séparateur ni « C:x.cube » de Windows, qui viserait le dossier
  // courant d'un lecteur (A35-09).
  const std::filesystem::path name = core::path_from_utf8(key);
  const std::string extension = core::ascii_lower(core::to_utf8(name.extension()));
  if (key.empty() || key.find('/') != std::string::npos || key.find('\\') != std::string::npos ||
      key.find(':') != std::string::npos || name.filename() != name || extension != ".cube")
    return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : nom de fichier invalide"});
  std::filesystem::path path = luts_dir / key;
  std::ifstream file(path, std::ios::binary);
  if (!file) {  // pas chez l'utilisateur : celui livré
    path = shipped_luts_dir() / key;
    file.open(path, std::ios::binary);
  }
  if (!file) return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : " + key + " introuvable"});
  // La taille d'abord (P2.30) : un fichier énorme n'est pas lu en entier pour être refusé ensuite.
  std::error_code sized;
  if (const auto bytes = std::filesystem::file_size(path, sized); sized || bytes > render::kLutMaxBytes)
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "look : " + key + " trop grand"});
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto lut = render::parse_cube(text);
  if (!lut) {
    lut.error().detail = key + " : " + lut.error().detail;
    return std::unexpected(std::move(lut.error()));
  }
  // Les entrees noires d'un gamut d'entree sans usage ne doivent pas assombrir le bord de la zone utile :
  // cela vaut pour les LUT livrees, que l'on connait ; celle de l'utilisateur reste telle qu'il l'a faite
  // (P3.69).
  const bool shipped = path.parent_path() == shipped_luts_dir();
  if (const std::size_t filled = shipped ? render::fill_unreachable(*lut) : 0; filled > 0)
    log_channel("app")->warn("look : {} : {} entrees inutilisees remplacees", key, filled);
  return lut;
}

std::vector<double> anime_look_mixes(double units) {
  std::vector<double> mixes;
  while (units > 1e-9) {
    mixes.push_back(std::min(units, 1.0));
    units -= 1.0;
  }
  return mixes;
}

std::vector<LookLayer> anime_look_layers(int contrast, int highlight) {
  // L'ordre de la pile par défaut : base, 3/7 du contraste, éclats, skin fix, 4/7 du contraste.
  const double c = 7.0 * (1.0 + std::clamp(contrast, -100, 100) / 100.0);
  const double h = 1.5 * (1.0 + std::clamp(highlight, -100, 100) / 100.0);
  std::vector<LookLayer> layers{LookLayer{std::string(kAnimeLookBase), 1.0}};
  const auto add = [&](std::string_view name, double units) {
    for (const double mix : anime_look_mixes(units)) layers.push_back(LookLayer{std::string(name), mix});
  };
  add(kAnimeLookContrast, c * 3.0 / 7.0);
  add(kAnimeLookHighlight, h);
  layers.push_back(LookLayer{std::string(kAnimeLookSkin), 1.0});
  add(kAnimeLookContrast, c * 4.0 / 7.0);
  return layers;
}

std::vector<LookLayer> look_slots(const PlaySettings& s) {
  return anime_look_layers(s.look_contrast, s.look_highlight);
}

bool same_look_files(const PlaySettings& a, const PlaySettings& b) {
  const std::vector<LookLayer> first = look_slots(a);
  const std::vector<LookLayer> second = look_slots(b);
  return std::ranges::equal(first, second,
                            [](const LookLayer& x, const LookLayer& y) { return x.key == y.key; });
}

}  // namespace ajp::app::play
