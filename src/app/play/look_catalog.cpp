#include "app/play/look_catalog.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <system_error>

#include <SDL3/SDL_filesystem.h>

#include "app/play/i18n.h"
#include "core/dir_entries.h"
#include "core/log.h"

namespace ajp::app::play {

std::filesystem::path shipped_luts_dir() {
  // Les LUT livrées avec le lecteur : luts/ à côté de SDL_GetBasePath (Contents/Resources dans AJP.app),
  // comme models/.
  const char* base = SDL_GetBasePath();
  return base == nullptr ? std::filesystem::path() : std::filesystem::path(base) / "luts";
}

std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir) {
  if (key.starts_with(kAnimeLookPrefix)) {
    // « contraste:hautes lumières », deux entiers lus par from_chars (P2.35).
    int contrast = 0;
    int highlight = 0;
    const char* last = key.data() + key.size();
    const auto first = std::from_chars(key.data() + kAnimeLookPrefix.size(), last, contrast);
    const bool read = first.ec == std::errc{} && first.ptr != last && *first.ptr == ':' &&
                      std::from_chars(first.ptr + 1, last, highlight).ec == std::errc{};
    if (!read) return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : " + key + " illisible"});
    return anime_look(contrast, highlight, luts_dir);
  }
  if (key.empty() || key.find('/') != std::string::npos || key.find('\\') != std::string::npos)
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

std::string anime_look_key(int contrast, int highlight) {
  return std::format("{}{}:{}", kAnimeLookPrefix, contrast, highlight);
}

std::vector<double> anime_look_mixes(double units) {
  std::vector<double> mixes;
  while (units > 1e-9) {
    mixes.push_back(std::min(units, 1.0));
    units -= 1.0;
  }
  return mixes;
}

std::expected<render::CubeLut, Refusal> anime_look(int contrast, int highlight,
                                                   const std::filesystem::path& luts_dir) {
  // Les quatre derniers looks faits (P2.9) : un 65³ pèse 3 Mo, chaque valeur du curseur en faisait un de
  // plus, gardé jusqu'à la fin du processus.
  constexpr std::size_t kKeptLooks = 4;
  // Jamais détruits : un fil de lecture détaché (LookLoader) peut encore s'en servir à la sortie du
  // processus.
  static std::mutex& lock = *new std::mutex;  // NOLINT(cppcoreguidelines-owning-memory)
  static auto& made =                         // le plus récent devant
      *new std::list<
          std::pair<std::pair<int, int>, render::CubeLut>>;  // NOLINT(cppcoreguidelines-owning-memory)
  const std::scoped_lock held(lock);
  const std::pair<int, int> key{contrast, highlight};
  if (const auto found =
          std::ranges::find(made, key, &std::pair<std::pair<int, int>, render::CubeLut>::first);
      found != made.end()) {
    made.splice(made.begin(), made, found);
    return made.front().second;
  }
  std::map<std::string_view, render::CubeLut> files;
  for (const std::string_view name :
       {kAnimeLookBase, kAnimeLookContrast, kAnimeLookHighlight, kAnimeLookSkin}) {
    auto lut = load_look(std::string(name), luts_dir);
    if (!lut) return std::unexpected(std::move(lut.error()));
    files.emplace(name, std::move(*lut));
  }
  // L'ordre de la pile par défaut : base, 3/7 du contraste, éclats, skin fix, 4/7 du contraste.
  const double c = 7.0 * (1.0 + std::clamp(contrast, -100, 100) / 100.0);
  const double h = 1.5 * (1.0 + std::clamp(highlight, -100, 100) / 100.0);
  std::vector<const render::CubeLut*> luts{&files.at(kAnimeLookBase)};
  std::vector<double> mixes{1.0};
  const auto add = [&](std::string_view name, double units) {
    for (const double mix : anime_look_mixes(units)) {
      luts.push_back(&files.at(name));
      mixes.push_back(mix);
    }
  };
  add(kAnimeLookContrast, c * 3.0 / 7.0);
  add(kAnimeLookHighlight, h);
  luts.push_back(&files.at(kAnimeLookSkin));
  mixes.push_back(1.0);
  add(kAnimeLookContrast, c * 4.0 / 7.0);
  render::CubeLut out = render::compose_luts(luts, mixes);
  out.title = "HDR anime";
  log_channel("app")->info("look HDR anime : contrast {}, highlight {} ({} LUT)", contrast, highlight,
                           luts.size());
  made.emplace_front(key, out);
  if (made.size() > kKeptLooks) made.pop_back();
  return out;
}

std::vector<LookLayer> look_slots(const PlaySettings& s) {
  return {LookLayer{anime_look_key(s.look_contrast, s.look_highlight), 1.0}};
}

bool same_look_files(const PlaySettings& a, const PlaySettings& b) {
  const std::vector<LookLayer> first = look_slots(a);
  const std::vector<LookLayer> second = look_slots(b);
  return std::ranges::equal(first, second,
                            [](const LookLayer& x, const LookLayer& y) { return x.key == y.key; });
}

}  // namespace ajp::app::play
