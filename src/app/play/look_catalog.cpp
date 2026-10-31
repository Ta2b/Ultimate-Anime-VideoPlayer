#include "app/play/look_catalog.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <system_error>

#include <SDL3/SDL_filesystem.h>

#include "app/play/i18n.h"
#include "core/log.h"
#include "render/looks.h"

namespace ajp::app::play {

std::filesystem::path shipped_luts_dir() {
  // Les LUT livrées avec le lecteur : luts/ à côté de SDL_GetBasePath (Contents/Resources dans AJP.app),
  // comme models/.
  const char* base = SDL_GetBasePath();
  return base == nullptr ? std::filesystem::path() : std::filesystem::path(base) / "luts";
}

namespace {

// Les .cube d'un dossier, par nom.
void collect_cubes(const std::filesystem::path& dir, std::vector<std::string>& files) {
  std::error_code error;
  if (dir.empty() || !std::filesystem::is_directory(dir, error)) return;
  for (const auto& item : std::filesystem::directory_iterator(dir, error)) {
    std::string extension = item.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (item.is_regular_file(error) && extension == ".cube") files.push_back(item.path().filename().string());
  }
}

}  // namespace

std::vector<LookEntry> list_looks(const std::filesystem::path& luts_dir) {
  std::vector<LookEntry> entries;
  for (const render::BuiltinLook& look : render::builtin_looks())
    entries.push_back({std::string(look.key), std::string(i18n::tr(look.label.data()))});
  std::vector<std::string> files;
  collect_cubes(luts_dir,
                files);  // ceux de l'utilisateur d'abord, puis ceux livrés : un nom n'apparaît qu'une fois
  collect_cubes(shipped_luts_dir(), files);
  std::ranges::sort(files);
  files.erase(std::ranges::unique(files).begin(), files.end());
  for (const std::string& file : files) entries.push_back({file, file});
  return entries;
}

std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir) {
  if (key.starts_with(render::kBuiltinPrefix)) return render::builtin_look(key);
  if (key.empty() || key.find('/') != std::string::npos || key.find('\\') != std::string::npos)
    return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : nom de fichier invalide"});
  std::ifstream file(luts_dir / key, std::ios::binary);
  if (!file) file.open(shipped_luts_dir() / key, std::ios::binary);  // pas chez l'utilisateur : celui livré
  if (!file) return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : " + key + " introuvable"});
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto lut = render::parse_cube(text);
  if (!lut) {
    lut.error().detail = key + " : " + lut.error().detail;
    return std::unexpected(std::move(lut.error()));
  }
  // Les entrees noires d'un gamut d'entree sans usage ne doivent pas assombrir le bord de la zone utile.
  if (const std::size_t filled = render::fill_unreachable(*lut); filled > 0)
    log_channel("app")->info("look : {} : {} entrees inutilisees remplacees", key, filled);
  return lut;
}

std::vector<LookLayer> look_slots(const PlaySettings& s) {
  std::vector<LookLayer> slots{{s.look, s.look_mix}};
  for (const LookLayer& layer : s.look_layers) slots.push_back(layer);
  if (slots.size() > render::kLookSlots) slots.resize(render::kLookSlots);
  return slots;
}

bool same_look_files(const PlaySettings& a, const PlaySettings& b) {
  const std::vector<LookLayer> first = look_slots(a);
  const std::vector<LookLayer> second = look_slots(b);
  return std::ranges::equal(first, second,
                            [](const LookLayer& x, const LookLayer& y) { return x.key == y.key; });
}

}  // namespace ajp::app::play
