#include "app/play/look_catalog.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
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
    entries.push_back({std::string(look.key), std::string(i18n::tr_view(look.label))});
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
  if (key.starts_with(kAnimeLookPrefix)) {
    int contrast = 0;
    int highlight = 0;
    if (std::sscanf(key.c_str() + kAnimeLookPrefix.size(), "%d:%d", &contrast, &highlight) != 2)
      return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : " + key + " illisible"});
    return anime_look(contrast, highlight, luts_dir);
  }
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

namespace {

std::atomic<bool> contract_v2_looks{false};
std::atomic<bool> no_looks{false};

std::vector<LookLayer> stack_of(const PlaySettings& s) {
  std::vector<LookLayer> slots{{s.look, s.look_mix}};
  for (const LookLayer& layer : s.look_layers) slots.push_back(layer);
  if (slots.size() > render::kLookSlots) slots.resize(render::kLookSlots);
  return slots;
}

}  // namespace

void set_contract_v2_looks(bool on) {
  contract_v2_looks.store(on, std::memory_order_relaxed);
}

void set_no_looks(bool on) {
  no_looks.store(on, std::memory_order_relaxed);
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
  static std::mutex lock;
  static std::map<std::pair<int, int>, render::CubeLut> made;
  const std::scoped_lock held(lock);
  if (const auto found = made.find({contrast, highlight}); found != made.end()) return found->second;
  std::map<std::string_view, render::CubeLut> files;
  for (const std::string_view name :
       {kAnimeLookBase, kAnimeLookContrast, kAnimeLookHighlight, kAnimeLookSkin}) {
    auto lut = load_look(std::string(name), luts_dir);
    if (!lut) return std::unexpected(std::move(lut.error()));
    files.emplace(name, std::move(*lut));
  }
  // L'ordre de la pile de Tatoub : base, 3/7 du contraste, éclats, skin fix, 4/7 du contraste.
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
  made.emplace(std::pair{contrast, highlight}, out);
  return out;
}

std::vector<LookLayer> look_slots(const PlaySettings& s) {
  if (no_looks.load(std::memory_order_relaxed)) return {};
  if (contract_v2_looks.load(std::memory_order_relaxed))
    return {LookLayer{std::string(kContractV2Look), 1.0}};
  return {LookLayer{anime_look_key(s.look_contrast, s.look_highlight), 1.0}};
}

std::vector<LookLayer> stack_slots(const PlaySettings& s) {
  std::vector<LookLayer> slots = stack_of(s);
  if (contract_v2_looks.load(std::memory_order_relaxed)) {
    static const std::vector<LookLayer> kDefault = stack_of(PlaySettings{});
    if (slots == kDefault) return {LookLayer{std::string(kContractV2Look), 1.0}};
  }
  return slots;
}

bool same_look_files(const PlaySettings& a, const PlaySettings& b) {
  const std::vector<LookLayer> first = look_slots(a);
  const std::vector<LookLayer> second = look_slots(b);
  return std::ranges::equal(first, second,
                            [](const LookLayer& x, const LookLayer& y) { return x.key == y.key; });
}

}  // namespace ajp::app::play
