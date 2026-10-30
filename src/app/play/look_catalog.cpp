#include "app/play/look_catalog.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <system_error>

#include "app/play/i18n.h"
#include "render/looks.h"

namespace ajp::app::play {

std::vector<LookEntry> list_looks(const std::filesystem::path& luts_dir) {
  std::vector<LookEntry> entries;
  for (const render::BuiltinLook& look : render::builtin_looks())
    entries.push_back({std::string(look.key), std::string(i18n::tr(look.label.data()))});
  std::vector<std::string> files;
  std::error_code error;
  if (!luts_dir.empty() && std::filesystem::is_directory(luts_dir, error)) {
    for (const auto& item : std::filesystem::directory_iterator(luts_dir, error)) {
      std::string extension = item.path().extension().string();
      std::ranges::transform(extension, extension.begin(),
                             [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (item.is_regular_file(error) && extension == ".cube")
        files.push_back(item.path().filename().string());
    }
  }
  std::ranges::sort(files);
  for (const std::string& file : files) entries.push_back({file, file});
  return entries;
}

std::expected<render::CubeLut, Refusal> load_look(const std::string& key,
                                                  const std::filesystem::path& luts_dir) {
  if (key.starts_with(render::kBuiltinPrefix)) return render::builtin_look(key);
  if (key.empty() || key.find('/') != std::string::npos || key.find('\\') != std::string::npos)
    return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : nom de fichier invalide"});
  std::ifstream file(luts_dir / key, std::ios::binary);
  if (!file) return std::unexpected(Refusal{RefusalCode::FileUnreadable, "look : " + key + " introuvable"});
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto lut = render::parse_cube(text);
  if (!lut) {
    lut.error().detail = key + " : " + lut.error().detail;
    return std::unexpected(std::move(lut.error()));
  }
  return lut;
}

}  // namespace ajp::app::play
