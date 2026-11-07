#include "app/play/data_dirs.h"

#include <array>
#include <format>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <shlobj.h>
#include <windows.h>
#endif

namespace ajp::app::play {

namespace {

// Ce qui part en local : les moteurs construits (engines), les paquets de moteurs (backends), les modèles
// téléchargés (models) et dérivés (modeles), les téléchargements en cours, les LUT, le cache des plans et
// celui des shaders. Le reste (play.toml, settings.toml, calibration.toml, journal) reste itinérant.
constexpr std::array<std::string_view, 8> kHeavy{"engines",   "backends", "models", "modeles",
                                                 "downloads", "luts",     "plans",  "shaders.cache"};

}  // namespace

std::filesystem::path local_data(const std::filesystem::path& roaming) {
#ifdef _WIN32
  PWSTR found = nullptr;
  if (SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &found) == S_OK && found) {
    std::filesystem::path local(found);
    CoTaskMemFree(found);
    // Les deux derniers noms du dossier de SDL (« ajp\ajp\ », barre finale comprise).
    std::filesystem::path pref = roaming;
    if (!pref.has_filename()) pref = pref.parent_path();
    local = local / pref.parent_path().filename() / pref.filename();
    std::error_code made;
    std::filesystem::create_directories(local, made);
    if (!made) return local;
  } else if (found) {
    CoTaskMemFree(found);
  }
#endif
  return roaming;
}

int move_heavy_data(const std::filesystem::path& from, const std::filesystem::path& to,
                    const std::function<void(const std::string&)>& said) {
  if (from.empty() || to.empty()) return 0;
  std::error_code same;
  if (std::filesystem::equivalent(from, to, same)) return 0;
  int moved = 0;
  for (const std::string_view name : kHeavy) {
    const std::filesystem::path source = from / name;
    const std::filesystem::path target = to / name;
    std::error_code error;
    if (!std::filesystem::exists(source, error) || std::filesystem::exists(target, error)) continue;
    std::filesystem::create_directories(to, error);
    std::filesystem::rename(source, target, error);
    if (error) {
      // Un autre volume : copie puis effacement ; une copie incomplète est retirée, l'ancien dossier sert
      // encore.
      error.clear();
      std::filesystem::copy(source, target, std::filesystem::copy_options::recursive, error);
      if (error) {
        std::error_code ignored;
        std::filesystem::remove_all(target, ignored);
        constexpr std::string_view kRefused = "{} : non déplacé vers {} ({})";  // i18n-ok : journal
        const std::string line = std::format(kRefused, source.string(), to.string(), error.message());
        if (said) said(line);
        continue;
      }
      std::filesystem::remove_all(source, error);  // un reste ne sert plus : la copie complète est lue
    }
    ++moved;
    if (said) said(std::format("{} déplacé vers {}", source.string(), target.string()));  // i18n-ok : journal
  }
  return moved;
}

}  // namespace ajp::app::play
