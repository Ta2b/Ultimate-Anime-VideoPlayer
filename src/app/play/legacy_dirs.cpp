#include "app/play/legacy_dirs.h"

#include <format>
#include <system_error>

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_stdinc.h>

#include "core/utf8_path.h"

namespace kirari::app::play {

namespace fs = std::filesystem;

fs::path legacy_pref_dir(fs::path pref) {
  if (pref.empty()) return {};
  if (!pref.has_filename()) pref = pref.parent_path();  // la barre finale de SDL
  return pref.parent_path() / kLegacyName / kLegacyName;
}

bool adopt_legacy_dir(const fs::path& legacy, const fs::path& given,
                      const std::function<void(const std::string&)>& said) {
  fs::path target = given;
  if (!target.empty() && !target.has_filename()) target = target.parent_path();  // la barre finale de SDL
  const auto say = [&said](const std::string& line) {
    if (said) said(line);
  };
  std::error_code error;
  if (legacy.empty() || target.empty() || !fs::is_directory(legacy, error)) return false;
  if (fs::exists(target, error)) {
    if (!fs::is_directory(target, error) || !fs::is_empty(target, error)) return false;
    fs::remove(target, error);
    if (error) {
      say(std::format("ancien dossier {} gardé : {} ne se retire pas ({})",  // i18n-ok : journal
                      core::to_utf8(legacy), core::to_utf8(target), error.message()));
      return false;
    }
  }
  fs::create_directories(target.parent_path(), error);
  fs::rename(legacy, target, error);
  if (error) {
    // Le dossier vide que SDL attend revient : le lecteur démarre sur un dossier neuf.
    std::error_code made;
    fs::create_directories(target, made);
    say(std::format("ancien dossier {} gardé : {}",  // i18n-ok : journal
                    core::to_utf8(legacy), error.message()));
    return false;
  }
  say(std::format("ancien dossier {} repris dans {}",  // i18n-ok : journal
                  core::to_utf8(legacy), core::to_utf8(target)));
  // <base>/ajp, vide une fois ajp/ajp parti ; s'il garde autre chose, il reste.
  if (const fs::path parent = legacy.parent_path();
      parent.filename() == kLegacyName && fs::is_empty(parent, error))
    fs::remove(parent, error);
  return true;
}

namespace {

std::vector<std::string>& adopted_lines() {
  static std::vector<std::string> lines;
  return lines;
}

}  // namespace

fs::path prefs_dir() {
  char* pref = SDL_GetPrefPath("", std::string(kDataName).c_str());
  const fs::path data = pref != nullptr ? core::path_from_utf8(pref) : fs::path();
  SDL_free(pref);
  adopt_legacy_dir(legacy_pref_dir(data), data,
                   [](const std::string& line) { adopted_lines().push_back(line); });
  return data;
}

const std::vector<std::string>& adopted_dir_lines() {
  return adopted_lines();
}

}  // namespace kirari::app::play
