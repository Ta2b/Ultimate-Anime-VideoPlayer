#include "app/play/data_dirs.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <fstream>
#include <iterator>
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

// Les premiers octets d'un fichier (au plus `count`), vide s'il ne se lit pas.
std::string head_of(const std::filesystem::path& file, std::size_t count) {
  std::ifstream in(file, std::ios::binary);
  std::string text(count, '\0');
  in.read(text.data(), static_cast<std::streamsize>(count));
  text.resize(static_cast<std::size_t>(std::max<std::streamsize>(in.gcount(), 0)));
  return text;
}

// « 1.2.0 », « v0.9.1-rc2 » : un dossier de version des mises à jour.
bool version_name(std::string_view name) {
  if (name.starts_with('v')) name.remove_prefix(1);
  return !name.empty() && std::isdigit(static_cast<unsigned char>(name.front())) != 0 &&
         std::ranges::all_of(name, [](char c) {
           return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '.' || c == '-';
         });
}

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

int purge_misplaced(const std::filesystem::path& shared,
                    const std::function<void(const std::string&)>& said) {
  namespace fs = std::filesystem;
  if (shared.empty()) return 0;
  int removed = 0;
  std::error_code error;
  const auto drop = [&](const fs::path& path) {
    std::error_code gone;
    if (fs::remove_all(path, gone) > 0 && !gone) {
      ++removed;
      if (said) said(std::format("{} retiré (d'une version d'avant)", path.string()));  // i18n-ok : journal
    }
  };
  const fs::path state = shared / "update.toml";
  if (fs::is_regular_file(state, error) && fs::file_size(state, error) < 65536 &&
      head_of(state, 65536).contains("last_check_unix"))
    drop(state);
  const fs::path marker = shared / "engines-relaunched";
  if (fs::is_regular_file(marker, error) && fs::file_size(marker, error) <= 64) drop(marker);
  const fs::path cache = shared / "shaders.cache";
  if (fs::is_regular_file(cache, error) && head_of(cache, 8) == "pl_cache") drop(cache);
  const fs::path plans = shared / "plans";
  if (fs::is_directory(plans, error)) {
    for (const fs::directory_entry& entry : fs::directory_iterator(plans, error))
      if (entry.is_regular_file(error) && entry.path().extension() == ".shots") drop(entry.path());
    if (fs::is_empty(plans, error) && !error) drop(plans);
  }
  const fs::path updates = shared / "updates";
  if (fs::is_directory(updates, error)) {
    bool ours = true;
    for (const fs::directory_entry& entry : fs::directory_iterator(updates, error)) {
      const std::string name = entry.path().filename().string();
      ours = ours && entry.is_directory(error) &&
             (name == "extract" || name == "previous" || version_name(name));
    }
    if (ours && !error) drop(updates);
  }
  return removed;
}

}  // namespace ajp::app::play
