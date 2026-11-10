#include "app/play/data_dirs.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>

#include "core/utf8_path.h"

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

// Le dossier updates/ des versions qui s'appliquaient elles-mêmes : rien que des versions, extract et
// previous. Un parcours qui échoue ne reconnaît rien : rien ne part.
bool update_leftovers(const std::filesystem::path& updates) {
  namespace fs = std::filesystem;
  std::error_code error;
  if (!fs::is_directory(updates, error)) return false;
  fs::directory_iterator it(updates, error);
  for (; !error && it != fs::directory_iterator(); it.increment(error)) {
    const std::string name = core::to_utf8(it->path().filename());
    if (!it->is_directory(error) || !(name == "extract" || name == "previous" || version_name(name)))
      return false;
  }
  return !error;
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
  namespace fs = std::filesystem;
  if (from.empty() || to.empty()) return 0;
  std::error_code same;
  if (fs::equivalent(from, to, same)) return 0;
  const auto say = [&](const std::string& line) {
    if (said) said(line);
  };
  int moved = 0;
  for (const std::string_view name : kHeavy) {
    const fs::path source = from / name;
    const fs::path target = to / name;
    fs::path staging = target;
    staging += ".migration";
    std::error_code error;
    // Une copie coupée (processus tué, coupure de courant) ne laisse que « .migration » : retirée, la copie
    // recommence (A19-01).
    if (fs::exists(staging, error)) fs::remove_all(staging, error);
    // La cible n'existe qu'entière (renommée une fois copiée) : elle n'est jamais écrasée.
    if (!fs::exists(source, error) || fs::exists(target, error)) continue;
    fs::create_directories(to, error);
    fs::rename(source, target, error);
    if (error) {
      // Un autre volume : copie complète à côté, puis un renommage sur le volume de la cible ; la source ne
      // part qu'après. Une copie refusée est retirée, l'ancien dossier sert encore.
      error.clear();
      fs::copy(source, staging, fs::copy_options::recursive, error);
      if (!error) fs::rename(staging, target, error);
      if (error) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        constexpr std::string_view kRefused = "{} : non déplacé vers {} ({})";  // i18n-ok : journal
        say(std::format(kRefused, core::to_utf8(source), core::to_utf8(to), error.message()));
        continue;
      }
      fs::remove_all(source, error);  // un reste ne sert plus : la copie complète est lue
    }
    ++moved;
    constexpr std::string_view kMoved = "{} déplacé vers {}";  // i18n-ok : journal
    say(std::format(kMoved, core::to_utf8(source), core::to_utf8(target)));
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
      if (said)
        said(std::format("{} retiré (d'une version d'avant)", core::to_utf8(path)));  // i18n-ok : journal
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
    // Parcours sans exception (A19-09) : un incrément refusé arrête la boucle.
    fs::directory_iterator it(plans, error);
    for (; !error && it != fs::directory_iterator(); it.increment(error)) {
      std::error_code kind;
      if (it->is_regular_file(kind) && it->path().extension() == ".shots") drop(it->path());
    }
    error.clear();
    if (fs::is_empty(plans, error) && !error) drop(plans);
  }
  if (const fs::path updates = shared / "updates"; update_leftovers(updates)) drop(updates);
  return removed;
}

int purge_update_leftovers(const std::filesystem::path& data,
                           const std::function<void(const std::string&)>& said) {
  namespace fs = std::filesystem;
  if (data.empty()) return 0;
  const fs::path updates = data / "updates";
  if (!update_leftovers(updates)) return 0;
  std::error_code error;
  const auto count = fs::remove_all(updates, error);
  if (error) {
    if (said) said(std::format("{} : {}", core::to_utf8(updates), error.message()));  // i18n-ok : journal
    return 0;
  }
  if (said && count > 0)
    said(std::format("{} retiré, ancienne mise à jour", core::to_utf8(updates)));  // i18n-ok : journal
  return count > 0 ? 1 : 0;
}

}  // namespace ajp::app::play
