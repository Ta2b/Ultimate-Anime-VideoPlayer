#include "app/play/engine_versions.h"

#include <charconv>
#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "core/file_sync.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

// Un dossier de remplacement en cours (« 11.partiel », « 11.ancien ») n'est jamais une version.
bool transient(const fs::path& dir) {
  const fs::path extension = dir.extension();
  return extension == ".partiel" || extension == ".ancien";
}

// Les dossiers frères de la version de l'index qui portent le module ou la note du paquet.
std::vector<fs::path> other_versions(const EngineCatalog& catalog, const core::EnginePackage& package) {
  const fs::path current = core::engine_dir(catalog.layout, package);
  const fs::path module = core::backend_library(catalog.layout, package).filename();
  std::vector<fs::path> found;
  std::error_code error;
  fs::directory_iterator it(current.parent_path(), error);
  for (; !error && it != fs::directory_iterator(); it.increment(error)) {
    const fs::path& dir = it->path();
    std::error_code ignored;
    if (dir.filename() == current.filename() || transient(dir) || !it->is_directory(ignored)) continue;
    if (fs::is_regular_file(dir / module, ignored) ||
        fs::is_regular_file(dir / std::string(kPackageNote), ignored))
      found.push_back(dir);
  }
  return found;
}

}  // namespace

std::optional<InstalledPackage> read_package_note(const fs::path& dir) {
  std::ifstream in(dir / std::string(kPackageNote));
  if (!in) return std::nullopt;
  InstalledPackage found;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.starts_with("sha256=")) found.sha256 = line.substr(7);
    if (line.starts_with("licence=")) found.license = line.substr(8);
    if (line.starts_with("fichier=")) {  // « fichier=<taille> <nom> »
      const std::string_view rest = std::string_view(line).substr(8);
      const std::size_t space = rest.find(' ');
      std::uint64_t size = 0;
      if (space == std::string_view::npos || space + 1 >= rest.size()) continue;
      const std::string_view digits = rest.substr(0, space);
      const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), size);
      if (error != std::errc{} || end != digits.data() + digits.size()) continue;
      found.files.emplace_back(std::string(rest.substr(space + 1)), size);
    }
  }
  if (found.sha256.empty()) return std::nullopt;
  return found;
}

InstalledState installed_state(const EngineCatalog& catalog, const core::EnginePackage& package) {
  const fs::path current = core::engine_dir(catalog.layout, package);
  if (installed(catalog, package)) {
    InstalledState here{.state = PackageState::Replaced, .dir = current, .note = read_package_note(current)};
    if (here.note && here.note->sha256 == package.sha256) {
      here.state = PackageState::Current;
      // « Installé » ne veut pas dire « le module existe » : un fichier tiers mis en quarantaine par un
      // antivirus laisse un dossier « à jour » qui ne se charge pas (B06-11). La note en garde les noms et
      // les tailles.
      for (const auto& [name, size] : here.note->files) {
        std::error_code missing;
        const std::uintmax_t actual = fs::file_size(current / core::path_from_utf8(name), missing);
        if (missing || actual != size) {
          here.state = PackageState::Replaced;
          here.missing = name;
          break;
        }
      }
    }
    return here;
  }
  if (const auto other = other_installed_version(catalog, package))
    return {.state = PackageState::NewVersion, .dir = *other, .note = read_package_note(*other)};
  return {};
}

std::optional<fs::path> other_installed_version(const EngineCatalog& catalog,
                                                const core::EnginePackage& package) {
  const std::vector<fs::path> others = other_versions(catalog, package);
  std::optional<fs::path> newest;
  fs::file_time_type newest_time{};
  for (const fs::path& dir : others) {
    std::error_code error;
    const fs::file_time_type at = fs::last_write_time(dir, error);
    if (!newest || (!error && at > newest_time)) {
      newest = dir;
      newest_time = error ? fs::file_time_type{} : at;
    }
  }
  return newest;
}

int remove_other_versions(const EngineCatalog& catalog, const core::EnginePackage& package) {
  int removed = 0;
  for (const fs::path& dir : other_versions(catalog, package)) {
    std::error_code error;
    fs::remove_all(dir, error);
    if (error) {
      log_channel("app")->warn("moteurs : {} non retiré : {}", core::to_utf8(dir.filename()),
                               error.message());
      continue;
    }
    ++removed;
    log_channel("app")->info("moteurs : ancienne version retirée ({})", core::to_utf8(dir.filename()));
  }
  return removed;
}

namespace {

bool finish_deferred_swap(const fs::path& target, const fs::path& staging, const fs::path& aside) {
  std::error_code error;
  if (!read_package_note(staging) || !fs::is_directory(staging, error)) return false;
  fs::remove_all(aside, error);
  // Defender ouvre les fichiers qu'il vient de voir : le renommage se réessaie (B31-06).
  error = core::rename_retrying(target, aside, kInstallRenamePatience);
  if (error) {
    log_channel("app")->warn("moteurs : {} toujours tenu, remplacement remis : {}",
                             core::to_utf8(target.filename()), error.message());
    return false;
  }
  error = core::rename_retrying(staging, target, kInstallRenamePatience);
  if (error) {
    const std::error_code back = core::rename_retrying(aside, target, kInstallRenamePatience);
    log_channel("app")->warn("moteurs : {} non mis en place : {}{}", core::to_utf8(staging.filename()),
                             error.message(),
                             back ? " (l'ancien n'est pas revenu : " + back.message() + ")" : "");
    return false;
  }
  fs::remove_all(aside, error);
  log_channel("app")->info("moteurs : nouvelle version mise en place au lancement ({})",
                           core::to_utf8(target.filename()));
  return true;
}

}  // namespace

bool recover_interrupted_install(const EngineCatalog& catalog, const core::EnginePackage& package) {
  const fs::path target = core::engine_dir(catalog.layout, package);
  std::error_code error;
  fs::path staging = target;
  staging += ".partiel";
  fs::path aside = target;
  aside += ".ancien";
  const bool present = fs::exists(target, error);
  if (error) return false;
  // Un remplacement remis au lancement suivant (A06-09) : la version en place tenait encore une DLL ouverte,
  // la nouvelle attend entière dans « .partiel ». Rien n'est chargé encore : l'échange se fait maintenant.
  if (present) return finish_deferred_swap(target, staging, aside);
  // La note s'écrit en dernier, après chaque fichier sur le disque : un « .partiel » qui l'a est complet.
  for (const fs::path& copy : {staging, aside}) {
    if (copy == staging && !read_package_note(copy)) continue;
    if (!fs::is_directory(copy, error)) continue;
    error = core::rename_retrying(copy, target, kInstallRenamePatience);
    if (error) {
      log_channel("app")->warn("moteurs : {} non remis en place : {}", core::to_utf8(copy.filename()),
                               error.message());
      return false;
    }
    log_channel("app")->info("moteurs : installation coupée, {} remis en place",
                             core::to_utf8(copy.filename()));
    // La nouvelle version en place, la précédente mise de côté ne sert plus.
    if (copy == staging) fs::remove_all(aside, error);
    return true;
  }
  return false;
}

}  // namespace ajp::app::play
