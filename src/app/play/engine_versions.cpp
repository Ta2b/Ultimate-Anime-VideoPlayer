#include "app/play/engine_versions.h"

#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

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
  }
  if (found.sha256.empty()) return std::nullopt;
  return found;
}

InstalledState installed_state(const EngineCatalog& catalog, const core::EnginePackage& package) {
  const fs::path current = core::engine_dir(catalog.layout, package);
  if (installed(catalog, package)) {
    InstalledState here{.state = PackageState::Replaced, .dir = current, .note = read_package_note(current)};
    if (here.note && here.note->sha256 == package.sha256) here.state = PackageState::Current;
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

bool recover_interrupted_install(const EngineCatalog& catalog, const core::EnginePackage& package) {
  const fs::path target = core::engine_dir(catalog.layout, package);
  std::error_code error;
  if (fs::exists(target, error) || error) return false;
  fs::path staging = target;
  staging += ".partiel";
  fs::path aside = target;
  aside += ".ancien";
  // La note s'écrit en dernier, après chaque fichier sur le disque : un « .partiel » qui l'a est complet.
  for (const fs::path& copy : {staging, aside}) {
    if (copy == staging && !read_package_note(copy)) continue;
    if (!fs::is_directory(copy, error)) continue;
    fs::rename(copy, target, error);
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
