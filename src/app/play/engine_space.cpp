// La place que demande une installation de moteur : octets à tirer, octets installés, pic sur le disque, et
// le refus net avant tout téléchargement quand elle manque.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

#include "app/play/engine_install.h"
#include "app/play/i18n.h"
#include "core/dir_entries.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

std::uint64_t download_bytes(const EngineCatalog& catalog, const core::EnginePackage& package,
                             std::string_view arch) {
  std::uint64_t total = package.size;
  for (const core::VendorFile& file :
       core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, arch))
    total += file.packed_size;
  return total;
}

std::string format_size(std::uint64_t bytes) {
  constexpr double kMiB = 1048576.0;
  const double mib = static_cast<double>(bytes) / kMiB;
  if (mib >= 1024.0) return std::format("{:.1f} Go", mib / 1024.0);
  return std::format("{} Mo", static_cast<std::uint64_t>(std::lround(mib)));
}

std::string shown_size(std::uint64_t bytes) {
  constexpr double kMiB = 1048576.0;
  const double mib = static_cast<double>(bytes) / kMiB;
  if (mib >= 1024.0) return i18n::tr_format("{:.1f} GB", mib / 1024.0);
  return i18n::tr_format("{} MB", static_cast<std::uint64_t>(std::lround(mib)));
}

std::uint64_t free_bytes(const std::filesystem::path& where) {
  std::error_code error;
  std::filesystem::path probe = where;
  while (!probe.empty() && !std::filesystem::exists(probe, error)) {
    const std::filesystem::path parent = probe.parent_path();
    if (parent == probe) break;
    probe = parent;
  }
  const std::filesystem::space_info info = std::filesystem::space(probe, error);
  return error ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(info.available);  // inconnu : on essaie
}

std::uint64_t directory_bytes(const std::filesystem::path& dir) {
  std::error_code error;
  std::uint64_t total = 0;
  for (const std::filesystem::directory_entry& entry : core::tree_entries(dir, error)) {
    std::error_code each;
    if (entry.is_regular_file(each)) total += static_cast<std::uint64_t>(entry.file_size(each));
  }
  return total;
}

namespace {

// Un fichier tiers que reuse_vendor_file reprendra sans doute : même nom et même taille dans un dossier de
// engines_root, et un volume qui accepte les liens durs (exFAT et certains montages réseau n'en font pas : la
// reprise copierait alors la taille entière, C06-09). Le SHA-256 n'est pas recalculé ici (1 Go à lire) :
// c'est une estimation de place, la vérification se fait à l'installation. Un voisin altéré à taille égale
// rend le pic de place sous-estimé de sa taille : l'écart reste sous la marge de check_space, et le cas est
// rare (B06-09).
bool vendor_on_hand(const core::VendorFile& file, const std::filesystem::path& engines_root) {
  if (engines_root.empty()) return false;
  std::error_code error;
  std::filesystem::directory_iterator it(engines_root, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    std::error_code checked;
    const std::filesystem::path candidate = it->path() / file.name;
    if (!std::filesystem::is_regular_file(candidate, checked) ||
        std::filesystem::file_size(candidate, checked) != file.size || checked)
      continue;
    // Un essai de lien sur le fichier lui-même, retiré aussitôt : le volume dit s'il en accepte.
    std::filesystem::path probe = candidate;
    probe += ".ajp-lien";
    std::filesystem::create_hard_link(candidate, probe, checked);
    const bool linkable = !checked;
    std::filesystem::remove(probe, checked);
    if (linkable) return true;
  }
  return false;
}

// Les octets déjà tirés d'un téléchargement coupé (<sha256>.part, jamais plus que sa taille finale) : ils
// occupent déjà le disque, donc déjà retirés de la place libre (C06-09).
std::uint64_t part_bytes(const EngineCatalog& catalog, std::string_view sha256, std::uint64_t at_most) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(core::download_path(catalog.layout, sha256), error);
  return error ? 0 : std::min<std::uint64_t>(size, at_most);
}

}  // namespace

InstallSpace install_space(const EngineCatalog& catalog, const core::EnginePackage& package,
                           std::string_view arch, std::uint64_t existing,
                           const std::filesystem::path& engines_root) {
  InstallSpace space;
  const std::uint64_t archive_have = part_bytes(catalog, package.sha256, package.size);
  space.download = package.size - archive_have;
  // L'archive du moteur, puis son extraction (module et outils ; l'estimation vaut 3 fois l'archive).
  std::uint64_t staged = package.size * 3;
  std::uint64_t highest = package.size + staged;
  std::uint64_t linked = 0;
  std::uint64_t have = archive_have;  // déjà sur le disque, déjà absent de la place libre
  for (const core::VendorFile& file :
       core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, arch)) {
    // Repris de la version en place par un lien dur : ni téléchargé ni écrit une seconde fois (A06-06).
    if (vendor_on_hand(file, engines_root)) {
      linked += file.size;
      continue;
    }
    const std::uint64_t file_have = part_bytes(catalog, file.sha256, file.packed_size);
    space.download += file.packed_size - file_have;
    have += file_have;
    // Chaque fichier tiers : son téléchargement et sa copie décompressée coexistent avec ce qui est déjà
    // posé.
    highest = std::max(highest, package.size + staged + file.packed_size + file.size);
    staged += file.size;
  }
  space.installed = staged + linked;
  const std::uint64_t needed = std::max(highest, package.size + staged) + existing;
  const std::uint64_t peak = needed - std::min(needed, have);
  space.peak = peak + peak / 20;  // 5 % de marge
  return space;
}

InstallSpace install_space_in(const EngineCatalog& catalog, const core::EnginePackage& package,
                              std::string_view arch, const std::filesystem::path& installed_dir) {
  return install_space(catalog, package, arch, directory_bytes(installed_dir), installed_dir.parent_path());
}

std::expected<void, Refusal> check_space(const InstallSpace& space, std::uint64_t available,
                                         const std::filesystem::path& where) {
  if (available >= space.peak) return {};
  return refuse(
      RefusalCode::InsufficientSpace,
      i18n::tr_format(
          "not enough disk space: the install needs {} ({} to download, {} once installed), {} left on {}",
          format_size(space.peak), format_size(space.download), format_size(space.installed),
          format_size(available), core::to_utf8(where)));
}

}  // namespace ajp::app::play
