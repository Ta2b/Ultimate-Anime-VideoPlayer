#include "app/play/engine_install.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/engine_tools.h"
#include "app/play/engine_versions.h"
#include "app/play/i18n.h"
#include "core/dir_entries.h"
#include "core/file_sync.h"
#include "core/inflate.h"
#include "core/log.h"
#include "core/text.h"
#include "core/utf8_path.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

using prepare::ansi_safe_path;  // curl et tar relisent leur ligne de commande en ANSI sous Windows
core::Os host_os() {
  return core::kHostOs;
}

EngineCatalog load_catalog(const std::filesystem::path& index, const std::filesystem::path& data_dir,
                           const std::filesystem::path& engines_dir) {
  EngineCatalog catalog{.layout = core::install_layout(host_os(), data_dir)};
  if (!engines_dir.empty()) {
    const core::InstallLayout moved = core::install_layout(host_os(), engines_dir);
    catalog.layout.engines = moved.engines;
    catalog.layout.downloads = moved.downloads;
  }
  std::ifstream in(index, std::ios::binary);
  if (!in) {
    catalog.problem = i18n::tr_format("{} missing: only bundled engines are offered", core::to_utf8(index));
    return catalog;
  }
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  auto parsed = core::read_engine_index(text);
  if (!parsed) {
    catalog.problem = std::format("{} : {}", core::to_utf8(index), parsed.error().detail);
    return catalog;
  }
  for (core::EnginePackage& package : parsed->packages) {
    if (package.platform != catalog.layout.os) continue;
    // Un fichier NVIDIA écarté (un champ inconnu d'une release plus récente) : le paquet s'installerait sans
    // sa bibliothèque et se dirait à jour sans se charger (B06-02).
    if (core::vendor_file_skipped(*parsed, package.engine, catalog.layout.os)) {
      catalog.problem += std::format("{}{} : un fichier tiers de {} est écarté, paquet écarté",
                                     catalog.problem.empty() ? "" : " ; ", core::to_utf8(index),
                                     core::engine_name(package.engine));
      continue;
    }
    catalog.packages.push_back(std::move(package));
  }
  for (core::VendorFile& file : parsed->vendor_files) {
    if (file.platform == catalog.layout.os) catalog.vendor_files.push_back(std::move(file));
  }
  // Une entrée écartée (un moteur d'une release plus récente) n'empêche pas les autres.
  const auto add_problem = [&](const std::string& what) {
    catalog.problem +=
        std::format("{}{} : {}", catalog.problem.empty() ? "" : " ; ", core::to_utf8(index), what);
  };
  for (const Refusal& skipped : parsed->skipped)
    add_problem(i18n::tr_format("discarded, {}", skipped.detail));
  return catalog;
}

const core::EnginePackage* package_for(const EngineCatalog& catalog, core::EngineId engine) {
  return core::find_package(catalog.packages, engine, catalog.layout.os);
}

bool installed(const EngineCatalog& catalog, const core::EnginePackage& package) {
  std::error_code error;
  return std::filesystem::is_regular_file(core::backend_library(catalog.layout, package), error);
}

std::optional<InstalledPackage> installed_package(const EngineCatalog& catalog,
                                                  const core::EnginePackage& package) {
  return read_package_note(core::engine_dir(catalog.layout, package));
}

std::vector<std::filesystem::path> installed_backends(const EngineCatalog& catalog) {
  std::vector<std::filesystem::path> out;
  // Un ancien paquet MIGraphX ou OpenVINO installé ne s'ouvre plus : TensorRT seul.
  const core::EnginePackage* package = package_for(catalog, core::EngineId::TensorRt);
  if (package != nullptr && installed(catalog, *package))
    out.push_back(core::backend_library(catalog.layout, *package));
  return out;
}

int purge_stale_downloads(const EngineCatalog& catalog, std::string_view keep) {
  namespace fs = std::filesystem;
  std::vector<std::string> known{std::string(keep)};
  for (const core::EnginePackage& package : catalog.packages) known.push_back(package.sha256);
  for (const core::VendorFile& file : catalog.vendor_files) known.push_back(file.sha256);
  int removed = 0;
  std::error_code error;
  fs::directory_iterator it(catalog.layout.downloads, error);
  for (; !error && it != fs::directory_iterator(); it.increment(error)) {
    // « <sha256>.<suite> » : un paquet, un .part ou un morceau ; le reste n'est pas à nous.
    const std::string name = core::to_utf8(it->path().filename());
    const std::string sha = name.substr(0, name.find('.'));
    if (sha.size() != 64 ||
        !std::ranges::all_of(sha, [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }))
      continue;
    if (std::ranges::find(known, sha) != known.end()) continue;
    std::error_code gone;
    if (fs::remove_all(it->path(), gone) > 0 && !gone) ++removed;
  }
  return removed;
}

namespace {

std::unexpected<Refusal> deferred_swap(const std::filesystem::path& target, const std::error_code& error) {
  log_channel("app")->warn(
      "{} tenu ({}) : la nouvelle version attend le prochain lancement",  // i18n-ok : journal
      core::to_utf8(target.filename()), error.message());
  return std::unexpected(
      Refusal{RefusalCode::FileUnwritable,
              i18n::tr("the new version is ready: it replaces the installed one at the next launch")});
}

// Les étapes de l'installation ; `step` dit où elle en est, pour le journal si elle échoue. Une seule
// progression globale (octets tirés, archive et fichiers tiers) au lieu d'une barre par fichier.
// Chaque fichier d'un dossier extrait, sur le disque : FlushFileBuffers sous Windows, fsync ailleurs.
std::expected<void, Refusal> sync_tree(const std::filesystem::path& root) {
  std::error_code error;
  std::filesystem::recursive_directory_iterator it(root, error);
  for (; !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
    std::error_code kind;
    if (!it->is_regular_file(kind) || it->is_symlink(kind)) continue;
    if (const std::error_code synced = core::sync_file(it->path()))
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", core::to_utf8(it->path()), synced.message()));
  }
  if (error)
    return refuse(RefusalCode::FileUnreadable, std::format("{} : {}", core::to_utf8(root), error.message()));
  return {};
}

// Chaque fichier extrait sur le disque, puis la note du paquet en dernier, avec son dossier (A36-03).
std::expected<void, Refusal> seal_staging(const std::filesystem::path& staging,
                                          const core::EnginePackage& package,
                                          const std::vector<core::VendorFile>& vendor,
                                          std::string_view arch) {
  if (auto synced = sync_tree(staging); !synced) return synced;
  const std::filesystem::path path = staging / std::string(kPackageNote);
  std::ofstream note(path, std::ios::trunc);
  note << "sha256=" << package.sha256 << "\nlicence=" << package.license << '\n';
  // L'architecture pour laquelle les fichiers tiers ont été choisis : changer de carte (sm86 vers sm89) rend
  // le paquet à compléter, pas « à jour » sans la ressource du nouveau GPU (C06-03).
  if (!arch.empty()) note << "arch=" << arch << '\n';
  // Les fichiers tiers posés, avec leur taille : installed_state les revérifie (B06-11).
  for (const core::VendorFile& file : vendor) note << "fichier=" << file.size << ' ' << file.name << '\n';
  note.close();
  if (!note || core::sync_file(path))
    return refuse(RefusalCode::FileUnwritable, i18n::tr_format("{}: package note", core::to_utf8(staging)));
  core::sync_directory(staging);
  return {};
}

// Le dossier extrait et complet devient la version installée : ses fichiers sur le disque, la note, puis les
// renommages (l'ancien dossier mis de côté, remis en place sur échec).
std::expected<void, Refusal> swap_in_place(const core::EnginePackage& package,
                                           const std::vector<core::VendorFile>& vendor, std::string_view arch,
                                           const std::filesystem::path& staging,
                                           const std::filesystem::path& target) {
  std::error_code error;
  // Chaque fichier sur le disque, puis la note en dernier (A36-03) : après une coupure, un dossier de version
  // (ou un « .partiel » qui a sa note, recover_interrupted_install) est entier, jamais fait de fichiers
  // vides.
  if (const auto sealed = seal_staging(staging, package, vendor, arch); !sealed) {
    std::filesystem::remove_all(staging, error);
    return std::unexpected(sealed.error());
  }
  // L'ancien dossier est mis de côté avant d'être remplacé, puis effacé : si l'un de ses fichiers est
  // encore ouvert (Windows), il ne reste pas à moitié supprimé, et il est remis en place sur échec.
  std::filesystem::path aside = target;
  aside += ".ancien";
  std::filesystem::remove_all(aside, error);
  const bool replaced = std::filesystem::exists(target, error);
  if (replaced) {
    error.clear();
    // Defender ou l'indexeur tiennent un fichier quelques secondes après l'extraction : on réessaie
    // avant de renvoyer au prochain lancement (B31-06).
    error = core::rename_retrying(target, aside, kInstallRenamePatience);
    // Windows : une DLL de la version en place est encore chargée. La nouvelle, entière et notée, attend
    // dans « .partiel » : recover_interrupted_install l'échange au lancement suivant (A06-09).
    if (error) return deferred_swap(target, error);
  }
  error.clear();
  error = core::rename_retrying(staging, target, kInstallRenamePatience);
  if (error) {
    std::string reason = error.message();
    if (replaced) {  // l'ancien revient, au mieux ; si c'est impossible, le message le dit
      const std::error_code back = core::rename_retrying(aside, target, kInstallRenamePatience);
      if (back) reason += std::format(" (l'ancien n'est pas revenu : {})", back.message());
    }
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", core::to_utf8(target), reason));
  }
  core::sync_directory(target.parent_path());
  if (replaced) std::filesystem::remove_all(aside, error);
  error.clear();
  return {};
}

std::expected<std::filesystem::path, Refusal> install_steps(const EngineCatalog& catalog,
                                                            const core::EnginePackage& package,
                                                            const InstallProgress& progress,
                                                            std::string& step) {
  const auto curl = system_tool("curl");
  const auto tar = system_tool("tar");
  if (!curl || !tar)
    return refuse(RefusalCode::LibraryUnavailable, i18n::tr("curl or tar not found on this system"));
  const std::string name(core::engine_name(package.engine));
  const std::vector<core::VendorFile> vendor =
      core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, progress.cuda_arch);
  double total_bytes = static_cast<double>(package.size);
  for (const core::VendorFile& file : vendor) total_bytes += static_cast<double>(file.packed_size);
  double done_bytes = 0.0;
  float shown = 0.0F;
  // Un pourcentage qui ne recule jamais : octets tirés sur octets à tirer.
  const auto overall = [&](double local_bytes, const std::string& stage) {
    shown = std::max(shown, static_cast<float>(std::min(0.99, (done_bytes + local_bytes) / total_bytes)));
    if (progress.report) progress.report(shown, stage);
  };
  const auto report = [&](float fraction, const std::string& stage) {
    if (progress.report) progress.report(fraction, stage);
  };
  std::error_code error;
  step = i18n::tr("checking free space");
  // La place d'abord : un refus net, sans téléchargement, plutôt qu'un disque qui se remplit puis une
  // installation recommencée.
  const std::filesystem::path target_dir = core::engine_dir(catalog.layout, package);
  const InstallSpace space = install_space_in(catalog, package, progress.cuda_arch, target_dir);
  log_channel("app")->info(
      "installation de {} : architecture {}, à tirer {}, installé {}, pic {}, libre {} ({})",
      name,  // i18n-ok : journal
      progress.cuda_arch.empty() ? i18n::tr("unknown (all files)") : progress.cuda_arch,
      format_size(space.download), format_size(space.installed), format_size(space.peak),
      format_size(free_bytes(catalog.layout.engines)), core::to_utf8(catalog.layout.engines.filename()));
  if (const auto fits = check_space(space, free_bytes(catalog.layout.engines), catalog.layout.engines); !fits)
    return std::unexpected(fits.error());
  const std::filesystem::path part = core::download_path(catalog.layout, package.sha256);
  std::filesystem::create_directories(part.parent_path(), error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", core::to_utf8(part.parent_path()), error.message()));

  // Sous Windows, curl et tar relisent leur ligne de commande dans la page ANSI : un chemin qu'elle ne sait
  // pas écrire et qui n'a pas de nom court (8.3) deviendrait « ???? », et l'échec ne dirait pas pourquoi. On
  // le sait avant de tirer un octet (C06-07). Toujours vrai hors de Windows.
  {
    std::filesystem::path staging_dir = core::engine_dir(catalog.layout, package);
    staging_dir += ".partiel";
    for (const std::filesystem::path& each : {part, staging_dir, catalog.layout.downloads}) {
      if (!prepare::ansi_command_path(each)) {
        return refuse(
            RefusalCode::FileUnwritable,
            i18n::tr_format("this folder cannot be passed to curl and tar (name outside the Windows "
                            "ANSI code page, no short 8.3 name): {}. Choose another engines folder "
                            "or enable \"Use Unicode UTF-8\" in the Windows region settings",
                            core::to_utf8(each)));
      }
    }
  }
  // Le transfert reprend un .part laissé par un essai interrompu ; un .part déjà complet mais faux
  // repart de zéro.
  const auto resumed = core::verify_file(part, package.size, package.sha256, progress.stop);
  if (!resumed && resumed.error().code == RefusalCode::Cancelled) return std::unexpected(resumed.error());
  if (!resumed) {
    if (std::filesystem::file_size(part, error) >= package.size && !error)
      std::filesystem::remove(part, error);
    step = i18n::tr("downloading the engine archive");
    overall(0.0, i18n::tr_format("downloading {} ({})", name, format_size(package.size)));
    auto fetched = run_tool(
        {.program = *curl,
         .args =
             [&] {
               std::vector<std::string> args = curl_download_args(progress.allow_local_files, package.url);
               args.insert(args.end(), {"--continue-at", "-", "--progress-bar", "--output",
                                        ansi_safe_path(part), package.url});
               return args;
             }(),
         .timeout = std::chrono::hours(4)},
        [&](std::string_view line) {
          if (const auto f = curl_percent(line))
            overall(static_cast<double>(*f) * static_cast<double>(package.size),
                    i18n::tr_format("downloading {}", name));
        },
        progress.stop, i18n::tr_format("downloading {}", name));
    if (!fetched) return std::unexpected(fetched.error());
    step = i18n::tr("verifying the archive");
    overall(static_cast<double>(package.size), i18n::tr_format("verifying {}", name));
    if (auto ok = core::verify_file(part, package.size, package.sha256, progress.stop); !ok) {
      std::filesystem::remove(part, error);  // un fichier faux ne se reprend pas
      return std::unexpected(ok.error());
    }
  }

  // Extraction à côté, puis renommage : un dossier de version n'existe qu'entier.
  const std::filesystem::path target = core::engine_dir(catalog.layout, package);
  std::filesystem::path staging = target;
  staging += ".partiel";
  std::filesystem::remove_all(staging, error);
  std::filesystem::create_directories(staging, error);
  if (error) return refuse(RefusalCode::FileUnwritable, core::to_utf8(staging) + " : " + error.message());
  step = i18n::tr("extracting the archive");
  done_bytes = static_cast<double>(package.size);
  overall(0.0, i18n::tr_format("extracting {}", name));
  if (auto ok = run_tool({.program = *tar,
                          .args = {"-xzf", ansi_safe_path(part), "-C", ansi_safe_path(staging)},
                          .timeout = std::chrono::minutes(30)},
                         {}, progress.stop, i18n::tr_format("extracting {}", name));
      !ok) {
    std::filesystem::remove_all(staging, error);
    return std::unexpected(ok.error());
  }
  // L'archive reste jusqu'à l'échange réussi : si un fichier tiers échoue (disque plein, réseau), la reprise
  // la retrouve vérifiée au lieu de retélécharger le moteur (C06-05). Son espace est déjà dans le pic estimé.
  const std::filesystem::path library = staging / core::backend_library(catalog.layout, package).filename();
  if (!std::filesystem::is_regular_file(library, error)) {
    std::filesystem::remove_all(staging, error);
    return refuse(RefusalCode::HashMismatch,
                  i18n::tr_format("package {} without {}", name, core::to_utf8(library.filename())));
  }
  // Les fichiers que NVIDIA publie, tirés de ses archives et joints au module : ce qui est à nous
  // vient de notre release, le reste du lien direct (core::VendorFile).
  for (std::size_t i = 0; i < vendor.size(); ++i) {
    const core::VendorFile& file = vendor[i];
    step = i18n::tr_format("file {} ({}/{})", file.name, i + 1, vendor.size());
    const std::string label = std::format("{} ({}/{})", file.name, i + 1, vendor.size());
    overall(0.0, i18n::tr_format("downloading {}", label));
    log_channel("app")->info("installation de {} : {} ({} tirés, {} écrits)", name,
                             step,  // i18n-ok : journal
                             format_size(file.packed_size), format_size(file.size));
    InstallProgress local = progress;
    local.report = [&, label](float fraction, const std::string&) {
      overall(static_cast<double>(fraction) * static_cast<double>(file.packed_size),
              fraction >= 0.9F ? i18n::tr_format("decompressing {}", label)
                               : i18n::tr_format("downloading {}", label));
    };
    // P3.8 : un fichier de même SHA-256 déjà installé (version précédente du module, autre moteur du même
    // dossier) se reprend sans retélécharger ses centaines de Mo.
    if (reuse_vendor_file(file, staging / file.name, target.parent_path(), progress.stop)) {
      log_channel("app")->info("installation de {} : {} repris de l'installation précédente", name,
                               file.name);  // i18n-ok : journal
      done_bytes += static_cast<double>(file.packed_size);
      continue;
    }
    if (auto ok = fetch_vendor_file(file, staging / file.name, catalog.layout.downloads, local); !ok) {
      std::filesystem::remove_all(staging, error);
      return std::unexpected(ok.error());
    }
    done_bytes += static_cast<double>(file.packed_size);
  }
  step = i18n::tr("replacing the installed version");
  if (auto swapped = swap_in_place(package, vendor, progress.cuda_arch, staging, target); !swapped)
    return std::unexpected(swapped.error());
  std::filesystem::remove(part, error);
  report(1.0F, i18n::tr_format("{} installed", name));
  return core::backend_library(catalog.layout, package);
}

}  // namespace

std::expected<std::filesystem::path, Refusal> install_package(const EngineCatalog& catalog,
                                                              const core::EnginePackage& package,
                                                              const InstallProgress& progress) {
  const auto log = log_channel("app");
  const std::string name(core::engine_name(package.engine));
  // Une installation coupée entre ses deux renommages remet d'abord sa copie complète (A06-08) ; les restes
  // d'un essai raté ou interrompu partent ensuite, avant tout nouvel essai : ils ne s'accumulent pas.
  recover_interrupted_install(catalog, package);
  std::error_code error;
  for (const char* suffix : {".partiel", ".ancien"}) {
    std::filesystem::path leftover = core::engine_dir(catalog.layout, package);
    leftover += suffix;
    // Un « .partiel » entier, noté, que recover_interrupted_install n'a pas pu échanger (DLL encore tenue)
    // est la version à venir : il reste tant que le nouvel essai n'a pas extrait la sienne (B06-03).
    if (std::string_view(suffix) == ".partiel" &&
        std::filesystem::is_regular_file(leftover / std::string(kPackageNote), error))
      continue;
    if (std::filesystem::exists(leftover, error)) {
      log->info("installation de {} : reste d'un essai précédent retiré ({})", name,
                core::to_utf8(leftover.filename()));
      std::filesystem::remove_all(leftover, error);
    }
  }
  // Les téléchargements coupés que l'index ne nomme plus (une autre version) ne servent plus (A06-04).
  if (const int purged = purge_stale_downloads(catalog, package.sha256); purged > 0)
    log->info("téléchargements : {} reste(s) d'un autre index retiré(s)", purged);
  std::string step = i18n::tr("starting");
  const auto started = std::chrono::steady_clock::now();
  auto result = install_steps(catalog, package, progress, step);
  const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started).count();
  if (result) {
    log->info("installation de {} : terminée en {} s", name, seconds);
  } else {
    log->warn("installation de {} : échec à l'étape « {} » après {} s : {} {} (libre : {})", name, step,
              seconds, ajp::name(result.error().code), result.error().detail,
              format_size(free_bytes(catalog.layout.engines)));
  }
  return result;
}

}  // namespace ajp::app::play
