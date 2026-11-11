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
    catalog.problem = i18n::tr_format("{} missing: only bundled engines are offered", index.string());
    return catalog;
  }
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  auto parsed = core::read_engine_index(text);
  if (!parsed) {
    catalog.problem = std::format("{} : {}", index.string(), parsed.error().detail);
    return catalog;
  }
  for (core::EnginePackage& package : parsed->packages) {
    if (package.platform != catalog.layout.os) continue;
    // Un fichier NVIDIA écarté (un champ inconnu d'une release plus récente) : le paquet s'installerait sans
    // sa bibliothèque et se dirait à jour sans se charger (B06-02).
    if (core::vendor_file_skipped(*parsed, package.engine, catalog.layout.os)) {
      catalog.problem += std::format("{}{} : un fichier tiers de {} est écarté, paquet écarté",
                                     catalog.problem.empty() ? "" : " ; ", index.string(),
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
    catalog.problem += std::format("{}{} : {}", catalog.problem.empty() ? "" : " ; ", index.string(), what);
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

namespace {

// Un mot entier : « a100 » n'est pas dans « rtx a1000 » (une RTX Ampere sm86, pas une A100 sm80 : B06-06).
bool contains_word(std::string_view text, std::string_view word) {
  for (std::size_t at = text.find(word); at != std::string_view::npos; at = text.find(word, at + 1)) {
    const bool before = at == 0 || std::isalnum(static_cast<unsigned char>(text[at - 1])) == 0;
    const std::size_t end = at + word.size();
    const bool after = end >= text.size() || std::isalnum(static_cast<unsigned char>(text[end])) == 0;
    if (before && after) return true;
  }
  return false;
}

}  // namespace

std::string cuda_arch_from_name(std::string_view gpu_name) {
  std::string name(gpu_name);
  name = core::ascii_lower(name);
  const auto has = [&](std::string_view part) { return name.find(part) != std::string::npos; };
  const auto has_word = [&](std::string_view word) { return contains_word(name, word); };
  const auto series = [&](char digit) {  // « rtx 4050 » : la série à deux chiffres, ici 40
    const std::size_t at = name.find("rtx ");
    return at != std::string::npos && at + 5 < name.size() && name[at + 4] == digit && name[at + 5] >= '0' &&
           name[at + 5] <= '9';
  };
  if (has("h100") || has("h200")) return "sm90";
  // Deux exceptions à la règle par série (A37-03) : la RTX 2050 portable est une Ampere (GA107), la
  // RTX 3050 A Laptop une Ada (AD106).
  if (has("rtx 2050")) return "sm86";
  if (has("3050 a ")) return "sm89";
  if (has_word("a100")) return "sm80";
  if (has("blackwell")) return "sm120";
  // Les cartes pro : « RTX 5000 Ada Generation » est une Ada, « Quadro RTX 4000 » une Turing (P3.16).
  if (has(" ada")) return "sm89";
  if (has("quadro rtx")) return "sm75";
  if (series('5')) return "sm120";
  if (has("l40") || series('4')) return "sm89";
  if (series('3') || has("rtx a")) return "sm86";
  if (series('2') || has("gtx 16") || has("titan rtx")) return "sm75";
  return {};
}

bool fp8_presumed(std::string_view gpu_name) {
  // gpu::fp8_allowed : Ada et Hopper ; une RTX 50 (sm120) n'a pas de FP8 tant qu'aucune mesure ne le
  // justifie.
  const std::string arch = cuda_arch_from_name(gpu_name);
  return arch == "sm89" || arch == "sm90";
}

bool span_fp8_presumed(std::string_view gpu_name) {
  return cuda_arch_from_name(gpu_name) == "sm89";
}

std::string effective_cuda_arch(std::string_view detected, std::string_view gpu_name) {
  if (!detected.empty()) return std::string(detected);
  if (std::string by_name = cuda_arch_from_name(gpu_name); !by_name.empty()) return by_name;
  return "ptx";  // inconnue : jamais « toutes les architectures » (2 Go tirés, 3,3 Go écrits)
}

namespace {

// Chaque carte : son nom et son architecture (« NVIDIA GeForce RTX 4070, 8.9 » : sm89).
std::vector<std::pair<std::string, std::string>> run_nvidia_smi() {
  std::vector<std::pair<std::string, std::string>> cards;
  const auto smi = system_tool("nvidia-smi");
  if (!smi) return cards;
  auto done = run_tool(
      {.program = *smi,
       .args = {"--query-gpu=name,compute_cap", "--format=csv,noheader"},
       .timeout = std::chrono::seconds(10)},
      [&](std::string_view line) {
        const std::size_t comma = line.rfind(',');
        if (comma == std::string_view::npos) return;
        std::string_view cap = line.substr(comma + 1);
        while (!cap.empty() && (cap.front() == ' ' || cap.back() == ' ' || cap.back() == '\r'))
          cap = cap.front() == ' ' ? cap.substr(1) : cap.substr(0, cap.size() - 1);
        const std::size_t dot = cap.find('.');
        if (dot == std::string_view::npos || dot == 0 || dot + 1 >= cap.size()) return;
        const auto digit = [](char c) { return c >= '0' && c <= '9'; };
        if (!digit(cap[dot - 1]) || !digit(cap[dot + 1])) return;
        // sm100 et plus : « 10.0 » donne sm100 ; la majeure a deux chiffres.
        std::string arch = std::format("sm{}{}", cap.substr(0, dot), cap[dot + 1]);
        cards.emplace_back(std::string(line.substr(0, comma)), std::move(arch));
      },
      {}, "nvidia-smi");
  if (!done) cards.clear();
  return cards;
}

}  // namespace

std::string detect_cuda_arch(std::string_view gpu_name) {
  // Une fois par processus (P2.12) : la carte ne change pas, et nvidia-smi prend jusqu'à deux secondes. Un
  // résultat vide (échec, délai de 10 s juste après l'ouverture de session, pilote pas encore prêt) n'est pas
  // retenu pour toujours : il se retente, deux essais au plus (B06-07).
  static std::mutex mutex;
  static std::optional<std::vector<std::pair<std::string, std::string>>> known;
  static int attempts = 0;
  const std::scoped_lock lock(mutex);
  if (!known || (known->empty() && attempts < 2)) {
    known = run_nvidia_smi();
    ++attempts;
  }
  // La carte de rendu (P3.16) : sur une machine à deux cartes NVIDIA, la première n'est pas forcément elle.
  for (const auto& [name, arch] : *known)
    if (!gpu_name.empty() && name == gpu_name) return arch;
  // Un nom demandé que nvidia-smi n'a pas écrit pareil (espace, suffixe, carte pro) : avec plusieurs cartes,
  // on ne devine pas, l'architecture se déduit du nom (effective_cuda_arch) ; avec une seule, c'est elle
  // (B37-05).
  if (!gpu_name.empty() && known->size() > 1) return {};
  return known->empty() ? std::string() : known->front().second;
}

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
// engines_root. Le SHA-256 n'est pas recalculé ici (1 Go à lire) : c'est une estimation de place, la
// vérification se fait à l'installation. Un voisin altéré à taille égale rend le pic de place sous-estimé de
// sa taille : l'écart reste sous la marge de check_space, et le cas est rare (B06-09).
bool vendor_on_hand(const core::VendorFile& file, const std::filesystem::path& engines_root) {
  if (engines_root.empty()) return false;
  std::error_code error;
  std::filesystem::directory_iterator it(engines_root, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    std::error_code checked;
    const std::filesystem::path candidate = it->path() / file.name;
    if (std::filesystem::is_regular_file(candidate, checked) &&
        std::filesystem::file_size(candidate, checked) == file.size && !checked)
      return true;
  }
  return false;
}

}  // namespace

InstallSpace install_space(const EngineCatalog& catalog, const core::EnginePackage& package,
                           std::string_view arch, std::uint64_t existing,
                           const std::filesystem::path& engines_root) {
  InstallSpace space;
  space.download = package.size;
  // L'archive du moteur, puis son extraction (module et outils ; l'estimation vaut 3 fois l'archive).
  std::uint64_t staged = package.size * 3;
  std::uint64_t highest = package.size + staged;
  std::uint64_t linked = 0;
  for (const core::VendorFile& file :
       core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, arch)) {
    // Repris de la version en place par un lien dur : ni téléchargé ni écrit une seconde fois (A06-06).
    if (vendor_on_hand(file, engines_root)) {
      linked += file.size;
      continue;
    }
    space.download += file.packed_size;
    // Chaque fichier tiers : son téléchargement et sa copie décompressée coexistent avec ce qui est déjà
    // posé.
    highest = std::max(highest, package.size + staged + file.packed_size + file.size);
    staged += file.size;
  }
  space.installed = staged + linked;
  const std::uint64_t peak = std::max(highest, package.size + staged) + existing;
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
          format_size(available), where.string()));
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
                                          const std::vector<core::VendorFile>& vendor) {
  if (auto synced = sync_tree(staging); !synced) return synced;
  const std::filesystem::path path = staging / std::string(kPackageNote);
  std::ofstream note(path, std::ios::trunc);
  note << "sha256=" << package.sha256 << "\nlicence=" << package.license << '\n';
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
                                           const std::vector<core::VendorFile>& vendor,
                                           const std::filesystem::path& staging,
                                           const std::filesystem::path& target) {
  std::error_code error;
  // Chaque fichier sur le disque, puis la note en dernier (A36-03) : après une coupure, un dossier de version
  // (ou un « .partiel » qui a sa note, recover_interrupted_install) est entier, jamais fait de fichiers
  // vides.
  if (const auto sealed = seal_staging(staging, package, vendor); !sealed) {
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
    const std::string reason = error.message();
    if (replaced) core::rename_retrying(aside, target, kInstallRenamePatience);
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), reason));
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
      format_size(free_bytes(catalog.layout.engines)), catalog.layout.engines.string());
  if (const auto fits = check_space(space, free_bytes(catalog.layout.engines), catalog.layout.engines); !fits)
    return std::unexpected(fits.error());
  const std::filesystem::path part = core::download_path(catalog.layout, package.sha256);
  std::filesystem::create_directories(part.parent_path(), error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", part.parent_path().string(), error.message()));

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
               std::vector<std::string> args = curl_download_args(progress.allow_local_files);
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
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", staging.string(), error.message()));
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
  // L'archive du moteur n'est plus utile une fois extraite : sa place revient avant les gros téléchargements.
  std::filesystem::remove(part, error);
  const std::filesystem::path library = staging / core::backend_library(catalog.layout, package).filename();
  if (!std::filesystem::is_regular_file(library, error)) {
    std::filesystem::remove_all(staging, error);
    return refuse(RefusalCode::HashMismatch,
                  i18n::tr_format("package {} without {}", name, library.filename().string()));
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
  if (auto swapped = swap_in_place(package, vendor, staging, target); !swapped)
    return std::unexpected(swapped.error());
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
      log->info("installation de {} : reste d'un essai précédent retiré ({})", name, leftover.string());
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
