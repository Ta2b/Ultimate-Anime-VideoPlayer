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

namespace ajp::app::play {
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
    if (package.platform == catalog.layout.os) catalog.packages.push_back(std::move(package));
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

std::string cuda_arch_from_name(std::string_view gpu_name) {
  std::string name(gpu_name);
  name = core::ascii_lower(name);
  const auto has = [&](std::string_view part) { return name.find(part) != std::string::npos; };
  const auto series = [&](char digit) {  // « rtx 4050 » : la série à deux chiffres, ici 40
    const std::size_t at = name.find("rtx ");
    return at != std::string::npos && at + 5 < name.size() && name[at + 4] == digit && name[at + 5] >= '0' &&
           name[at + 5] <= '9';
  };
  if (has("h100") || has("h200")) return "sm90";
  if (has("a100")) return "sm80";
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
  // Une fois par processus (P2.12) : la carte ne change pas, et nvidia-smi prend jusqu'à deux secondes.
  static std::mutex mutex;
  static std::optional<std::vector<std::pair<std::string, std::string>>> known;
  const std::scoped_lock lock(mutex);
  if (!known) known = run_nvidia_smi();
  // La carte de rendu (P3.16) : sur une machine à deux cartes NVIDIA, la première n'est pas forcément elle.
  for (const auto& [name, arch] : *known)
    if (!gpu_name.empty() && name == gpu_name) return arch;
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

InstallSpace install_space(const EngineCatalog& catalog, const core::EnginePackage& package,
                           std::string_view arch, std::uint64_t existing) {
  InstallSpace space;
  space.download = package.size;
  // L'archive du moteur, puis son extraction (module et outils ; l'estimation vaut 3 fois l'archive).
  std::uint64_t staged = package.size * 3;
  std::uint64_t highest = package.size + staged;
  for (const core::VendorFile& file :
       core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, arch)) {
    space.download += file.packed_size;
    // Chaque fichier tiers : son téléchargement et sa copie décompressée coexistent avec ce qui est déjà
    // posé.
    highest = std::max(highest, package.size + staged + file.packed_size + file.size);
    staged += file.size;
  }
  space.installed = staged;
  const std::uint64_t peak = std::max(highest, package.size + staged) + existing;
  space.peak = peak + peak / 20;  // 5 % de marge
  return space;
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

bool reuse_vendor_file(const core::VendorFile& file, const std::filesystem::path& dest,
                       const std::filesystem::path& engines_root, const std::stop_token& stop) {
  std::error_code error;
  // Parcours sans exception (A06-10) : un dossier voisin illisible arrête la recherche, il ne fait pas
  // échouer l'installation.
  std::filesystem::directory_iterator it(engines_root, error);
  for (; !error && it != std::filesystem::directory_iterator(); it.increment(error)) {
    const std::filesystem::path candidate = it->path() / file.name;
    if (candidate.parent_path() == dest.parent_path()) continue;
    std::error_code checked;
    if (!std::filesystem::is_regular_file(candidate, checked) ||
        std::filesystem::file_size(candidate, checked) != file.size || checked)
      continue;
    if (!core::verify_file(candidate, file.size, file.sha256, stop)) continue;
    std::error_code placed;
    std::filesystem::create_directories(dest.parent_path(), placed);
    std::filesystem::remove(dest, placed);
    placed.clear();
    std::filesystem::create_hard_link(candidate, dest, placed);
    if (placed) {
      placed.clear();
      std::filesystem::copy_file(candidate, dest, std::filesystem::copy_options::overwrite_existing, placed);
    }
    if (!placed) return true;
  }
  return false;
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

// La taille d'un fichier, 0 s'il n'existe pas ; `error` dit s'il ne se lit pas.
std::uint64_t size_or_zero(const std::filesystem::path& path, std::error_code& error) {
  return std::filesystem::exists(path, error) ? std::filesystem::file_size(path, error) : 0;
}

// Ajoute au .part le morceau reçu, même coupé. Faux si le morceau dépasse la plage demandée (`limit`
// octets) : un serveur qui l'ignore rend la roue entière (200), le morceau est alors jeté (A06-07).
bool append_chunk(const std::filesystem::path& chunk, const std::filesystem::path& packed,
                  std::uint64_t limit) {
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::exists(chunk, error) ? std::filesystem::file_size(chunk, error) : 0;
  if (error) return true;  // rien de lisible : la reprise suivante le dira
  if (size > limit) return false;
  std::ifstream piece(chunk, std::ios::binary);
  std::ofstream whole(packed, std::ios::binary | std::ios::app);
  if (piece && whole) whole << piece.rdbuf();
  return true;
}

// Tire les octets compressés de `file` dans `packed`, par plages, en reprenant un .part laissé (P3.7).
std::expected<void, Refusal> pull_packed(const core::VendorFile& file, const std::filesystem::path& curl,
                                         const std::filesystem::path& packed,
                                         const std::filesystem::path& chunk,
                                         const InstallProgress& progress) {
  std::error_code error;
  // P3.7 : un .part laissé par un essai coupé se reprend par plage ; trop long, il repart de zéro.
  std::uint64_t have = size_or_zero(packed, error);
  if (error || have > file.packed_size) {
    std::filesystem::remove(packed, error);
    have = 0;
  }
  const std::uint64_t last = file.data_offset + file.packed_size - 1;
  // Trois reprises au plus après une coupure, chacune depuis ce qui est déjà là (curl réessaie aussi
  // lui-même).
  std::optional<Refusal> failure;
  for (int attempt = 0; attempt < 4 && have < file.packed_size; ++attempt) {
    std::filesystem::remove(chunk, error);
    std::vector<std::string> args = curl_download_args();
    // --max-filesize : un serveur ou un proxy qui ignore la plage et rend la roue entière (200) est arrêté
    // tout de suite (A06-07).
    args.insert(args.end(), {"--progress-bar", "--max-filesize", std::to_string(file.packed_size - have),
                             "--range", std::format("{}-{}", file.data_offset + have, last), "--output",
                             core::to_utf8(chunk), file.url});
    const double done = static_cast<double>(have) / static_cast<double>(file.packed_size);
    auto fetched = run_tool(
        {.program = curl, .args = std::move(args), .timeout = std::chrono::hours(2)},
        [&](std::string_view line) {
          if (const auto f = curl_percent(line); f && progress.report)
            progress.report(static_cast<float>(done + ((1.0 - done) * *f)) * 0.85F, file.name);
        },
        progress.stop, i18n::tr_format("downloading {}", file.name));
    // Ce qui est arrivé, même coupé, s'ajoute au .part ; un morceau plus long que la plage demandée n'est pas
    // la plage (réponse 200 entière) : jeté (A06-07).
    if (!append_chunk(chunk, packed, file.packed_size - have))
      fetched = refuse(RefusalCode::HashMismatch,
                       i18n::tr_format("{}: the server ignored the byte range", file.name));
    std::filesystem::remove(chunk, error);
    have = size_or_zero(packed, error);
    if (fetched) break;
    failure = fetched.error();
    if (failure->code == RefusalCode::Cancelled) return std::unexpected(*failure);
  }
  if (have < file.packed_size && failure) return std::unexpected(*failure);
  return {};
}

std::expected<void, Refusal> fetch_vendor_file(const core::VendorFile& file,
                                               const std::filesystem::path& dest,
                                               const std::filesystem::path& scratch,
                                               const InstallProgress& progress) {
  const auto curl = system_tool("curl");
  if (!curl) return refuse(RefusalCode::LibraryUnavailable, i18n::tr("curl not found on this system"));
  std::error_code error;
  std::filesystem::create_directories(scratch, error);
  // Nommés par l'empreinte du fichier (A06-04) : un .part d'une autre version au même nom ne se reprend pas.
  const std::filesystem::path packed = scratch / (file.sha256 + ".part");
  const std::filesystem::path chunk = scratch / (file.sha256 + ".morceau");
  const auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove(packed, ignored);
    std::filesystem::remove(chunk, ignored);
  };
  if (file.packed_size == 0) return refuse(RefusalCode::FieldOutOfRange, file.name + i18n::tr(": zero size"));
  if (auto pulled = pull_packed(file, *curl, packed, chunk, progress); !pulled) return pulled;
  const std::uintmax_t got = std::filesystem::file_size(packed, error);
  if (error || got != file.packed_size) {
    cleanup();
    return refuse(RefusalCode::HashMismatch, i18n::tr_format("{}: {} bytes received, {} expected", file.name,
                                                             error ? 0 : got, file.packed_size));
  }
  if (progress.report) progress.report(0.9F, file.name);  // tiré : la décompression commence
  std::filesystem::path partial = dest;
  partial += ".partiel";
  std::ifstream in(packed, std::ios::binary);
  std::ofstream out(partial, std::ios::binary | std::ios::trunc);
  const auto close = [&] {
    in.close();
    out.close();
  };
  if (!in || !out) {
    close();
    cleanup();
    return refuse(RefusalCode::FileUnwritable, partial.string());
  }
  core::Sha256 hash;
  std::uint64_t produced = 0;
  bool write_failed = false;
  bool overflow = false;
  const auto sink = [&](std::span<const std::uint8_t> bytes) {
    if (progress.stop.stop_requested()) return false;
    // Jamais plus que la taille de l'index : un flux deflate forgé peut produire mille fois sa taille
    // (A06-05).
    if (produced + bytes.size() > file.size) {
      overflow = true;
      return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
      write_failed = true;
      return false;
    }
    hash.update(std::as_bytes(bytes));
    produced += bytes.size();
    return true;
  };
  std::expected<void, Refusal> result;
  if (file.deflated) {
    auto inflated = core::inflate_raw(
        [&](std::span<std::uint8_t> buffer) {
          in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
          return static_cast<std::size_t>(in.gcount());
        },
        sink);
    if (!inflated) result = std::unexpected(inflated.error());
  } else {
    std::array<std::uint8_t, 1U << 16> buffer{};
    while (in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size())) ||
           in.gcount() > 0) {
      if (!sink(std::span<const std::uint8_t>(buffer).first(static_cast<std::size_t>(in.gcount())))) break;
    }
  }
  const bool flushed = static_cast<bool>(out.flush());
  close();
  cleanup();
  const auto discard = [&] {
    std::error_code ignored;
    std::filesystem::remove(partial, ignored);
  };
  if (progress.stop.stop_requested()) {
    discard();
    return refuse(RefusalCode::Cancelled, file.name);
  }
  if (write_failed || !flushed) {
    discard();
    return refuse(RefusalCode::FileUnwritable, partial.string());
  }
  if (overflow) {
    discard();
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : plus de {} octets décompressés, refusé", file.name, file.size));
  }
  if (!result) {
    discard();
    return std::unexpected(result.error());
  }
  const std::string sha = core::to_hex(hash.finish());
  if (produced != file.size || sha != file.sha256) {
    discard();
    return refuse(RefusalCode::HashMismatch,
                  std::format("{} : {} octets, SHA-256 {} ; {} octets et {} attendus", file.name, produced,
                              sha, file.size, file.sha256));
  }
  std::filesystem::rename(partial, dest, error);
  if (error) {
    discard();
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", dest.string(), error.message()));
  }
  if (progress.report) progress.report(1.0F, file.name);
  return {};
}

namespace {

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
                                          const core::EnginePackage& package) {
  if (auto synced = sync_tree(staging); !synced) return synced;
  const std::filesystem::path path = staging / std::string(kPackageNote);
  std::ofstream note(path, std::ios::trunc);
  note << "sha256=" << package.sha256 << "\nlicence=" << package.license << '\n';
  note.close();
  if (!note || core::sync_file(path))
    return refuse(RefusalCode::FileUnwritable, i18n::tr_format("{}: package note", core::to_utf8(staging)));
  core::sync_directory(staging);
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
  const InstallSpace space = install_space(catalog, package, progress.cuda_arch, directory_bytes(target_dir));
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
               std::vector<std::string> args = curl_download_args();
               args.insert(args.end(),
                           {"--continue-at", "-", "--progress-bar", "--output", part.string(), package.url});
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
                          .args = {"-xzf", part.string(), "-C", staging.string()},
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
  // Chaque fichier sur le disque, puis la note en dernier (A36-03) : après une coupure, un dossier de version
  // (ou un « .partiel » qui a sa note, recover_interrupted_install) est entier, jamais fait de fichiers
  // vides.
  if (const auto sealed = seal_staging(staging, package); !sealed) {
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
    std::filesystem::rename(target, aside, error);
    if (error) {
      const std::string reason = error.message();
      std::filesystem::remove_all(staging, error);
      return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), reason));
    }
  }
  error.clear();
  std::filesystem::rename(staging, target, error);
  if (error) {
    const std::string reason = error.message();
    std::error_code restore;
    if (replaced) std::filesystem::rename(aside, target, restore);
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), reason));
  }
  core::sync_directory(target.parent_path());
  if (replaced) std::filesystem::remove_all(aside, error);
  error.clear();
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
