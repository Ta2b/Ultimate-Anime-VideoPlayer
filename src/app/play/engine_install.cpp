#include "app/play/engine_install.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

#include "app/play/engine_tools.h"
#include "core/inflate.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool any_exists(std::initializer_list<std::string_view> paths) {
  std::error_code error;
  for (const std::string_view p : paths) {
    if (std::filesystem::exists(std::filesystem::path(p), error)) return true;
  }
  return false;
}

// MIGraphX de Windows ML demande Windows 11 24H2 (build 26100) ; le pilote AMD, Windows ML le
// vérifie lui-même (winml_library.h).
constexpr unsigned long kWinmlBuild = 26100;

#ifdef _WIN32
// Le build de Windows par RtlGetVersion (GetVersionEx rend celui du manifeste) ; 0 s'il échappe.
unsigned long windows_build() {
  using GetVersion = LONG(WINAPI*)(OSVERSIONINFOW*);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  if (ntdll == nullptr) return 0;
  const auto get =
      reinterpret_cast<GetVersion>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
  OSVERSIONINFOW info{};
  info.dwOSVersionInfoSize = sizeof(info);
  if (get == nullptr || get(&info) != 0) return 0;
  return info.dwBuildNumber;
}
#else
unsigned long windows_build() {
  return 0;
}
#endif

std::optional<std::string> windows_prerequisite(core::EngineId engine) {
  if (engine != core::EngineId::MiGraphX) return std::nullopt;  // les pilotes fournissent le reste
  const unsigned long build = windows_build();
  if (build == 0 || build >= kWinmlBuild) return std::nullopt;
  return std::format("Windows 11 24H2 requis (build {} installé) : mettre Windows à jour par Windows Update",
                     build);
}

}  // namespace

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
    catalog.problem = std::format("{} absent : seuls les moteurs livrés sont proposés", index.string());
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
  for (const Refusal& skipped : parsed->skipped) add_problem(std::format("écarté, {}", skipped.detail));
  if (catalog.layout.os == core::Os::Linux) {
    if (auto maxine = core::parse_maxine_source(text)) {
      catalog.maxine = std::move(*maxine);
    } else {
      add_problem(maxine.error().detail);
    }
  }
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
  std::ifstream in(core::engine_dir(catalog.layout, package) / std::string(kPackageNote));
  if (!in) return std::nullopt;
  InstalledPackage found;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("sha256=")) found.sha256 = line.substr(7);
    if (line.starts_with("licence=")) found.license = line.substr(8);
  }
  if (found.sha256.empty()) return std::nullopt;
  return found;
}

std::vector<std::filesystem::path> installed_backends(const EngineCatalog& catalog) {
  std::vector<std::filesystem::path> out;
  for (const core::EngineId id :
       {core::EngineId::TensorRt, core::EngineId::MiGraphX, core::EngineId::OpenVino}) {
    const core::EnginePackage* package = package_for(catalog, id);
    if (package != nullptr && installed(catalog, *package))
      out.push_back(core::backend_library(catalog.layout, *package));
  }
  return out;
}

std::optional<std::string> missing_prerequisite(core::EngineId engine) {
  if (host_os() == core::Os::Windows) return windows_prerequisite(engine);
  if (host_os() != core::Os::Linux) return std::nullopt;
  switch (engine) {
    case core::EngineId::MiGraphX:
      // Le module est lié au runtime HIP de ROCm 7 (libamdhip64.so.7) : un ROCm 6 ne le charge pas.
      if (!any_exists({"/opt/rocm/lib/libmigraphx_c.so.3", "/opt/rocm/lib/libmigraphx_c.so"}))
        return "ROCm absent : installer « migraphx » depuis le dépôt ROCm 7 d'AMD (sudo apt install "
               "migraphx)";
      if (!any_exists({"/opt/rocm/lib/libamdhip64.so.7"}))
        return "ROCm 7 requis (ROCm 6 installé) : passer au dépôt ROCm 7 d'AMD, puis sudo apt install "
               "migraphx";
      return std::nullopt;
    case core::EngineId::OpenVino:
      if (any_exists({"/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1",
                      "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so",
                      "/usr/lib64/libze_intel_gpu.so.1", "/usr/lib/libze_intel_gpu.so.1"})) {
        return std::nullopt;
      }
      return "pilote compute Intel absent (sudo apt install intel-opencl-icd libze-intel-gpu1)";
    default:
      return std::nullopt;
  }
}

std::string cuda_arch_from_name(std::string_view gpu_name) {
  std::string name(gpu_name);
  std::ranges::transform(name, name.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const auto has = [&](std::string_view part) { return name.find(part) != std::string::npos; };
  const auto series = [&](char digit) {  // « rtx 4050 » : la série à deux chiffres, ici 40
    const std::size_t at = name.find("rtx ");
    return at != std::string::npos && at + 5 < name.size() && name[at + 4] == digit && name[at + 5] >= '0' &&
           name[at + 5] <= '9';
  };
  if (has("h100") || has("h200")) return "sm90";
  if (has("a100")) return "sm80";
  if (has("blackwell") || series('5')) return "sm120";
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

std::string detect_cuda_arch() {
  const auto smi = system_tool("nvidia-smi");
  if (!smi) return {};
  std::string arch;
  auto done = run_tool(
      {.program = *smi,
       .args = {"--query-gpu=compute_cap", "--format=csv,noheader"},
       .timeout = std::chrono::seconds(10)},
      [&](std::string_view line) {
        // « 8.9 » : la première carte, la carte de rendu.
        if (!arch.empty()) return;
        const std::size_t dot = line.find('.');
        if (dot == std::string_view::npos || dot == 0 || dot + 1 >= line.size()) return;
        const auto digit = [](char c) { return c >= '0' && c <= '9'; };
        if (!digit(line[dot - 1]) || !digit(line[dot + 1])) return;
        arch = std::format("sm{}{}", line[dot - 1], line[dot + 1]);
        // sm100 et plus : « 10.0 » donne sm100 ; la majeure a deux chiffres.
        if (dot >= 2 && digit(line[dot - 2]))
          arch = std::format("sm{}{}{}", line[dot - 2], line[dot - 1], line[dot + 1]);
      },
      {}, "nvidia-smi");
  return done ? arch : std::string();
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
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(
           dir, std::filesystem::directory_options::skip_permission_denied, error)) {
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
  return refuse(RefusalCode::InsufficientSpace,
                std::format("espace disque insuffisant : l'installation demande {} (dont {} à télécharger, "
                            "{} une fois installé), il reste {} sur {}",
                            format_size(space.peak), format_size(space.download),
                            format_size(space.installed), format_size(available), where.string()));
}

std::expected<void, Refusal> fetch_vendor_file(const core::VendorFile& file,
                                               const std::filesystem::path& dest,
                                               const std::filesystem::path& scratch,
                                               const InstallProgress& progress) {
  const auto curl = system_tool("curl");
  if (!curl) return refuse(RefusalCode::LibraryUnavailable, "curl introuvable sur ce système");
  std::error_code error;
  std::filesystem::create_directories(scratch, error);
  const std::filesystem::path packed = scratch / (file.name + ".part");
  std::filesystem::remove(packed, error);
  const auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove(packed, ignored);
  };
  if (file.packed_size == 0) return refuse(RefusalCode::FieldOutOfRange, file.name + " : taille nulle");
  const std::uint64_t last = file.data_offset + file.packed_size - 1;
  auto fetched = run_tool(
      {.program = *curl,
       .args = {"--fail", "--location", "--retry", "1", "--retry-delay", "2", "--progress-bar", "--range",
                std::format("{}-{}", file.data_offset, last), "--output", packed.string(), file.url},
       .timeout = std::chrono::hours(2)},
      [&](std::string_view line) {
        if (const auto f = curl_percent(line); f && progress.report) progress.report(*f * 0.85F, file.name);
      },
      progress.stop, std::format("téléchargement de {}", file.name));
  if (!fetched) {
    cleanup();
    return std::unexpected(fetched.error());
  }
  const std::uintmax_t got = std::filesystem::file_size(packed, error);
  if (error || got != file.packed_size) {
    cleanup();
    return refuse(RefusalCode::HashMismatch, std::format("{} : {} octets reçus, {} attendus", file.name,
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
  const auto sink = [&](std::span<const std::uint8_t> bytes) {
    if (progress.stop.stop_requested()) return false;
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
std::expected<std::filesystem::path, Refusal> install_steps(const EngineCatalog& catalog,
                                                            const core::EnginePackage& package,
                                                            const InstallProgress& progress,
                                                            std::string& step) {
  const auto curl = system_tool("curl");
  const auto tar = system_tool("tar");
  if (!curl || !tar)
    return refuse(RefusalCode::LibraryUnavailable, "curl ou tar introuvables sur ce système");
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
  step = "vérification de la place";
  // La place d'abord : un refus net, sans téléchargement, plutôt qu'un disque qui se remplit puis une
  // installation recommencée.
  const std::filesystem::path target_dir = core::engine_dir(catalog.layout, package);
  const InstallSpace space = install_space(catalog, package, progress.cuda_arch, directory_bytes(target_dir));
  log_channel("app")->info(
      "installation de {} : architecture {}, à tirer {}, installé {}, pic {}, libre {} ({})", name,
      progress.cuda_arch.empty() ? "inconnue (tous les fichiers)" : progress.cuda_arch,
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
    step = "téléchargement de l'archive du moteur";
    overall(0.0, std::format("téléchargement de {} ({})", name, format_size(package.size)));
    auto fetched = run_tool(
        {.program = *curl,
         .args = {"--fail", "--location", "--retry", "1", "--retry-delay", "2", "--continue-at", "-",
                  "--progress-bar", "--output", part.string(), package.url},
         .timeout = std::chrono::hours(4)},
        [&](std::string_view line) {
          if (const auto f = curl_percent(line))
            overall(static_cast<double>(*f) * static_cast<double>(package.size),
                    std::format("téléchargement de {}", name));
        },
        progress.stop, std::format("téléchargement de {}", name));
    if (!fetched) return std::unexpected(fetched.error());
    step = "vérification de l'archive";
    overall(static_cast<double>(package.size), std::format("vérification de {}", name));
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
  step = "extraction de l'archive";
  done_bytes = static_cast<double>(package.size);
  overall(0.0, std::format("extraction de {}", name));
  if (auto ok = run_tool({.program = *tar,
                          .args = {"-xzf", part.string(), "-C", staging.string()},
                          .timeout = std::chrono::minutes(30)},
                         {}, progress.stop, std::format("extraction de {}", name));
      !ok) {
    std::filesystem::remove_all(staging, error);
    return std::unexpected(ok.error());
  }
  // L'archive du moteur n'est plus utile une fois extraite : sa place revient avant les gros téléchargements.
  std::filesystem::remove(part, error);
  // RTX HDR n'a pas de module : son paquet s'importe ensuite dans le magasin des SDK.
  const bool module = package.engine != core::EngineId::MaxineVfx;
  const std::filesystem::path library = staging / core::backend_library(catalog.layout, package).filename();
  if (module && !std::filesystem::is_regular_file(library, error)) {
    std::filesystem::remove_all(staging, error);
    return refuse(RefusalCode::HashMismatch,
                  std::format("paquet {} sans {}", name, library.filename().string()));
  }
  // Les fichiers que NVIDIA publie, tirés de ses archives et joints au module : ce qui est à nous
  // vient de notre release, le reste du lien direct (core::VendorFile).
  for (std::size_t i = 0; i < vendor.size(); ++i) {
    const core::VendorFile& file = vendor[i];
    step = std::format("fichier {} ({}/{})", file.name, i + 1, vendor.size());
    const std::string label = std::format("{} ({}/{})", file.name, i + 1, vendor.size());
    overall(0.0, "téléchargement de " + label);
    log_channel("app")->info("installation de {} : {} ({} tirés, {} écrits)", name, step,
                             format_size(file.packed_size), format_size(file.size));
    InstallProgress local = progress;
    local.report = [&, label](float fraction, const std::string&) {
      overall(static_cast<double>(fraction) * static_cast<double>(file.packed_size),
              (fraction >= 0.9F ? "décompression de " : "téléchargement de ") + label);
    };
    if (auto ok = fetch_vendor_file(file, staging / file.name, catalog.layout.downloads, local); !ok) {
      std::filesystem::remove_all(staging, error);
      return std::unexpected(ok.error());
    }
    done_bytes += static_cast<double>(file.packed_size);
  }
  step = "remplacement de la version installée";
  {
    std::ofstream note(staging / std::string(kPackageNote), std::ios::trunc);
    note << "sha256=" << package.sha256 << "\nlicence=" << package.license << '\n';
    if (!note.flush()) {
      std::filesystem::remove_all(staging, error);
      return refuse(RefusalCode::FileUnwritable, std::format("{} : note du paquet", staging.string()));
    }
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
  if (replaced) std::filesystem::remove_all(aside, error);
  error.clear();
  report(1.0F, std::format("{} installé", name));
  return module ? core::backend_library(catalog.layout, package) : target;
}

}  // namespace

std::expected<std::filesystem::path, Refusal> install_package(const EngineCatalog& catalog,
                                                              const core::EnginePackage& package,
                                                              const InstallProgress& progress) {
  const auto log = log_channel("app");
  const std::string name(core::engine_name(package.engine));
  // Les restes d'un essai raté ou interrompu partent avant tout nouvel essai : ils ne s'accumulent pas.
  std::error_code error;
  for (const char* suffix : {".partiel", ".ancien"}) {
    std::filesystem::path leftover = core::engine_dir(catalog.layout, package);
    leftover += suffix;
    if (std::filesystem::exists(leftover, error)) {
      log->info("installation de {} : reste d'un essai précédent retiré ({})", name, leftover.string());
      std::filesystem::remove_all(leftover, error);
    }
  }
  std::string step = "démarrage";
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
