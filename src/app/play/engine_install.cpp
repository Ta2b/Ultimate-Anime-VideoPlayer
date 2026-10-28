#include "app/play/engine_install.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

#include "app/play/engine_tools.h"
#include "core/inflate.h"

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

EngineCatalog load_catalog(const std::filesystem::path& index, const std::filesystem::path& data_dir) {
  EngineCatalog catalog{.layout = core::install_layout(host_os(), data_dir)};
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
       .args = {"--fail", "--location", "--retry", "3", "--progress-bar", "--range",
                std::format("{}-{}", file.data_offset, last), "--output", packed.string(), file.url},
       .timeout = std::chrono::hours(2)},
      [&](std::string_view line) {
        if (const auto f = curl_percent(line); f && progress.report) progress.report(*f * 0.7F, file.name);
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
  return {};
}

std::expected<std::filesystem::path, Refusal> install_package(const EngineCatalog& catalog,
                                                              const core::EnginePackage& package,
                                                              const InstallProgress& progress) {
  const auto curl = system_tool("curl");
  const auto tar = system_tool("tar");
  if (!curl || !tar)
    return refuse(RefusalCode::LibraryUnavailable, "curl ou tar introuvables sur ce système");
  const std::string name(core::engine_name(package.engine));
  const auto report = [&](float fraction, const std::string& stage) {
    if (progress.report) progress.report(fraction, stage);
  };
  std::error_code error;
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
    report(0.0F, std::format("téléchargement de {} ({} Mo)", name, package.size >> 20));
    auto fetched = run_tool(
        {.program = *curl,
         .args = {"--fail", "--location", "--retry", "3", "--continue-at", "-", "--progress-bar", "--output",
                  part.string(), package.url},
         .timeout = std::chrono::hours(4)},
        [&](std::string_view line) {
          if (const auto f = curl_percent(line)) report(*f * 0.9F, std::format("téléchargement de {}", name));
        },
        progress.stop, std::format("téléchargement de {}", name));
    if (!fetched) return std::unexpected(fetched.error());
    report(0.9F, std::format("vérification de {}", name));
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
  report(0.93F, std::format("installation de {}", name));
  if (auto ok = run_tool({.program = *tar,
                          .args = {"-xzf", part.string(), "-C", staging.string()},
                          .timeout = std::chrono::minutes(30)},
                         {}, progress.stop, std::format("extraction de {}", name));
      !ok) {
    std::filesystem::remove_all(staging, error);
    return std::unexpected(ok.error());
  }
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
  const std::vector<core::VendorFile> vendor =
      core::vendor_files_for(catalog.vendor_files, package.engine, catalog.layout.os, progress.cuda_arch);
  for (std::size_t i = 0; i < vendor.size(); ++i) {
    report(0.93F + 0.06F * static_cast<float>(i) / static_cast<float>(vendor.size()),
           std::format("téléchargement de {} ({}/{})", vendor[i].name, i + 1, vendor.size()));
    if (auto ok = fetch_vendor_file(vendor[i], staging / vendor[i].name, catalog.layout.downloads, progress);
        !ok) {
      std::filesystem::remove_all(staging, error);
      return std::unexpected(ok.error());
    }
  }
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
  std::filesystem::remove(part, error);
  report(1.0F, std::format("{} installé", name));
  return module ? core::backend_library(catalog.layout, package) : target;
}

}  // namespace ajp::app::play
