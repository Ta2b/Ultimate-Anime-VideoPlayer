#include "app/play/engine_install.h"

#include <chrono>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <system_error>

#include "app/play/engine_tools.h"

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

}  // namespace

core::Os host_os() {
#if defined(_WIN32)
  return core::Os::Windows;
#elif defined(__APPLE__)
  return core::Os::MacOs;
#else
  return core::Os::Linux;
#endif
}

EngineCatalog load_catalog(const std::filesystem::path& index, const std::filesystem::path& data_dir) {
  EngineCatalog catalog{.layout = core::install_layout(host_os(), data_dir)};
  std::ifstream in(index, std::ios::binary);
  if (!in) {
    catalog.problem = std::format("{} absent : seuls les moteurs livrés sont proposés", index.string());
    return catalog;
  }
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  auto parsed = core::parse_engine_index(text);
  if (!parsed) {
    catalog.problem = std::format("{} : {}", index.string(), parsed.error().detail);
    return catalog;
  }
  for (core::EnginePackage& package : *parsed) {
    if (package.platform == catalog.layout.os) catalog.packages.push_back(std::move(package));
  }
  if (catalog.layout.os == core::Os::Linux) {
    if (auto maxine = core::parse_maxine_source(text)) {
      catalog.maxine = std::move(*maxine);
    } else {
      catalog.problem = std::format("{} : {}", index.string(), maxine.error().detail);
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
  if (host_os() != core::Os::Linux) return std::nullopt;  // les pilotes Windows les fournissent
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
  if (!core::verify_file(part, package.size, package.sha256)) {
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
    if (auto ok = core::verify_file(part, package.size, package.sha256); !ok) {
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
  std::filesystem::remove_all(target, error);
  std::filesystem::rename(staging, target, error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), error.message()));
  std::filesystem::remove(part, error);
  report(1.0F, std::format("{} installé", name));
  return module ? core::backend_library(catalog.layout, package) : target;
}

}  // namespace ajp::app::play
