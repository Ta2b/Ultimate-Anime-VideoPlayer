#include "app/play/engine_install.h"

#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <system_error>

#include "prepare/process.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Un exécutable du système à un emplacement connu : le lanceur ne cherche pas dans PATH.
std::optional<std::filesystem::path> system_tool(std::string_view name) {
#if defined(_WIN32)
  const std::array<std::filesystem::path, 1> dirs{"C:/Windows/System32"};
  const std::string file = std::string(name) + ".exe";
#else
  const std::array<std::filesystem::path, 4> dirs{"/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin"};
  const std::string file(name);
#endif
  for (const std::filesystem::path& dir : dirs) {
    std::error_code error;
    if (std::filesystem::is_regular_file(dir / file, error)) return dir / file;
  }
  return std::nullopt;
}

bool any_exists(std::initializer_list<std::string_view> paths) {
  std::error_code error;
  for (const std::string_view p : paths) {
    if (std::filesystem::exists(std::filesystem::path(p), error)) return true;
  }
  return false;
}

// La dernière fraction « 43.2% » d'une ligne de la barre de progression de curl.
std::optional<float> curl_percent(std::string_view line) {
  const std::size_t percent = line.rfind('%');
  if (percent == std::string_view::npos) return std::nullopt;
  std::size_t start = percent;
  while (start > 0 &&
         (std::isdigit(static_cast<unsigned char>(line[start - 1])) != 0 || line[start - 1] == '.'))
    --start;
  float value = 0;
  const auto [end, error] = std::from_chars(line.data() + start, line.data() + percent, value);
  if (error != std::errc{} || end != line.data() + percent) return std::nullopt;
  return value / 100.0F;
}

std::expected<void, Refusal> run_tool(const std::filesystem::path& program, std::vector<std::string> args,
                                      std::chrono::milliseconds timeout, const prepare::LineSink& on_line,
                                      const std::stop_token& stop, std::string_view what) {
  std::string last;
  const auto sink = [&](std::string_view line) {
    last = line;
    if (on_line) on_line(line);
  };
  auto exit =
      prepare::run_process({.program = program, .args = std::move(args), .timeout = timeout}, sink, stop);
  if (!exit) return std::unexpected(exit.error());
  if (exit->kind == prepare::ExitKind::Cancelled)
    return refuse(RefusalCode::Cancelled, std::format("{} annulé", what));
  if (exit->kind != prepare::ExitKind::Exited || exit->code != 0) {
    return refuse(RefusalCode::DecodeFailed,
                  std::format("{} : {} ({})", what, prepare::describe(*exit), last));
  }
  return {};
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
      if (any_exists({"/opt/rocm/lib/libmigraphx_c.so.3", "/opt/rocm/lib/libmigraphx_c.so"}))
        return std::nullopt;
      return "ROCm absent : installer « migraphx » depuis le dépôt ROCm d'AMD (sudo apt install migraphx)";
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
        *curl,
        {"--fail", "--location", "--retry", "3", "--continue-at", "-", "--progress-bar", "--output",
         part.string(), package.url},
        std::chrono::hours(4),
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
  if (auto ok = run_tool(*tar, {"-xzf", part.string(), "-C", staging.string()}, std::chrono::minutes(30), {},
                         progress.stop, std::format("extraction de {}", name));
      !ok) {
    std::filesystem::remove_all(staging, error);
    return std::unexpected(ok.error());
  }
  const std::filesystem::path library = staging / core::backend_library(catalog.layout, package).filename();
  if (!std::filesystem::is_regular_file(library, error)) {
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
  return core::backend_library(catalog.layout, package);
}

}  // namespace ajp::app::play
