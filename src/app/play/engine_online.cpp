#include "app/play/engine_online.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iterator>
#include <stop_token>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

namespace {

// Les fichiers de NVIDIA ne viennent que de là, comme dans l'engines.lock livré.
constexpr std::string_view kVendorPrefix = "https://pypi.nvidia.com/";
// Relu au plus une fois par jour : un paquet joint après coup paraît au lancement du lendemain au plus tard,
// ou tout de suite après un lancement sans copie gardée.
constexpr auto kFreshFor = std::chrono::hours(24);
constexpr int kFetchSeconds = 8;

std::string read_text(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool fresh(const std::filesystem::path& file) {
  std::error_code error;
  const auto written = std::filesystem::last_write_time(file, error);
  if (error) return false;
  const auto age = std::filesystem::file_time_type::clock::now() - written;
  return age >= std::filesystem::file_time_type::duration::zero() && age < kFreshFor;
}

}  // namespace

std::string release_download_prefix(std::string_view repo, std::string_view version) {
  return std::format("https://github.com/{}/releases/download/v{}/", repo, version);
}

int merge_release_index(EngineCatalog& catalog, std::string_view text, std::string_view prefix) {
  auto parsed = core::read_engine_index(text);
  if (!parsed) {
    catalog.problem += std::format("{}engines-nvidia.lock : {}", catalog.problem.empty() ? "" : " ; ",
                                   parsed.error().detail);
    return 0;
  }
  int taken = 0;
  for (core::EnginePackage& package : parsed->packages) {
    if (package.platform != catalog.layout.os) continue;
    if (!package.url.starts_with(prefix)) {
      catalog.problem += std::format("{}engines-nvidia.lock : {} hors de la release, écarté",
                                     catalog.problem.empty() ? "" : " ; ", package.url);
      continue;
    }
    const core::EngineId engine = package.engine;
    const auto mine = [&](const core::VendorFile& file) { return file.engine == engine; };
    // Les fichiers NVIDIA du même moteur, tous de pypi.nvidia.com, ou aucun : un index à moitié cru
    // installerait un moteur sans ses bibliothèques.
    const bool vendor_ok = std::ranges::all_of(parsed->vendor_files, [&](const core::VendorFile& file) {
      return file.platform != catalog.layout.os || !mine(file) || file.url.starts_with(kVendorPrefix);
    });
    if (!vendor_ok) {
      catalog.problem += std::format("{}engines-nvidia.lock : fichiers NVIDIA hors de {}, écarté",
                                     catalog.problem.empty() ? "" : " ; ", kVendorPrefix);
      continue;
    }
    std::erase_if(catalog.packages, [&](const core::EnginePackage& p) { return p.engine == engine; });
    std::erase_if(catalog.vendor_files, mine);
    for (const core::VendorFile& file : parsed->vendor_files) {
      if (file.platform == catalog.layout.os && mine(file)) catalog.vendor_files.push_back(file);
    }
    catalog.packages.push_back(std::move(package));
    ++taken;
  }
  return taken;
}

void add_release_engines(EngineCatalog& catalog, const std::filesystem::path& cache, std::string_view repo,
                         std::string_view version, const UpdateFetch& fetch) {
  if (repo.empty() || version.empty()) return;
  const std::string prefix = release_download_prefix(repo, version);
  if (!fresh(cache) && fetch) {
    std::error_code error;
    std::filesystem::create_directories(cache.parent_path(), error);
    std::filesystem::path part = cache;
    part += ".part";
    std::filesystem::remove(part, error);
    if (auto got = fetch(prefix + "engines-nvidia.lock", part, kFetchSeconds, std::stop_token{}); got) {
      std::filesystem::rename(part, cache, error);
      if (error) log_channel("app")->info("moteurs de la release : {}", error.message());
    } else {
      std::filesystem::remove(part, error);
      // 404 : la release n'a pas de paquet NVIDIA ; une copie gardée d'avant ne vaut plus.
      if (got.error().code == RefusalCode::TrackUnavailable) std::filesystem::remove(cache, error);
      log_channel("app")->info("moteurs de la release : {}", got.error().detail);
    }
  }
  const std::string text = read_text(cache);
  if (text.empty()) return;
  if (const int taken = merge_release_index(catalog, text, prefix); taken > 0)
    log_channel("app")->info("moteurs de la release : {} paquet(s) de {}", taken, prefix);
}

}  // namespace ajp::app::play
