#include "app/play/engine_online.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iterator>
#include <stop_token>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/ed25519.h"
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

std::string release_repo_prefix(std::string_view repo) {
  return std::format("https://github.com/{}/releases/download/", repo);
}

bool release_index_signed(std::string_view text, std::string_view signature,
                          const std::vector<std::string>& keys) {
  std::string_view sig = signature;
  while (!sig.empty() && (sig.back() == '\n' || sig.back() == '\r' || sig.back() == ' '))
    sig.remove_suffix(1);
  if (sig.empty()) return false;
  return std::ranges::any_of(keys,
                             [&](const std::string& key) { return ed25519_verify_hex(key, text, sig); });
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
                         std::string_view version, const std::vector<std::string>& keys,
                         const UpdateFetch& fetch) {
  if (repo.empty() || version.empty()) return;
  const std::string url = release_download_prefix(repo, version) + "engines-nvidia.lock";
  std::filesystem::path signature = cache;
  signature += ".sig";
  const auto log = log_channel("app");
  if (!fresh(cache) && fetch) {
    std::error_code error;
    std::filesystem::create_directories(cache.parent_path(), error);
    std::filesystem::path part = cache;
    part += ".part";
    std::filesystem::path sig_part = signature;
    sig_part += ".part";
    std::filesystem::remove(part, error);
    std::filesystem::remove(sig_part, error);
    auto got = fetch(url, part, kFetchSeconds, std::stop_token{});
    if (got) got = fetch(url + ".sig", sig_part, kFetchSeconds, std::stop_token{});
    if (got) {
      // La signature d'abord : une copie gardée a toujours la sienne à côté.
      std::filesystem::rename(sig_part, signature, error);
      if (!error) std::filesystem::rename(part, cache, error);
      if (error) log->info("moteurs de la release : {}", error.message());
    } else {
      std::filesystem::remove(part, error);
      std::filesystem::remove(sig_part, error);
      // 404 : la release n'a pas de paquet NVIDIA (ou pas de signature) ; une copie gardée d'avant ne vaut
      // plus.
      if (got.error().code == RefusalCode::TrackUnavailable) {
        std::filesystem::remove(cache, error);
        std::filesystem::remove(signature, error);
      }
      log->info("moteurs de la release : {}", got.error().detail);
    }
  }
  const std::string text = read_text(cache);
  if (text.empty()) return;
  // Signé en CI par la clé des mises à jour (A26-02) : sans signature valide, rien n'est cru, même la copie
  // gardée (un fichier du dossier de données changé à la main ne pointe pas ailleurs).
  if (!release_index_signed(text, read_text(signature), keys)) {
    catalog.problem += std::format("{}engines-nvidia.lock : signature absente ou fausse, écarté",
                                   catalog.problem.empty() ? "" : " ; ");
    return;
  }
  const std::string prefix = release_repo_prefix(repo);
  if (const int taken = merge_release_index(catalog, text, prefix); taken > 0)
    log->info("moteurs de la release : {} paquet(s) de {}", taken, prefix);
}

}  // namespace ajp::app::play
