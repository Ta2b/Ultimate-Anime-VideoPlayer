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
#include "core/toml_fields.h"
#include "core/utf8_path.h"
#include "prepare/cache_io.h"
#include "prepare/key_lock.h"

namespace kirari::app::play {

namespace {

// Les fichiers de NVIDIA ne viennent que de là, comme dans l'engines.lock livré.
constexpr std::string_view kVendorPrefix = "https://pypi.nvidia.com/";
// Relu au plus une fois par jour : une copie gardée de cette version, fraîche, ne se redemande pas.
constexpr auto kFreshFor = std::chrono::hours(24);
// Un 404 se retient moins : le paquet TensorRT est joint à la release après coup, quelques minutes ou heures
// après sa publication, et la requête de 8 s au plus ne coûte presque rien (C06-04).
constexpr auto kAbsentFor = std::chrono::hours(2);
// Un échec qui n'est pas un 404 (hors ligne, portail captif, délai) : un nouvel essai quelques heures après.
constexpr auto kRetryAfter = std::chrono::hours(6);
// Un seul budget pour les deux requêtes (l'index puis sa signature), pas 8 s chacune plus la marge du
// processus (C26-08).
constexpr int kFetchSeconds = 16;
constexpr std::string_view kSignedCopy = "kirari-signed-engine-index-v1\n";

std::string read_text(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool fresh(const std::filesystem::path& file, std::filesystem::file_time_type::duration within = kFreshFor) {
  std::error_code error;
  const auto written = std::filesystem::last_write_time(file, error);
  if (error) return false;
  const auto age = std::filesystem::file_time_type::clock::now() - written;
  return age >= std::filesystem::file_time_type::duration::zero() && age < within;
}

std::string_view signature_payload(std::string_view sig) {
  while (!sig.empty() && (sig.back() == '\n' || sig.back() == '\r' || sig.back() == ' '))
    sig.remove_suffix(1);
  return sig;
}

}  // namespace

std::string release_index_tag(std::string_view text) {
  auto root = core::fields::parse(text, RefusalCode::FileUnreadable, "engines-nvidia.lock");
  return root ? (*root)["tag"].value_or(std::string()) : std::string();
}

std::string release_download_prefix(std::string_view repo, std::string_view version) {
  return std::format("https://github.com/{}/releases/download/v{}/", repo, version);
}

std::string release_repo_prefix(std::string_view repo) {
  return std::format("https://github.com/{}/releases/download/", repo);
}

bool release_index_signed(std::string_view text, std::string_view signature,
                          const std::vector<std::string>& keys) {
  const auto sig = signature_payload(signature);
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
    if (core::vendor_file_skipped(*parsed, engine, catalog.layout.os)) {
      catalog.problem += std::format("{}engines-nvidia.lock : un fichier NVIDIA est écarté, paquet écarté",
                                     catalog.problem.empty() ? "" : " ; ");
      continue;
    }
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

namespace {

std::filesystem::path beside(const std::filesystem::path& cache, std::string_view suffix) {
  std::filesystem::path file = cache;
  file += suffix;
  return file;
}

}  // namespace

namespace {

// La marque (.absent, .tried) dit pour quelle version elle a été posée : celle d'une autre version, ou
// d'avant cette règle (un simple retour à la ligne), ne retient rien (C26-01).
bool marker_for(const std::filesystem::path& file, std::string_view version) {
  std::string text = read_text(file);
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
  return !text.empty() && text == version;
}

void write_marker(const std::filesystem::path& file, std::string_view version) {
  std::ofstream(file, std::ios::binary | std::ios::trunc) << version << '\n';
}

}  // namespace

bool release_index_due(const std::filesystem::path& cache, std::string_view version,
                       const std::vector<std::string>& keys, bool once_per_version) {
  const std::filesystem::path absent = beside(cache, ".absent");
  const std::filesystem::path tried = beside(cache, ".tried");
  std::error_code error;
  // Une copie d'une autre release (le lecteur vient d'être remplacé) ne vaut rien, fût-elle d'hier.
  const ReleaseIndexCopy copy = read_release_index(cache);
  const bool cache_current = release_index_signed(copy.text, copy.signature, keys) &&
                             release_index_tag(copy.text) == std::format("v{}", version);
  const bool absent_current = std::filesystem::exists(absent, error) && marker_for(absent, version);
  const bool tried_current = marker_for(tried, version);
  const auto within = once_per_version ? std::filesystem::file_time_type::duration::max() : kFreshFor;
  // Un 404 se retient moins qu'une copie : le paquet TensorRT est joint à la release après coup, quelques
  // minutes ou heures après sa publication (C06-04).
  if (cache_current && fresh(cache, within)) return false;
  if (absent_current && fresh(absent, kAbsentFor)) return false;
  return !(tried_current && fresh(tried, kRetryAfter));
}

void refresh_release_index(const std::filesystem::path& cache, std::string_view repo,
                           std::string_view version, const std::vector<std::string>& keys,
                           const UpdateFetch& fetch, const std::stop_token& stop) {
  if (repo.empty() || version.empty() || !fetch || stop.stop_requested() ||
      !release_index_due(cache, version, keys))
    return;
  const std::string url = release_download_prefix(repo, version) + "engines-nvidia.lock";
  const std::filesystem::path signature = beside(cache, ".sig");
  // Un 404 se retient quelques heures (A38-10, C06-04), un autre échec (hors ligne, portail captif) un peu
  // plus (A18-02) : les deux marques portent la version du lecteur, et une mise à jour les rend sans effet.
  const std::filesystem::path absent = beside(cache, ".absent");
  const std::filesystem::path tried = beside(cache, ".tried");
  const auto log = log_channel("app");
  std::error_code error;
  const auto parent = cache.has_parent_path() ? cache.parent_path() : std::filesystem::path(".");
  std::filesystem::create_directories(parent, error);
  // Deux démarrages ne doivent pas publier, ni retirer sur un 404, la copie l’un après l’autre.
  auto lock = prepare::KeyLock::acquire(beside(cache, ".refresh-lock"), stop, prepare::LockWait::Block,
                                        std::chrono::seconds(kFetchSeconds));
  if (!lock) {
    if (lock.error().code != RefusalCode::Cancelled)
      log->info("moteurs de la release : {}", lock.error().detail);
    return;
  }
  if (!release_index_due(cache, version, keys)) return;
  auto stage = prepare::reserve_stage(parent, ".kirari-index-", "engines-nvidia");
  if (!stage) {
    log->info("moteurs de la release : {}", stage.error().detail);
    return;
  }
  const auto part = *stage / "index";
  const auto sig_part = *stage / "signature";
  const auto begun = std::chrono::steady_clock::now();
  auto got = fetch(url, part, kFetchSeconds / 2, stop);
  const bool index_missing = !got && got.error().code == RefusalCode::TrackUnavailable;
  if (got) {
    const auto spent =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begun);
    got = fetch(url + ".sig", sig_part, std::max(2, kFetchSeconds - static_cast<int>(spent.count())), stop);
  }
  if (got) {
    const auto text = read_text(part);
    const auto sig = read_text(sig_part);
    if (!release_index_signed(text, sig, keys))
      got = refuse(RefusalCode::HashMismatch, "signature de l'index absente ou fausse");
    else if (release_index_tag(text) != std::format("v{}", version))
      got = refuse(RefusalCode::FieldOutOfRange, "étiquette de l'index différente de celle du lecteur");
    else if (!stop.stop_requested())
      got = core::write_text_atomically(cache,
                                        std::format("{}{}\n{}", kSignedCopy, signature_payload(sig), text));
  }
  if (auto cleaned = prepare::cleanup_stage(*stage); !cleaned)
    log->warn("moteurs de la release : {}", cleaned.error().detail);
  if (stop.stop_requested()) return;
  if (got) {
    std::filesystem::remove(absent, error);
    std::filesystem::remove(tried, error);
    return;
  }
  if (got.error().code == RefusalCode::Cancelled) return;  // lecteur fermé
  if (got.error().code == RefusalCode::TrackUnavailable) {
    // Sans index sur la release, retire la copie gardée. Si seule la signature manque,
    // la publication peut être en cours : la dernière copie vérifiée reste disponible.
    if (index_missing) {
      std::filesystem::remove(cache, error);
      std::filesystem::remove(signature, error);
    }
    write_marker(absent, version);
  } else {
    write_marker(tried, version);
  }
  log->info("moteurs de la release : {}", got.error().detail);
}

ReleaseIndexCopy read_release_index(const std::filesystem::path& cache) {
  std::string stored = read_text(cache);
  if (!stored.starts_with(kSignedCopy))
    return {.text = std::move(stored), .signature = read_text(beside(cache, ".sig"))};
  const auto newline = stored.find('\n', kSignedCopy.size());
  if (newline == std::string::npos) return {};
  return {.text = stored.substr(newline + 1),
          .signature = stored.substr(kSignedCopy.size(), newline - kSignedCopy.size())};
}

void merge_release_copy(EngineCatalog& catalog, const ReleaseIndexCopy& copy, std::string_view repo,
                        const std::vector<std::string>& keys, std::string_view version) {
  if (repo.empty() || copy.text.empty()) return;
  // Signé en CI par la clé des mises à jour (A26-02) : sans signature valide, rien n'est cru, même la copie
  // gardée (un fichier du dossier de données changé à la main ne pointe pas ailleurs).
  if (!release_index_signed(copy.text, copy.signature, keys)) {
    catalog.problem += std::format("{}engines-nvidia.lock : signature absente ou fausse, écarté",
                                   catalog.problem.empty() ? "" : " ; ");
    return;
  }
  // Un index signé ne vaut que pour la release qu'il nomme : un ancien (ou celui d'une autre version),
  // recopié avec sa signature dans cette release, ne ramène pas un ancien module (B51-06).
  const std::string expected = std::format("v{}", version);
  if (const std::string tag = release_index_tag(copy.text); tag != expected) {
    catalog.problem += std::format("{}engines-nvidia.lock : étiquette « {} » au lieu de « {} », écarté",
                                   catalog.problem.empty() ? "" : " ; ", tag, expected);
    return;
  }
  const std::string prefix = release_repo_prefix(repo);
  if (const int taken = merge_release_index(catalog, copy.text, prefix); taken > 0)
    log_channel("app")->info("moteurs de la release : {} paquet(s) de {}", taken, prefix);
}

void add_release_engines(EngineCatalog& catalog, const std::filesystem::path& cache, std::string_view repo,
                         std::string_view version, const std::vector<std::string>& keys,
                         const UpdateFetch& fetch) {
  if (repo.empty() || version.empty()) return;
  refresh_release_index(cache, repo, version, keys, fetch, std::stop_token{});
  merge_release_copy(catalog, read_release_index(cache), repo, keys, version);
}

}  // namespace kirari::app::play
