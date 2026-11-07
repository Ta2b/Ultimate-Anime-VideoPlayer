// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:757-806 et
// src/model_identity.cpp:157-197.
#include "prepare/cache.h"

#include <fstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

#include "prepare/cache_io.h"
#include "prepare/key_lock.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

namespace fs = std::filesystem;

// Supprime le dossier de travail à la sortie, quel que soit le chemin pris.
struct StageGuard {
  fs::path folder;
  explicit StageGuard(fs::path stage) : folder(std::move(stage)) {}
  StageGuard(const StageGuard&) = delete;
  StageGuard& operator=(const StageGuard&) = delete;
  // Un dossier qui résiste reste marqué : la production suivante le récupère sous le verrou.
  ~StageGuard() { std::ignore = cleanup_stage(folder); }
};

// L'empreinte rangée à côté de l'entrée : vide si elle manque ou ne se lit pas.
std::string stored_digest(const CacheEntry& entry) {
  std::ifstream in(entry.digest_file());
  std::string digest;
  if (!(in >> digest) || !is_sha256_hex(digest)) return {};
  return digest;
}

// Écrit l'empreinte par un temporaire renommé ; un échec la laisse absente, la reprise suivante la refait.
void store_digest(const CacheEntry& entry, const std::string& digest) {
  fs::path temporary = entry.digest_file();
  temporary += ".part";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);  // « \n » seul, sous Windows aussi
    out << digest << '\n';
    if (!out.flush()) return;
  }
  std::error_code error;
  fs::rename(temporary, entry.digest_file(), error);
  if (error) fs::remove(temporary, error);
}

// L'entrée publiée a-t-elle encore les octets de sa publication (P2.31) ? Une entrée d'avant l'empreinte
// la reçoit ici.
std::expected<bool, Refusal> intact(const CacheEntry& entry) {
  auto digest = sha256_file_known(entry.destination);
  if (!digest) return std::unexpected(digest.error());
  const std::string stored = stored_digest(entry);
  if (stored.empty()) {
    store_digest(entry, *digest);
    return true;
  }
  return stored == *digest;
}

// L'entrée publiée se reprend, sauf si ce sont exactement les octets refusés
// (aji_trt.cpp:775-786) : un remplaçant publié par un autre processus se reprend.
std::expected<bool, Refusal> reusable(const CacheEntry& entry) {
  if (!is_published(entry.destination)) return false;
  auto whole = intact(entry);
  if (!whole || !*whole) return whole;
  if (entry.rejected_sha256.empty()) return true;
  auto digest = sha256_file_known(entry.destination);
  if (!digest) return std::unexpected(digest.error());
  return *digest != entry.rejected_sha256;
}

}  // namespace

fs::path CacheEntry::digest_file() const {
  fs::path digest = destination;
  digest += ".sha256";
  return digest;
}

fs::path CacheEntry::lock_file() const {
  fs::path lock = destination;
  lock += ".key-lock";
  return lock;
}

std::string CacheEntry::owner() const {
  return kind + ":" + utf8_path(destination.filename());
}

bool is_published(const fs::path& file) {
  std::error_code error;
  return fs::is_regular_file(file, error) && fs::file_size(file, error) > 0 && !error;
}

std::expected<Production, Refusal> produce_once(const CacheEntry& entry, const Producer& produce,
                                                const std::stop_token& stop) {
  if (!entry.rejected_sha256.empty() && !is_sha256_hex(entry.rejected_sha256)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "rejected_sha256 n'est pas un SHA-256"});
  }
  if (entry.rejected_sha256.empty() && is_published(entry.destination)) {
    if (auto whole = intact(entry); whole && *whole) return Production::Reused;
  }
  const fs::path parent = entry.destination.parent_path();
  std::error_code error;
  fs::create_directories(parent, error);
  if (error) return std::unexpected(io_refusal("création du cache", parent, error));

  auto lock = KeyLock::acquire(entry.lock_file(), stop);
  if (!lock) return std::unexpected(lock.error());
  if (auto recovered = recover_owned_stages(parent, kStagePrefix, entry.owner()); !recovered) {
    return std::unexpected(recovered.error());
  }
  // Un autre processus a pu publier pendant l'attente du verrou : on la reprend.
  auto reuse = reusable(entry);
  if (!reuse) return std::unexpected(reuse.error());
  if (*reuse) return Production::Reused;

  auto folder = reserve_stage(parent, kStagePrefix, entry.owner());
  if (!folder) return std::unexpected(folder.error());
  const StageGuard stage(std::move(*folder));
  const fs::path output = stage.folder / "model.engine";
  if (auto made = produce(output); !made) return std::unexpected(made.error());
  if (stop.stop_requested())
    return std::unexpected(Refusal{RefusalCode::Cancelled, "annulé avant publication"});
  if (!is_published(output)) {
    return std::unexpected(
        Refusal{RefusalCode::BuilderFailed, "sortie vide ou absente : " + utf8_path(output)});
  }
  auto digest = sha256_file(output);
  if (!digest) return std::unexpected(digest.error());
  if (auto published = publish_file(output, entry.destination); !published) {
    return std::unexpected(published.error());
  }
  store_digest(entry, *digest);
  return Production::Produced;
}

}  // namespace ajp::prepare
