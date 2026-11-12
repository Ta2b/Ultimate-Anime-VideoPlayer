// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:757-806 et
// src/model_identity.cpp:157-197.
#include "prepare/cache.h"

#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

#include "core/file_sync.h"
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

// Ce que « .sha256 » garde : l'empreinte, puis la taille et la date du fichier au moment où elle a été écrite
// (« <sha256> <taille> <date> »). Tant que le moteur garde la même taille et la même date, il n'est pas relu
// ni rehaché à chaque ouverture de vidéo (B36-06) : des centaines de Mo lus du disque, cher à froid ou sur un
// disque lent. Un ancien fichier (l'empreinte seule) se lit encore, et force le hachage.
struct StoredDigest {
  std::string digest;  // vide : absente ou illisible
  bool stamped = false;
  std::uintmax_t size = 0;
  std::int64_t modified = 0;
};

// Taille et date du fichier, comme elles se rangent ; false s'il ne se lit pas.
bool stamp_of(const fs::path& file, std::uintmax_t& size, std::int64_t& modified) {
  std::error_code size_error;
  std::error_code time_error;
  size = fs::file_size(file, size_error);
  modified = static_cast<std::int64_t>(fs::last_write_time(file, time_error).time_since_epoch().count());
  return !size_error && !time_error;
}

StoredDigest stored_digest(const CacheEntry& entry) {
  std::ifstream in(entry.digest_file());
  StoredDigest stored;
  if (!(in >> stored.digest) || !is_sha256_hex(stored.digest)) return {};
  if (in >> stored.size >> stored.modified) stored.stamped = true;
  return stored;
}

// Écrit l'empreinte par un temporaire renommé ; un échec la laisse absente, la reprise suivante la refait.
void store_digest(const CacheEntry& entry, const std::string& digest) {
  std::uintmax_t size = 0;
  std::int64_t modified = 0;
  std::string text = digest;
  if (stamp_of(entry.destination, size, modified)) text += std::format(" {} {}", size, modified);
  std::ignore =
      core::write_text_atomically(entry.digest_file(), text + '\n');  // « \n » seul, sous Windows aussi
}

// L'entrée publiée a-t-elle encore les octets de sa publication (P2.31) ? Une entrée d'avant l'empreinte
// la reçoit ici, sous le verrou de la clé seulement (locked) : sans lui, un processus qui lit l'ancienne
// entrée pouvait écrire son empreinte après celle du remplaçant qu'un autre venait de publier (A05-06).
// Sans verrou, une empreinte manquante rend false : l'appelant passe par le chemin verrouillé.
// Une empreinte qui porte la taille et la date du fichier et les retrouve intactes se croit sans relecture.
std::expected<bool, Refusal> intact(const CacheEntry& entry, bool locked) {
  const StoredDigest stored = stored_digest(entry);
  if (stored.digest.empty() && !locked) return false;
  if (stored.stamped) {
    std::uintmax_t size = 0;
    std::int64_t modified = 0;
    if (stamp_of(entry.destination, size, modified) && size == stored.size && modified == stored.modified)
      return true;
  }
  auto digest = sha256_file_known(entry.destination);
  if (!digest) return std::unexpected(digest.error());
  if (stored.digest.empty()) {
    store_digest(entry, *digest);
    return true;
  }
  return stored.digest == *digest;
}

// L'entrée publiée se reprend si elle est intacte (son empreinte revérifiée, P2.31).
std::expected<bool, Refusal> reusable(const CacheEntry& entry) {
  if (!is_published(entry.destination)) return false;
  return intact(entry, true);  // sous le verrou de la clé
}

// Un moteur repris rafraîchit la date de son empreinte : la purge du cache (cache_purge.h) sait ce qui sert.
void mark_used(const CacheEntry& entry) {
  std::error_code error;
  fs::last_write_time(entry.digest_file(), fs::file_time_type::clock::now(), error);
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
  if (is_published(entry.destination)) {
    if (auto whole = intact(entry, false); whole && *whole) {
      mark_used(entry);
      return Production::Reused;
    }
  }
  const fs::path parent = entry.destination.parent_path();
  std::error_code error;
  fs::create_directories(parent, error);
  if (error) return std::unexpected(io_refusal("création du cache", parent, error));

  auto lock = KeyLock::acquire(entry.lock_file(), stop, LockWait::Block, entry.max_wait);
  if (!lock) return std::unexpected(lock.error());
  if (auto recovered = recover_owned_stages(parent, kStagePrefix, entry.owner()); !recovered) {
    return std::unexpected(recovered.error());
  }
  // Un autre processus a pu publier pendant l'attente du verrou : on la reprend.
  auto reuse = reusable(entry);
  if (!reuse) return std::unexpected(reuse.error());
  if (*reuse) {
    mark_used(entry);
    return Production::Reused;
  }

  auto folder = reserve_stage(parent, kStagePrefix, entry.owner());
  if (!folder) return std::unexpected(folder.error());
  const StageGuard stage(std::move(*folder));
  const fs::path output = stage.folder / "model.engine";
  if (auto made = produce(output); !made) return std::unexpected(made.error());
  // Un arrêt demandé après la fin de la production n'est pas honoré ici : le moteur est complet et vérifié,
  // sa publication est atomique et rapide, et le jeter perdrait des minutes de construction (B05-04). Un
  // arrêt demandé PENDANT la production, lui, a fait rendre Cancelled à produce plus haut.
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
