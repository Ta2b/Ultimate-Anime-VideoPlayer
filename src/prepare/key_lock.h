// Verrou exclusif par clé de cache, entre threads et entre processus.
//
// Repris de animejanai-inference-rtx@118c272 src/cache_key_lock.h:21-76.
// Le fichier verrou n'est retiré que par la purge du cache, pour un moteur disparu
// (cache_purge.cpp). Remplacer son inode pourrait laisser un troisième processus
// verrouiller un autre fichier pendant qu'un premier tient l'ancien : sous POSIX,
// acquire revérifie après la prise que le verrou porte encore son chemin, et sinon
// le rouvre (C34-08). Le noyau rend le verrou à la fermeture ou à la mort du processus.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <stop_token>

#include "core/types.h"

namespace ajp::prepare {

enum class LockWait : std::uint8_t {
  Block,    // attend, par tranches de 20 ms, jusqu'au verrou, à l'arrêt ou au délai maximum
  TryOnly,  // rend CacheBusy tout de suite si le verrou est tenu
};

// Au-delà, une attente Block rend CacheBusy (P2.31) : un autre processus figé ne bloque pas pour toujours.
// Plus long que la plus longue construction TensorRT mesurée (une vingtaine de minutes).
inline constexpr std::chrono::minutes kMaxLockWait{45};

class KeyLock {
 public:
  static std::expected<KeyLock, Refusal> acquire(const std::filesystem::path& lock_file,
                                                 const std::stop_token& stop = {},
                                                 LockWait wait = LockWait::Block,
                                                 std::chrono::milliseconds max_wait = kMaxLockWait);

  KeyLock(KeyLock&& other) noexcept;
  KeyLock& operator=(KeyLock&& other) noexcept;
  KeyLock(const KeyLock&) = delete;
  KeyLock& operator=(const KeyLock&) = delete;
  ~KeyLock();

 private:
  explicit KeyLock(std::intptr_t handle) : handle_(handle) {}
  void close() noexcept;

  std::intptr_t handle_;  // descripteur POSIX, ou HANDLE Windows
};

}  // namespace ajp::prepare
