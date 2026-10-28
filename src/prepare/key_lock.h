// Verrou exclusif par clé de cache, entre threads et entre processus.
//
// Repris de animejanai-inference-rtx@118c272 src/cache_key_lock.h:21-76.
// Le fichier verrou n'est jamais supprimé : remplacer son inode laisserait un
// troisième processus verrouiller un autre fichier pendant qu'un premier tient
// l'ancien. Le noyau rend le verrou à la fermeture ou à la mort du processus.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <stop_token>

#include "core/types.h"

namespace ajp::prepare {

enum class LockWait : std::uint8_t {
  Block,    // attend, par tranches de 20 ms, jusqu'au verrou ou à l'arrêt
  TryOnly,  // rend CacheBusy tout de suite si le verrou est tenu
};

class KeyLock {
 public:
  static std::expected<KeyLock, Refusal> acquire(const std::filesystem::path& lock_file,
                                                 const std::stop_token& stop = {},
                                                 LockWait wait = LockWait::Block);

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
