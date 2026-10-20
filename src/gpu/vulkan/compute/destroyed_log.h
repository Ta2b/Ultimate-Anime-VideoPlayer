// Journal des tampons et images détruits. Une valeur de poignée Vulkan se réutilise après
// destruction, pour un autre objet : un cache d'enregistrements dont la clé porte des poignées y
// lit celles qu'il doit oublier avant toute recherche, sans quoi il rejouerait des descripteurs
// vers une mémoire déjà rendue. Tout thread peut écrire ; la lecture ne prend le verrou que si
// une destruction a eu lieu depuis la précédente.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <vector>

namespace ajp::gpu::vk {

// Un handle non dispatchable en entier : un pointeur en 64 bits, un entier de 64 bits ailleurs.
template <class Handle>
std::uint64_t handle_word(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<std::uintptr_t>(handle);
  } else {
    return static_cast<std::uint64_t>(handle);
  }
}

class DestroyedLog {
 public:
  static constexpr std::size_t kCapacity = 1024;

  struct Read {
    std::uint64_t mark = 0;  // à garder pour la lecture suivante
    bool complete = true;    // faux : des destructions sont perdues, tout est à oublier
  };

  void note(std::uint64_t handle) {
    const std::scoped_lock lock(mutex_);
    const std::uint64_t at = count_.load(std::memory_order_relaxed);
    ring_[at % kCapacity] = handle;
    count_.store(at + 1, std::memory_order_release);
  }

  // Le numéro de la prochaine destruction.
  std::uint64_t mark() const { return count_.load(std::memory_order_acquire); }

  // Ajoute à out les poignées détruites depuis la marque from.
  Read since(std::uint64_t from, std::vector<std::uint64_t>& out) const {
    const std::scoped_lock lock(mutex_);
    const std::uint64_t end = count_.load(std::memory_order_relaxed);
    if (end - from > kCapacity) return {.mark = end, .complete = false};
    for (std::uint64_t i = from; i < end; ++i) out.push_back(ring_[i % kCapacity]);
    return {.mark = end, .complete = true};
  }

 private:
  mutable std::mutex mutex_;
  std::array<std::uint64_t, kCapacity> ring_{};
  std::atomic<std::uint64_t> count_{0};
};

}  // namespace ajp::gpu::vk
