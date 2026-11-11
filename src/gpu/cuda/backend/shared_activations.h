// La zone d'activation commune aux moteurs TensorRT d'une voie (kUSER_MANAGED). Sans TensorRT : se teste
// avec un pool simulé (set_pool), sans carte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <utility>

#include "core/types.h"

namespace ajp::gpu::cuda {

// Mémoire d'activation partagée par les moteurs d'une voie du backend (gpu/lanes.h) : ils
// s'exécutent l'un après l'autre sur sa file, une zone de la taille du plus gros suffit. Sans
// elle, RIFE, SPAN et DistilDRBA gardent chacun la leur : plusieurs centaines de Mo de
// plus sur les 6 Go d'une RTX 4050 mobile. Deux voies de calcul, deux zones : SPAN tourne
// pendant RIFE ou DistilDRBA sans écrire leur mémoire.
class SharedActivations {
 public:
  SharedActivations() = default;
  SharedActivations(const SharedActivations&) = delete;
  SharedActivations& operator=(const SharedActivations&) = delete;
  SharedActivations(SharedActivations&&) = delete;
  SharedActivations& operator=(SharedActivations&&) = delete;
  ~SharedActivations();

  // Au moins bytes octets ; une zone plus grande remplace l'ancienne après la fin du travail en
  // cours sur l'appareil (generation() change). Refus GpuMemoryExhausted : la zone garde au moins sa
  // taille d'avant (refaite à cette taille si l'ancienne a dû partir pour l'essai), pour les moteurs
  // déjà posés.
  std::expected<void, Refusal> reserve(std::int64_t bytes);
  void* data() const { return data_; }
  std::int64_t bytes() const { return bytes_; }
  std::uint64_t generation() const { return generation_; }
  // Le dernier moteur qui a posé du travail sur la zone (chaque moteur se nomme à son enqueue) : un moteur
  // qui y garde des tampons d'un appel au suivant (DistilDRBA fusionné, D6) sait si un autre y a écrit.
  const void* last_user() const { return last_user_; }
  void set_last_user(const void* user) { last_user_ = user; }
  // Allocation et libération par le backend (étape 11.4) : dans le pool, et l'ancienne zone rendue derrière
  // toutes les voies au lieu d'un cudaDeviceSynchronize, la dernière aussi à la destruction. Sans elles :
  // cudaMalloc, et cudaFree après synchronisation de l'appareil. Le backend les retire avant de lâcher la
  // zone à sa propre destruction : une zone qui lui survivrait se libère par cudaFree.
  void set_pool(std::function<void*(std::size_t)> alloc, std::function<void(void*)> release) {
    alloc_ = std::move(alloc);
    release_ = std::move(release);
  }

 private:
  void* allocate(std::size_t bytes);
  void release(void* memory);
  void* data_ = nullptr;
  std::function<void*(std::size_t)> alloc_;
  std::function<void(void*)> release_;
  std::int64_t bytes_ = 0;
  std::uint64_t generation_ = 0;
  const void* last_user_ = nullptr;
};

}  // namespace ajp::gpu::cuda
