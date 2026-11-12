// La zone d'activation commune aux moteurs TensorRT d'une voie (kUSER_MANAGED). Sans TensorRT : se teste
// avec un pool simulé (set_pool), sans carte.
#pragma once

#include <algorithm>
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
  // Ce qui s'écrit dans la zone (C03-01). Chaque moteur déclare à son enqueue les octets [0, end) qu'il y
  // écrit : used pour un moteur entier (TrtEngine, SPAN FP8), qui fait aussi monter activation_peak ; wrote
  // pour les tampons d'un moteur fusionné. Un moteur fusionné qui garde ses tampons d'un appel au suivant
  // (RIFE et DistilDRBA : l'encodage gardé, D6) les pose au-dessus de activation_peak et s'en déclare le
  // gardien (keep) ; ils restent valables (kept_by) tant qu'aucun autre moteur n'a écrit au-delà de leur
  // début. Un SPAN qui passe entre deux appels de RIFE sur la même zone (une seule voie) ne les efface donc
  // plus.
  void used(const void* user, std::int64_t end) {
    activation_peak_ = std::max(activation_peak_, end);
    wrote(user, end);
  }
  void wrote(const void* user, std::int64_t end) {
    if (keeper_ != nullptr && user != keeper_ && end > keep_from_) keeper_ = nullptr;
  }
  void keep(const void* keeper, std::int64_t from) {
    keeper_ = keeper;
    keep_from_ = from;
  }
  bool kept_by(const void* keeper) const { return keeper != nullptr && keeper_ == keeper; }
  // Les plus grandes activations qu'un moteur entier ait déclarées sur cette zone.
  std::int64_t activation_peak() const { return activation_peak_; }
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
  std::int64_t activation_peak_ = 0;
  const void* keeper_ = nullptr;
  std::int64_t keep_from_ = 0;
};

}  // namespace ajp::gpu::cuda
