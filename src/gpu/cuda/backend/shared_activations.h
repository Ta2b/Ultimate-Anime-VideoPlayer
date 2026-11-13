// La zone d'activation commune aux moteurs TensorRT d'une voie (kUSER_MANAGED). Sans TensorRT : se teste
// avec un pool simulé (set_pool), sans carte.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <utility>
#include <vector>

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
  // Ce qui s'écrit dans la zone (C03-01). Chaque moteur déclare à son enqueue les octets qu'il y écrit : used
  // pour un moteur entier (TrtEngine, SPAN FP8), [0, end), qui fait aussi monter activation_peak ; wrote pour
  // les tampons d'un moteur fusionné, [from, end). Un moteur fusionné qui garde ses tampons d'un appel au
  // suivant (RIFE et DistilDRBA : l'encodage gardé, D6) s'en déclare le gardien (keep) ; ils restent valables
  // (kept_by) tant qu'aucun autre moteur n'a écrit sur eux. Un SPAN qui passe entre deux appels de RIFE sur
  // la même zone (une seule voie) ne les efface donc pas, ni un autre moteur fusionné qui a sa propre région
  // (region_offset) : RIFE plein et RIFE demi-taille de l'échelle par paire gardent chacun leur encodage.
  void used(const void* user, std::int64_t end) {
    activation_peak_ = std::max(activation_peak_, end);
    wrote(user, 0, end);
  }
  void wrote(const void* user, std::int64_t from, std::int64_t end) {
    std::erase_if(keepers_, [&](const Keeper& k) { return k.user != user && k.from < end && from < k.end; });
  }
  void wrote(const void* user, std::int64_t end) { wrote(user, 0, end); }
  void keep(const void* keeper, std::int64_t from, std::int64_t end = kNoEnd) {
    std::erase_if(keepers_, [keeper](const Keeper& k) { return k.user == keeper; });
    keepers_.push_back(Keeper{.user = keeper, .from = from, .end = end});
  }
  bool kept_by(const void* keeper) const {
    return keeper != nullptr &&
           std::ranges::any_of(keepers_, [keeper](const Keeper& k) { return k.user == keeper; });
  }
  // Les plus grandes activations qu'un moteur entier ait déclarées sur cette zone.
  std::int64_t activation_peak() const { return activation_peak_; }
  // La région des tampons d'un moteur fusionné : au-dessus des activations (activation_peak, et floor, ses
  // propres activations) et des régions des moteurs fusionnés posés avant lui, alignée sur 256 octets. La
  // région est notée (ou mise à jour) à l'appel ; une région d'avant qui change de taille déplace celles
  // d'après, qui le voient à leur prochain appel (leur décalage change). drop_region au départ du moteur.
  std::int64_t region_offset(const void* user, std::int64_t floor, std::int64_t bytes) {
    const auto found = std::ranges::find(regions_, user, &Region::user);
    if (found == regions_.end()) {
      regions_.push_back(Region{.user = user, .floor = floor, .bytes = bytes});
    } else {
      found->floor = floor;
      found->bytes = bytes;
    }
    std::int64_t base = activation_peak_;
    for (const Region& r : regions_) base = std::max(base, r.floor);
    std::int64_t offset = align(base);
    for (const Region& r : regions_) {
      if (r.user == user) break;
      offset = align(offset + r.bytes);
    }
    return offset;
  }
  void drop_region(const void* user) {
    std::erase_if(regions_, [user](const Region& r) { return r.user == user; });
    std::erase_if(keepers_, [user](const Keeper& k) { return k.user == user; });
  }
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
  static constexpr std::int64_t kNoEnd = INT64_MAX;
  static constexpr std::int64_t align(std::int64_t bytes) { return (bytes + 255) / 256 * 256; }
  struct Keeper {
    const void* user = nullptr;
    std::int64_t from = 0;
    std::int64_t end = kNoEnd;
  };
  std::vector<Keeper> keepers_;  // au plus un par moteur fusionné de la zone
  struct Region {
    const void* user = nullptr;
    std::int64_t floor = 0;
    std::int64_t bytes = 0;
  };
  std::vector<Region> regions_;
};

}  // namespace ajp::gpu::cuda
