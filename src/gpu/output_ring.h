// Ring de sortie côté gpu : l'état de chaque slot et les valeurs de son sémaphore, selon le
// protocole de core/ring.h.
//
//   Free ──acquire──▶ Writing ──commit──▶ Ready ──deliver──▶ Shown ──SlotFree──▶ Free
//   Writing ──abandon──▶ Free (écriture refusée avant soumission)
//   Ready ──drop_before──▶ Free (Flush : l'image ne partira jamais)
//
// Ready : écrite, en attente d'une place dans la boîte de main. Shown : livrée à main, qui la
// rendra par SlotFree. Les valeurs d'un slot croissent toujours : l'écriture attend la
// dernière valeur connue et signale l'impaire suivante. Borné, sans allocation.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/messages.h"
#include "core/plan.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "gpu/program.h"

namespace ajp::gpu {

// Les slots d'une génération du ring, que render envoie à gpu à chaque création. Décision de
// conception de w2/fondations 371b2a2 : le message doit rejoindre core/messages.h ; d'ici là,
// il vit ici.
struct RingGeneration {
  std::vector<RingSlotDesc> slots;
};

enum class SlotState : std::uint8_t { Free, Writing, Ready, Shown };

// Ce que gpu appelle pour livrer une image au thread main ; app l'implémente par sa boîte.
// false : la borne est atteinte ; gpu garde l'image et réessaie au prochain SlotFree.
class RingFrameSink {
 public:
  virtual ~RingFrameSink() = default;
  virtual bool try_push(RingFrame&& frame) = 0;
};

enum class Release : std::uint8_t {
  Freed,            // le slot redevient libre
  OtherGeneration,  // SlotFree d'un ring remplacé : ignoré
};

class OutputRing {
 public:
  // Une écriture à soumettre : le slot et sa synchronisation.
  struct Lease {
    std::uint32_t slot = 0;
    const RingSlotDesc* desc = nullptr;
    RingWrite sync;
  };

  // Refus (FieldOutOfRange) : aucun slot ou plus de kMaxRingSlots, indices qui ne suivent pas
  // l'ordre, générations, tailles ou formats mêlés, taille nulle.
  static std::expected<OutputRing, Refusal> create(std::span<const RingSlotDesc> slots);

  std::uint32_t generation() const { return generation_; }
  // Le ring reçoit-il la sortie de ce Plan : même taille, même format ?
  bool fits(const core::Plan& plan) const;
  std::size_t size() const { return size_; }
  SlotState state(std::uint32_t slot) const { return slots_[slot].state; }
  std::size_t count(SlotState state) const;

  // Le prochain slot libre, à tour de rôle, qui passe en écriture ; rien si tous sont pris.
  std::optional<Lease> acquire();
  void abandon(std::uint32_t slot);
  // L'écriture est soumise : l'image part avec son slot, sa génération et la valeur à attendre.
  void commit(std::uint32_t slot, RingFrame frame);
  // Livre les images prêtes dans l'ordre de leur écriture, jusqu'au premier refus du puits.
  std::size_t deliver(RingFrameSink& sink);
  // Refus (SequenceOrder) : slot inconnu, slot qui n'était pas chez main, ou valeur autre que
  // ready_value + 1.
  std::expected<Release, Refusal> release(const SlotFree& message);
  // Les images prêtes d'une époque passée ne partiront pas : leur slot redevient libre.
  std::size_t drop_before(std::uint32_t epoch);

 private:
  struct Slot {
    RingSlotDesc desc;
    SlotState state = SlotState::Free;
    std::uint64_t wait_value = 0;   // à attendre avant la prochaine écriture
    std::uint64_t ready_value = 0;  // signalée par la dernière écriture
    RingFrame frame;                // Ready : l'image à livrer
  };

  OutputRing() = default;

  std::array<Slot, kMaxRingSlots> slots_{};
  std::size_t size_ = 0;
  std::uint32_t generation_ = 0;
  std::size_t next_ = 0;  // prochain slot essayé par acquire
  // Slots prêts, dans l'ordre de leur écriture.
  std::array<std::uint32_t, kMaxRingSlots> ready_{};
  std::size_t ready_head_ = 0;
  std::size_t ready_count_ = 0;
};

}  // namespace ajp::gpu
