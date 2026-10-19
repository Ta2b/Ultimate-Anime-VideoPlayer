// Files bornées de l'ordonnanceur de présentation, sans allocation : les images reçues en
// attente de leur vsync, et les slots à rendre à gpu.
#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "core/messages.h"
#include "core/ring.h"

namespace ajp::present {

// Un slot rendu à gpu (protocole : core/ring.h).
struct FrameRelease {
  SlotFree free;  // release_of(image) : slot, ready_value + 1, génération
  // false : l'image n'a jamais été composée. main doit tout de même attendre ready_value et
  // signaler free.semaphore_value côté GPU, sans rien dessiner (core/ring.h) : un signal depuis
  // l'hôte, avant celui de gpu, casserait l'ordre croissant du sémaphore.
  bool displayed = false;
};

// File circulaire des images reçues, dans l'ordre de présentation.
class FrameQueue {
 public:
  static constexpr std::size_t kCapacity = 32;

  bool empty() const { return count_ == 0; }
  bool full() const { return count_ == kCapacity; }
  std::size_t size() const { return count_; }
  const RingFrame& at(std::size_t i) const { return frames_[(head_ + i) % kCapacity]; }
  void push_back(const RingFrame& frame);  // la file n'est pas pleine
  RingFrame pop_front();                   // la file n'est pas vide
  void clear() { head_ = count_ = 0; }

 private:
  std::array<RingFrame, kCapacity> frames_{};
  std::size_t head_ = 0;
  std::size_t count_ = 0;
};

// Slots à rendre. Ceux déjà remis par la dernière décision sont retirés au début de la
// suivante ; ceux ajoutés entre-temps (image périmée, Flush) partent avec elle.
class ReleaseList {
 public:
  // Toutes les images que l'ordonnanceur tient à la fois (file, image montrée, slots à rendre,
  // remis ou non) : PresentScheduler::push() refuse au-delà, si bien qu'aucun ajout ne déborde.
  static constexpr std::size_t kCapacity = 2 * FrameQueue::kCapacity + 2;

  std::size_t size() const { return count_; }        // remis à la dernière décision compris
  void add(const RingFrame& frame, bool displayed);  // la liste n'est pas pleine
  void forget_delivered();                           // début d'une décision
  std::span<const FrameRelease> deliver();           // tout ce qui attend, remis à app

 private:
  std::array<FrameRelease, kCapacity> items_{};
  std::size_t count_ = 0;
  std::size_t delivered_ = 0;
};

}  // namespace ajp::present
