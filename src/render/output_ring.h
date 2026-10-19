// OutputRing : les slots du ring de sortie (core/ring.h), alloués et exportés par render.
//
// Chaque slot est une pl_tex RGBA16 UNORM échantillonnable et stockable, avec son sémaphore
// timeline. Le protocole hold/release de libplacebo porte celui du ring, sans copie :
//   - à la création, render efface le slot puis le cède à l'écrivain (pl_vulkan_hold_ex) en
//     VK_IMAGE_LAYOUT_GENERAL ; le sémaphore du slot reste à 0, la valeur que gpu attend au
//     premier tour ;
//   - acquire(RingFrame) rend le slot à libplacebo derrière l'attente GPU de ready_value
//     (pl_vulkan_release_ex) : main peut le composer ;
//   - release(slot) le recède à l'écrivain : libplacebo signale ready_value + 1 à la fin de son
//     rendu, et le SlotFree rendu porte cette valeur pour gpu.
// Aucune attente hôte par image. Thread main seulement.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "core/messages.h"
#include "core/ring.h"
#include "core/types.h"

struct pl_tex_t;

namespace ajp::render {

class RenderDevice;

// Qui écrit les slots : un module sur le même VkDevice (VK_QUEUE_FAMILY_IGNORED, les images de
// libplacebo sont partagées entre ses files), ou une autre API par import, comme CUDA
// (VK_QUEUE_FAMILY_EXTERNAL).
enum class SlotWriter : std::uint8_t { SameDevice, ExternalApi };

namespace ring_limits {
inline constexpr std::uint32_t kMaxSlots = 16;
}  // namespace ring_limits

struct OutputRingOptions {
  std::uint32_t slots = 4;  // 1 à ring_limits::kMaxSlots
  int width = 0;            // taille de sortie du Plan
  int height = 0;
  RingFormat format = RingFormat::Rgba16Unorm;
  std::uint32_t generation = 0;
  // OpaqueFd ou OpaqueWin32 pour l'interop ; None : pas d'export (moteur sur le même appareil).
  HandleKind export_kind = HandleKind::None;
  SlotWriter writer = SlotWriter::SameDevice;
};

class OutputRing {
 public:
  // FieldOutOfRange : taille ou nombre de slots ; FormatUnsupported : Yuv444p16 (pas encore) ;
  // FeatureUnavailable : format ou export absents de l'appareil ; RenderFailed : allocation.
  static std::expected<OutputRing, Refusal> create(const RenderDevice& device,
                                                   const OutputRingOptions& options);

  OutputRing(OutputRing&&) noexcept;
  OutputRing& operator=(OutputRing&&) noexcept;
  OutputRing(const OutputRing&) = delete;
  OutputRing& operator=(const OutputRing&) = delete;
  // Attend la fin du travail GPU en cours (pl_gpu_finish) : l'écrivain doit avoir cessé
  // d'écrire, et ne plus rien attendre de ces sémaphores.
  ~OutputRing();

  std::span<const RingSlotDesc> slots() const;
  std::uint32_t generation() const;
  // Allocation dédiée (VkMemoryDedicatedAllocateInfo) : l'importeur doit le savoir
  // (cudaExternalMemoryDedicated, ou la même image dédiée sous Vulkan). RingSlotDesc ne le
  // porte pas encore.
  bool dedicated(std::uint32_t slot) const;
  // VkImageUsageFlags des images des slots (VK_FORMAT_R16G16B16A16_UNORM, tuilage optimal,
  // partagées en VK_SHARING_MODE_CONCURRENT entre les familles graphique, calcul et transfert
  // de pl_vulkan quand elles diffèrent) : ce qu'un importeur Vulkan doit recréer à l'identique.
  std::uint32_t image_usage() const;

  // La texture du slot, pour la composer. Refus : génération d'un autre ring (SequenceOrder,
  // à ignorer), slot inconnu, slot déjà acquis, ready_value paire ou qui ne croît pas.
  std::expected<const pl_tex_t*, Refusal> acquire(const RingFrame& frame);
  // À appeler après le dernier rendu qui lit le slot (pl_render_image, copie hors écran…).
  std::expected<SlotFree, Refusal> release(std::uint32_t slot);

 private:
  struct Impl;
  explicit OutputRing(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
