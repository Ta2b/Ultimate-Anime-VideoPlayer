// Noyaux de la chaîne du backend CPU : chaque opération de KernelOps appelle la référence
// kernels/ref sur la mémoire hôte, tout de suite. La file du backend est donc toujours vide :
// une opération est finie à son retour, et la mesure d'une paire se relit dès l'appel suivant.
//
// Slots du ring : sans GPU, un slot est une mémoire hôte. RingSlotDesc::image porte l'adresse
// de memory_size octets, RGBA16 entrelacé en lignes de width × 8 octets à partir de
// memory_offset ; memory et semaphore_handle restent HandleKind::None. La valeur signalée
// est atteinte au retour de write_ring. Les lectures d'une image décodée finissent au retour :
// rien n'est gardé au-delà de l'appel.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <vector>

#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/cpu/host_tensor.h"
#include "gpu/kernel_ops.h"
#include "gpu/rife_input.h"

namespace ajp::gpu::cpu {

class CpuKernels final : public KernelOps, public RifeInputOps {
 public:
  explicit CpuKernels(const HostMemory& memory) : memory_(&memory) {}

  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                      const Tensor& rgb) override;
  std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter, const Tensor& out) override;
  std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) override;
  std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) override;
  std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion, float t,
                                          const Tensor& out) override;
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override;
  std::optional<PairScores> previous_scores() override;

  std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                          const Tensor& input) override;

 private:
  // Dernière valeur signalée par slot du ring, pour refuser une valeur qui reculerait.
  struct SlotSignal {
    std::uint32_t generation = 0;
    std::uint32_t index = 0;
    std::uint64_t value = 0;
  };

  const HostMemory* memory_;
  std::optional<PairScores> last_scores_;
  std::vector<SlotSignal> signals_;  // les slots de la génération la plus récente
  std::uint32_t newest_generation_ = 0;
};

}  // namespace ajp::gpu::cpu
