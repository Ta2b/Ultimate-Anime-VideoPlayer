// Noyaux de la chaîne (gpu/kernel_ops.h) sur le socle vk_compute : chaque appel vérifie ses
// tenseurs, puis rejoue un enregistrement fait au premier passage de ces mêmes paramètres
// (tampons, tailles, couleur, image), sur la file du backend. Rien ne s'alloue par image une
// fois les enregistrements faits, sauf pour une image décodée en mémoire hôte, recopiée dans
// un tampon d'envoi (décodage logiciel, tests).
//
// Une image décodée Vulkan se lit entre begin_vulkan_access et end_vulkan_access : sa
// disposition et sa file viennent de l'AVVkFrame, la soumission attend sa valeur et signale la
// suivante ; ses plans sont copiés côté GPU dans un tampon que lisent les noyaux YUV. Chaque
// DecodedFrame lue est gardée jusqu'à la fin de sa lecture GPU, constatée sans attente hôte.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "gpu/tensor.h"
#include "gpu/vulkan/backend/queue.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "gpu/vulkan/compute/kernel_types.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk {

// Clé d'un enregistrement : quelques mots qui fixent tous ses paramètres, sans allocation.
struct OpKey {
  std::array<std::uint64_t, 48> words{};
  std::size_t size = 0;

  OpKey& add(std::uint64_t v) {
    if (size < words.size()) words[size] = v;
    ++size;
    return *this;
  }
  bool valid() const { return size <= words.size(); }
  friend bool operator==(const OpKey&, const OpKey&) = default;
};

class VulkanKernelOps final : public KernelOps {
 public:
  static constexpr std::size_t kMaxRecordings = 64;
  static constexpr std::size_t kStagingSlots = 3;
  static constexpr std::size_t kScoreSlots = 3;

  VulkanKernelOps(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, BackendQueue& queue);
  VulkanKernelOps(const VulkanKernelOps&) = delete;
  VulkanKernelOps& operator=(const VulkanKernelOps&) = delete;
  VulkanKernelOps(VulkanKernelOps&&) = delete;
  VulkanKernelOps& operator=(VulkanKernelOps&&) = delete;
  // Attend la fin de tout ce que la file a reçu, puis libère images retenues, vues et tampons.
  ~VulkanKernelOps() override;

  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                      const Tensor& rgb) override;
  std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter, const Tensor& out) override;
  std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) override;
  std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) override;
  std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion, float t,
                                          const Tensor& out) override;
  std::expected<void, Refusal> warp_blend_poses(const Tensor& a, const Tensor& b, const Motion& motion,
                                                const Motion* previous, float t, const Tensor& out) override;
  std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                          const Tensor& input) override;
  std::expected<PairScores, Refusal> scores_now(const Tensor& a, const Tensor& b) override;
  std::expected<float, Refusal> flow_max_now(const Motion& motion) override;
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override;
  std::optional<PairScores> previous_scores() override;

  std::size_t recordings() const { return recordings_.size(); }
  std::size_t retained_frames() const { return retained_.size(); }

 private:
  struct Cached {
    OpKey key;
    std::unique_ptr<Recording> recording;
    std::uint64_t used = 0;
  };
  // Une image YUV dans un tampon de l'appareil, et ce qu'il faut pour la remplir.
  struct Staging {
    kernels::Format format = kernels::Format::Nv12;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Buffer buffer;  // Upload pour une image hôte, Device pour une image Vulkan
    YuvLayout layout;
    std::uint64_t busy_until = 0;  // valeur de la file après laquelle il est libre
  };
  struct Retained {
    std::shared_ptr<const FrameData> frame;
    std::uint64_t until = 0;
  };
  struct ScoreSlot {
    Buffer partials;  // Readback : lu par l'hôte sans copie
    Buffer blocks;
    std::uint64_t value = 0;  // 0 : rien en cours
    PairScores scores;
    std::uint32_t area = 0;  // pw × ph de RIFE
    std::uint32_t count = 0;
    std::uint64_t order = 0;
  };
  struct RingView {
    std::uint32_t generation = 0;
    std::uint32_t index = 0;
    std::uint64_t image = 0;
    VkImageView view = VK_NULL_HANDLE;
    std::uint64_t last_write = 0;
  };
  // Une image décodée prête à lire : sa vue YUV, et son accès Vulkan ouvert s'il y en a un.
  struct Source {
    YuvView view;
    std::size_t staging = 0;
    VkImage image = VK_NULL_HANDLE;
    std::optional<VulkanImageAccess> access;
  };

  // Garder les poses : previous nul, la paire précédente était tenue.
  struct Poses {
    bool keep = false;
    const Motion* previous = nullptr;
  };
  std::expected<void, Refusal> warp(const Tensor& a, const Tensor& b, const Motion& motion,
                                    const Poses& poses, float t, const Tensor& out);
  // Recording de la clé, fait par body au premier passage, puis soumis sur la file.
  template <class Body>
  std::expected<Recording*, Refusal> recording(const OpKey& key, Body&& body);
  std::expected<void, Refusal> submit(Recording& recording, std::span<const TimelinePoint> waits = {},
                                      std::span<const TimelinePoint> signals = {});
  void reclaim();
  std::expected<Source, Refusal> source(const DecodedFrame& frame, std::size_t avoid = SIZE_MAX);
  std::expected<std::size_t, Refusal> staging_for(kernels::Format format, std::uint32_t width,
                                                  std::uint32_t height, MemoryUse use, std::size_t avoid);
  void copy_image(VkCommandBuffer cb, const Source& src) const;
  static void add_source(OpKey& key, const Source& src);
  std::expected<void, Refusal> submit_frames(Recording& recording, std::span<Source> sources,
                                             std::span<const DecodedFrame* const> frames);
  std::expected<VkImageView, Refusal> ring_view(const RingSlotDesc& slot);
  std::expected<std::size_t, Refusal> score_slot(std::uint32_t width, std::uint32_t height);

  VkContext& context_;
  MemoryAllocator& memory_;
  PipelineCache& pipelines_;
  BackendQueue& queue_;
  KernelRunner runner_;
  std::vector<Cached> recordings_;
  std::uint64_t uses_ = 0;
  std::vector<Staging> staging_;
  std::vector<Retained> retained_;
  std::vector<ScoreSlot> scores_;
  std::uint64_t score_order_ = 0;
  std::uint64_t returned_order_ = 0;
  std::vector<RingView> ring_views_;
  std::uint32_t ring_generation_ = 0;
  // Mesures rendues tout de suite (scores_now, flow_max_now) : relues par l'hôte sans copie.
  std::optional<Buffer> pair_sums_;
  std::optional<Buffer> pair_worst_;
  std::optional<Buffer> flow_result_;
  std::expected<Buffer*, Refusal> readback(std::optional<Buffer>& buffer, VkDeviceSize bytes);
  std::vector<TimelinePoint> waits_;  // réutilisés d'un appel à l'autre
  std::vector<TimelinePoint> signals_;
};

}  // namespace ajp::gpu::vk
