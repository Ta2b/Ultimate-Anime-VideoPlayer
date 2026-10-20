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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
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
  // Assez pour les six splats d'un instant de GMFSS : trois vues de six mots chacun, l'instant,
  // le tampon des sommes et le code de l'opération.
  std::array<std::uint64_t, 128> words{};
  std::size_t size = 0;

  OpKey& add(std::uint64_t v) {
    if (size < words.size()) words[size] = v;
    ++size;
    return *this;
  }
  bool valid() const { return size <= words.size(); }
  std::span<const std::uint64_t> used() const { return {words.data(), std::min(size, words.size())}; }
  friend bool operator==(const OpKey& a, const OpKey& b) {
    return a.size == b.size && std::ranges::equal(a.used(), b.used());
  }
};

class VulkanKernelOps final : public KernelOps {
 public:
  static constexpr std::size_t kMaxRecordings = 64;
  static constexpr std::size_t kStagingSlots = 3;
  // Remises de tampons de transfert sans servir après lesquelles un tampon d'une autre forme
  // cède sa place : assez pour que deux formes en alternance gardent chacune les leurs.
  static constexpr std::uint64_t kStagingIdleUses = 256;
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
  std::expected<void, Refusal> rife_time(const Tensor& input, float t) override;
  std::expected<PairScores, Refusal> scores_now(const Tensor& a, const Tensor& b) override;
  std::expected<std::uint64_t, Refusal> scores_start(const Tensor& a, const Tensor& b) override;
  std::expected<PairScores, Refusal> scores_take(std::uint64_t ticket) override;
  std::expected<float, Refusal> flow_max_now(const Motion& motion) override;
  std::expected<std::uint64_t, Refusal> flow_max_start(const Motion& motion) override;
  std::expected<float, Refusal> flow_max_take(std::uint64_t ticket) override;
  std::expected<void, Refusal> gmfss_frames(const Tensor& a, const Tensor& b, const Tensor& frames) override;
  std::expected<void, Refusal> fill(const Tensor& tensor, float value) override;
  std::expected<void, Refusal> soft_splat(const Tensor& source, const Tensor& motion, float s,
                                          const Tensor& out) override;
  std::expected<void, Refusal> soft_splat_all(std::span<const SplatJob> jobs) override;
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
    std::uint64_t used = 0;        // staging_uses_ à sa dernière remise
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
    std::uint64_t pool = 0;  // pool_generation_ du lot de l'image ; 0 pour une image hôte
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
  // Retire les enregistrements dont la clé porte un tampon ou une image détruits depuis la
  // dernière recherche : leur poignée peut déjà désigner un autre objet (destroyed_log.h).
  void forget_destroyed();
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
  std::uint64_t destroyed_mark_ = 0;  // DestroyedLog::mark à la dernière purge
  std::vector<std::uint64_t> gone_;   // poignées détruites depuis, lues par forget_destroyed
  // Le lot d'images du décodeur en cours, gardé vivant : ses poignées ne peuvent pas passer à un
  // autre lot tant qu'on le tient. Un lot neuf change pool_generation_, qui entre dans les clés
  // des enregistrements qui copient depuis une image : ceux de l'ancien lot ne se rejouent plus.
  std::shared_ptr<const void> pool_;
  std::uintptr_t pool_id_ = 0;
  std::uint64_t pool_generation_ = 0;
  std::vector<Staging> staging_;
  std::uint64_t staging_uses_ = 0;
  std::vector<Retained> retained_;
  std::vector<ScoreSlot> scores_;
  std::uint64_t score_order_ = 0;
  std::uint64_t returned_order_ = 0;
  std::vector<RingView> ring_views_;
  std::uint32_t ring_generation_ = 0;
  // Mesures relues par l'hôte sans copie (scores_start, flow_max_start), kPendingReadbacks
  // emplacements à tour de rôle ; value : leur point sur la file, seul attendu par la relecture.
  struct PairRead {
    std::optional<Buffer> sums;
    std::optional<Buffer> worst;
    std::uint64_t ticket = 0;
    std::uint64_t value = 0;
    std::uint32_t blocks = 0;
    double area = 0.0;  // aire remplie de RIFE
  };
  struct FlowRead {
    std::optional<Buffer> result;
    std::uint64_t ticket = 0;
    std::uint64_t value = 0;
  };
  std::array<PairRead, kPendingReadbacks> pair_reads_;
  std::array<FlowRead, kPendingReadbacks> flow_reads_;
  std::uint64_t pair_tickets_ = 0;  // derniers tickets lancés, par sorte
  std::uint64_t flow_tickets_ = 0;
  // Le tampon d'un emplacement, agrandi au besoin une fois finie l'exécution qui l'écrit encore.
  std::expected<Buffer*, Refusal> readback(std::optional<Buffer>& buffer, VkDeviceSize bytes,
                                           std::uint64_t busy_until);
  // Sommes FP32 du softmax splatting, partagées par tous les splats (la file les ordonne) ;
  // agrandies seulement après la fin de tout ce que la file a reçu.
  std::optional<Buffer> splat_sums_;
  std::vector<TimelinePoint> waits_;  // réutilisés d'un appel à l'autre
  std::vector<TimelinePoint> signals_;
};

}  // namespace ajp::gpu::vk
