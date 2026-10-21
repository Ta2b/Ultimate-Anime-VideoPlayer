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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "core/frame.h"
#include "core/inbox.h"
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
  // Assez pour les dix splats d'un instant de GMFSS déformé après SPAN : trois vues de six mots
  // chacun, l'instant, la grille du mouvement, puis le tampon des sommes et le code de l'opération.
  std::array<std::uint64_t, 256> words{};
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
  // Enregistrements d'une même opération qui ne diffèrent que par leurs mots de paramètre
  // (l'instant t) : pendant qu'une exécution lit les siens, la suivante en prend un autre.
  static constexpr std::uint32_t kParameterSlots = 3;
  static constexpr std::size_t kStagingSlots = 3;
  // Remises de tampons de transfert sans servir après lesquelles un tampon d'une autre forme
  // cède sa place : assez pour que deux formes en alternance gardent chacune les leurs.
  static constexpr std::uint64_t kStagingIdleUses = 256;
  static constexpr std::size_t kScoreSlots = 3;
  // Envois d'images hôte en file pour le fil d'envoi ; au-delà, le fil du GPU attend les siens.
  static constexpr std::size_t kUploadBound = 8;

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
  std::expected<void, Refusal> rife_pair(const Tensor& a, const Tensor& b, float t,
                                         const Tensor& input) override;
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
  std::expected<void, Refusal> gmfss_compose(const Tensor& high_a, const Tensor& high_b, const Tensor& low_a,
                                             const Tensor& low_b, const Tensor& fusion,
                                             const kernels::Window& window, float t,
                                             const Tensor& out) override;
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override;
  std::optional<PairScores> previous_scores() override;

  std::size_t recordings() const { return recordings_.size(); }
  std::size_t retained_frames() const { return retained_.size(); }
  std::size_t uploads_pending() const;   // copies d'images hôte en file ou en cours sur le fil d'envoi
  std::uint64_t uploads_queued() const;  // copies d'images hôte mises en file, depuis le début
  std::size_t plane_views() const { return plane_views_.size(); }  // images décodées lues en place

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
    std::uint64_t upload =
        0;  // dernier envoi hôte vers lui (uploads_queued_) ; occupé tant qu'il n'est pas fait
    // L'image décodée qu'il contient (mémo de copie) : une opération suivante sur la même image
    // (mesure, puis remplissage RIFE, puis ingestion) la relit sans la recopier. Valide tant que
    // l'image vit (le lot recycle les adresses) et que le tampon n'a pas reçu une autre image.
    const FrameData* held_ptr = nullptr;
    std::weak_ptr<const FrameData> held;
  };
  // Une image hôte à recopier dans un tampon d'envoi, par le fil d'envoi : chaque plan, ligne
  // par ligne. L'image est tenue le temps de la copie.
  struct UploadPlane {
    const std::byte* in = nullptr;
    std::size_t stride = 0;
    std::byte* out = nullptr;
    std::uint32_t pitch = 0;
    std::uint32_t row = 0;  // octets utiles par ligne
    std::uint32_t rows = 0;
  };
  struct Upload {
    std::shared_ptr<const FrameData> frame;
    std::array<UploadPlane, 3> planes;
    std::uint32_t count = 0;
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
  // Une image décodée prête à lire : sa vue YUV dans un tampon de transfert (staging), ou ses
  // plans lus en place (planes, staging SIZE_MAX), et son accès Vulkan ouvert s'il y en a un.
  struct Source {
    YuvView view;
    std::size_t staging = 0;
    VkImage image = VK_NULL_HANDLE;
    std::uint64_t pool = 0;  // pool_generation_ du lot de l'image ; 0 pour une image hôte
    std::optional<VulkanImageAccess> access;
    std::optional<YuvPlanes> planes;
  };
  // Les vues des deux plans d'une image décodée lue en place, gardées tant que son lot vit.
  struct PlaneViews {
    std::uint64_t image = 0;
    std::uint64_t pool = 0;
    std::array<VkImageView, 2> views{};
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
  // Le même, avec des mots de paramètre (Recorder::parameter, dans l'ordre de body) hors de la
  // clé : l'instant t qui change à chaque image (un facteur 11011/3200 à 165 Hz en donne des
  // milliers) ne fait ni enregistrement neuf ni éviction. Parmi kParameterSlots enregistrements
  // de la clé : celui qui a déjà ces mots, sinon un à l'arrêt (mots réécrits), sinon un neuf,
  // sinon le plus ancien, attendu sur l'hôte (le GPU a kParameterSlots exécutions de retard).
  template <class Body>
  std::expected<Recording*, Refusal> recording_with(OpKey key, std::span<const std::uint32_t> parameters,
                                                    Body&& body);
  // Le Cached de key, ou nullptr.
  Cached* cached(const OpKey& key);
  // Retire les enregistrements dont la clé porte un tampon ou une image détruits depuis la
  // dernière recherche : leur poignée peut déjà désigner un autre objet (destroyed_log.h).
  void forget_destroyed();
  std::expected<void, Refusal> submit(Recording& recording, std::span<const TimelinePoint> waits = {},
                                      std::span<const TimelinePoint> signals = {});
  void reclaim();
  // L'image prête à lire : en place si direct et si son lot le permet (échantillonnable, format
  // mutable, format multiplan connu), depuis le tampon qui la contient déjà (mémo), sinon
  // recopiée (le fil d'envoi pour une image hôte, le GPU pour une image Vulkan) dans un tampon
  // autre que avoid.
  std::expected<Source, Refusal> source(const DecodedFrame& frame, std::size_t avoid = SIZE_MAX,
                                        bool direct = true);
  // Le tampon qui contient déjà l'image, de la forme demandée et autre que avoid.
  std::optional<std::size_t> memo_for(const DecodedFrame& frame, kernels::Format format, std::uint32_t width,
                                      std::uint32_t height, std::size_t avoid) const;
  // Les plans de l'image lus en place : ses vues, créées à la première lecture.
  std::expected<YuvPlanes, Refusal> plane_views_for(VkImage image, kernels::Format format,
                                                    std::uint32_t width, std::uint32_t height);
  // Attend la file et détruit les vues de plans (changement de lot, fin de vie).
  void release_plane_views();
  // Les plans de l'image se lisent en place : créée échantillonnable à format mutable, au format
  // multiplan de format, dont les plans s'échantillonnent sur l'appareil.
  bool reads_in_place(const VulkanImageAccess& access, kernels::Format format);
  std::expected<std::size_t, Refusal> staging_for(kernels::Format format, std::uint32_t width,
                                                  std::uint32_t height, MemoryUse use, std::size_t avoid);
  // Met l'image hôte en file pour le fil d'envoi (lancé au premier appel), vers le tampon slot.
  std::expected<void, Refusal> upload(const DecodedFrame& frame, std::size_t slot,
                                      std::span<const UploadPlane> planes);
  // Attend que tout envoi en file soit fait : avant la soumission du lot (VkContext::before_flush)
  // et avant de rendre un tampon d'envoi. Refus : le fil d'envoi a cédé.
  std::expected<void, Refusal> uploads_wait();
  std::uint64_t uploads_done() const;
  void serve_uploads(const std::stop_token& stop);
  // Passage de l'image en GENERAL sur notre file (barrière depuis son état), puis sa copie plan
  // par plan vers son tampon (copy_image) ou rien d'autre (enter_image : lecture en place).
  void enter_image(VkCommandBuffer cb, const Source& src) const;
  void copy_image(VkCommandBuffer cb, const Source& src) const;
  // enter_image ou copy_image selon la source, dans l'enregistrement.
  Done prepare_image(Recorder& r, const Source& src) const;
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
  RetiredRecordings retired_;               // sortis du cache, libérés à leur arrêt (après recordings_)
  std::vector<std::uint32_t> splat_times_;  // les instants d'un soft_splat_all, réutilisés
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
  std::vector<PlaneViews> plane_views_;
  // Formats d'image dont l'échantillonnage des plans a été vérifié (reads_in_place), et verdict.
  std::vector<std::pair<std::uint32_t, bool>> plane_formats_;
  // Le fil d'envoi : les copies d'images hôte, hors du fil du GPU, attendues seulement quand le
  // lot part (uploads_wait), d'ordinaire déjà faites. Sa boîte reçoit les envois ; le compte des
  // envois faits et le refus du fil, sous uploads_mutex_, réveillent le fil du GPU.
  // uploads_queued_ n'est écrit que par le fil du GPU, uploads_done_ que par le fil d'envoi.
  Inbox<Upload> uploads_{{kUploadBound}};
  mutable std::mutex uploads_mutex_;
  std::condition_variable uploads_finished_;
  std::uint64_t uploads_queued_ = 0;
  std::uint64_t uploads_done_ = 0;
  std::optional<Refusal> upload_failure_;
  std::jthread upload_thread_;
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
