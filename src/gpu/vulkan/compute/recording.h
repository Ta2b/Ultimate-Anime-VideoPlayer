// Enregistrement d'un command buffer réutilisable : écrit une fois par un Recorder (dispatchs,
// remplissages, barrières entre eux), puis rejoué à chaque soumission (VkContext::submit).
// Chaque exécution signale la valeur suivante du sémaphore timeline de l'enregistrement, et
// la suivante l'attend côté GPU : deux exécutions ne se chevauchent jamais, sans attente hôte.
// L'enregistrement possède ses tampons de travail et de constantes ; les tampons et images
// liés par l'appelant doivent vivre aussi longtemps que lui.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/descriptors.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/sync.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

class Recording {
 public:
  Recording(const Recording&) = delete;
  Recording& operator=(const Recording&) = delete;
  Recording(Recording&&) = delete;
  Recording& operator=(Recording&&) = delete;
  // Attend la fin de la dernière exécution soumise avant de libérer quoi que ce soit.
  ~Recording();

  VkCommandBuffer command_buffer() const { return command_buffer_; }
  const VkContext& context() const { return context_; }
  bool finished() const { return finished_; }
  std::uint32_t dispatches() const { return dispatches_; }
  std::uint32_t barriers() const { return barriers_; }    // entre commandes dépendantes (hors la finale)
  std::uint64_t submitted() const { return submitted_; }  // exécutions soumises
  // Atteint quand la dernière exécution soumise s'achève ; valeur 0 avant la première.
  TimelinePoint last_run() const { return runs_.at(submitted_); }
  const TimelineSemaphore& runs() const { return runs_; }
  // La dernière exécution soumise est finie, ou aucune ne l'a été : constat, sans attente. Une
  // exécution encore dans le lot du contexte (VkContext::enqueue) n'est pas finie.
  bool idle() const;

  // Les mots de paramètre (l'instant d'un warp, d'un splat), dans l'ordre de Recorder::parameter :
  // les noyaux les lisent à chaque exécution, et ils changent entre deux sans réenregistrement.
  std::span<const std::uint32_t> parameters() const { return parameters_; }
  // Réécrit les mots ; seulement à l'arrêt (idle), sinon le GPU les lirait pendant qu'ils
  // changent. Refus InputMismatch : autre nombre de mots, ou exécution en cours.
  Done set_parameters(std::span<const std::uint32_t> values);

 private:
  friend class Recorder;
  friend class VkContext;
  Recording(VkContext& context, TimelineSemaphore runs, DescriptorArena arena)
      : context_(context), runs_(std::move(runs)), arena_(std::move(arena)) {}

  VkContext& context_;
  TimelineSemaphore runs_;
  DescriptorArena arena_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  std::vector<Buffer> owned_;
  std::vector<std::uint32_t> parameters_;
  std::vector<std::byte*> parameter_words_;  // la projection de chaque mot, dans owned_
  std::uint32_t dispatches_ = 0;
  std::uint32_t barriers_ = 0;
  std::uint64_t submitted_ = 0;
  bool finished_ = false;
};

// Enregistrements sortis d'un cache (éviction, tampon détruit) alors que le GPU exécute peut-être
// encore le dernier : gardés jusqu'à leur arrêt (Recording::idle) au lieu d'une attente hôte à
// l'éviction, puis libérés par collect(). Au-delà de kMaxRetired, le plus ancien est libéré tout de
// suite (~Recording attend sa fin). Les tampons qu'ils lient doivent leur survivre, comme au cache.
class RetiredRecordings {
 public:
  static constexpr std::size_t kMaxRetired = 32;

  RetiredRecordings() = default;
  RetiredRecordings(const RetiredRecordings&) = delete;
  RetiredRecordings& operator=(const RetiredRecordings&) = delete;
  RetiredRecordings(RetiredRecordings&&) = default;
  RetiredRecordings& operator=(RetiredRecordings&&) = default;
  ~RetiredRecordings() = default;

  void retire(std::unique_ptr<Recording> recording);
  // Libère ceux qui sont à l'arrêt : constat sur leur sémaphore, sans attente.
  void collect();
  std::size_t size() const { return recordings_.size(); }

 private:
  std::vector<std::unique_ptr<Recording>> recordings_;
};

// Ce qu'un dispatch attend : ce qui le précède et touche l'un de ses tampons ou une image (une
// barrière avant lui, seulement alors), ou rien (aucune barrière, même sur un tampon commun : il
// n'écrit rien que le précédent lit ni ne lit ce qu'il écrit, comme les additions atomiques de
// splats sur des sommes distinctes).
enum class DispatchOrder : std::uint8_t { AfterPrevious, BesidePrevious };

class Recorder {
 public:
  // Refus : ceux de DescriptorArena::create, VulkanCallFailed.
  static std::expected<Recorder, Refusal> begin(VkContext& context, MemoryAllocator& memory,
                                                DescriptorLimits limits = {});

  VkContext& context() const { return *context_; }
  MemoryAllocator& memory() const { return *memory_; }

  // Tampon de travail possédé par l'enregistrement.
  std::expected<BufferView, Refusal> scratch(VkDeviceSize bytes, MemoryUse use = MemoryUse::Device);
  // Données de l'hôte figées dans l'enregistrement (tables de poids) : un tampon Upload rempli
  // maintenant, lu à chaque exécution.
  std::expected<BufferView, Refusal> constants(std::span<const std::byte> bytes);
  // Un mot de paramètre, value d'abord : un tampon Upload de 4 octets que les noyaux lisent à
  // chaque exécution et que Recording::set_parameters réécrit entre deux. Ce qui ne change que
  // lui (l'instant t) ne demande ainsi pas d'enregistrement neuf.
  std::expected<BufferView, Refusal> parameter(std::uint32_t value);

  // Un dispatch, après une barrière sur tout ce qui le précède dans l'enregistrement si une
  // commande depuis la dernière barrière a touché l'un de ses tampons (une plage qui recouvre
  // l'une des siennes : chaque vue liée compte comme lue et écrite) ou une image (order
  // AfterPrevious) ; sans barrière, à côté du précédent (BesidePrevious). images : les vues de
  // stockage du pipeline, puis ses vues échantillonnées. Refus :
  // InputMismatch (nombre de tampons, d'images ou d'octets de push différent du pipeline ; vue
  // vide, mal alignée ou trop grande), FeatureUnavailable (groupes au-delà des limites),
  // GpuMemoryExhausted (pools de descripteurs pleins, DescriptorArena::kMaxPools).
  Done dispatch(const Pipeline& pipeline, std::span<const BufferView> buffers,
                std::span<const VkImageView> images, std::span<const std::byte> push,
                std::array<std::uint32_t, 3> groups, DispatchOrder order = DispatchOrder::AfterPrevious);

  // Remplit la vue de value à chaque exécution (offset et taille multiples de 4), après une
  // barrière si une commande depuis la dernière a touché la plage.
  Done fill(const BufferView& view, std::uint32_t value);

  // Commandes écrites par l'appelant avec ses propres fonctions (copies, horodatages), après
  // une barrière sur tout ce qui précède ; ce qui suit les attend aussi (leurs accès sont
  // inconnus : la commande suivante a sa barrière). Refus InputMismatch : déjà terminé.
  Done commands(const std::function<void(VkCommandBuffer)>& record);

  // Barrière finale vers l'hôte et les transferts, puis vkEndCommandBuffer. Le Recorder est
  // ensuite vide. Refus InputMismatch : déjà terminé.
  std::expected<std::unique_ptr<Recording>, Refusal> finish();

 private:
  Recorder(VkContext& context, MemoryAllocator& memory, std::unique_ptr<Recording> recording)
      : context_(&context), memory_(&memory), recording_(std::move(recording)) {}
  struct Touched {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
  };
  Done open() const;
  // Une commande depuis la dernière barrière a touché l'une de ces vues, ou une image si images.
  bool depends(std::span<const BufferView> buffers, bool images) const;
  void touch(std::span<const BufferView> buffers, bool images);
  // Barrière mémoire sur tout ce qui précède, s'il reste quelque chose de touché.
  void barrier();
  Done check_view(const BufferView& view, std::string_view what) const;

  VkContext* context_;
  MemoryAllocator* memory_;
  std::unique_ptr<Recording> recording_;
  // Ce que les commandes ont touché depuis la dernière barrière : la suivante n'en pose une que
  // si elle en dépend.
  std::vector<Touched> touched_;
  bool touched_images_ = false;
  bool touched_all_ = false;  // commands() : accès inconnus
};

}  // namespace ajp::gpu::vk
