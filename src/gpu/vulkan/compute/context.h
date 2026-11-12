// Contexte de calcul Vulkan : un appareil, sa file compute et ses tables de fonctions.
// Deux modes. Adopter l'appareil d'un autre (libplacebo en lecture) : le contexte ne détruit
// alors que ce qu'il a créé lui-même. Créer son propre appareil (tests, outils, banc).
// Un contexte appartient au thread qui l'utilise ; il ne se copie ni ne se déplace.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/caps.h"
#include "gpu/vulkan/compute/destroyed_log.h"
#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/library.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace spdlog {
class logger;
}  // namespace spdlog

namespace ajp::gpu::vk {

class Recording;

// Un appareil à créer.
// Délai d'une fin de vie (VkContext::settle) : au moins kSettleFloor, et kSettleFactor fois la plus longue
// attente hôte réussie jusque-là, sans dépasser kSettleCeiling (P2.16) : un GPU lent n'est pas tenu pour
// bloqué sur un délai fait pour un GPU rapide.
inline constexpr std::chrono::milliseconds kSettleFloor{3000};
inline constexpr std::chrono::milliseconds kSettleCeiling{30000};
inline constexpr int kSettleFactor = 8;

struct CreateOptions {
  // Chargeur : ce vkGetInstanceProcAddr s'il est donné, sinon la bibliothèque library, sinon
  // le chargeur du système.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  std::optional<std::filesystem::path> library;
  std::optional<DeviceUuid> device;     // choisi par UUID ; par défaut le premier énuméré
  std::vector<std::string> extensions;  // extensions d'appareil exigées en plus
  // Couche VK_LAYER_KHRONOS_validation ; refus FeatureUnavailable si elle manque. Ses messages
  // (avertissements et erreurs, par VK_EXT_debug_utils du chargeur) vont au journal et à
  // validation_sink s'il est donné, avec error pour une erreur, depuis n'importe quel thread ;
  // sans VK_EXT_debug_utils, la couche écrit sur sa sortie standard.
  bool validation = false;
  std::function<void(bool error, std::string_view message)> validation_sink;
  // VK_KHR_cooperative_matrix et le modèle mémoire Vulkan, activés si l'appareil les a tous
  // deux ; caps().enabled.cooperative_matrix dit s'ils le sont.
  bool cooperative_matrix = false;
  // Tests : sur un appareil à une file, les voies en plus reprennent la file principale (même
  // ordre d'exécution, mais toute la logique des voies est exercée).
  bool alias_lanes = false;
  // Délai minimum d'une fin de vie (settle) ; il grandit avec la plus longue attente réussie (P2.16).
  std::chrono::milliseconds settle_floor = kSettleFloor;
};

// Un appareil existant. Les champs reprennent ceux de pl_vulkan.
struct AdoptedDevice {
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t api_version = 0;  // VkApplicationInfo.apiVersion de l'instance
  std::uint32_t queue_family = 0;
  std::uint32_t queue_index = 0;
  // Autres files de la même famille, à nous seuls (aucun verrou) : les voies de calcul en plus
  // (VkContext::queue_lanes, gpu/lanes.h). Vide : une seule voie.
  std::vector<std::uint32_t> lane_queue_indices;
  const VkPhysicalDeviceFeatures2* features = nullptr;  // la chaîne passée à vkCreateDevice
  std::vector<std::string> extensions;                  // extensions activées sur le VkDevice
  // Accès exclusif à la file partagée (pl_vulkan.lock_queue et unlock_queue) ; vides quand la
  // file n'est qu'à nous.
  std::function<void()> lock_queue;
  std::function<void()> unlock_queue;
};

// Ce que le rappel de VK_EXT_debug_utils reçoit : le journal et le relais de CreateOptions.
struct ValidationRelay {
  std::shared_ptr<spdlog::logger> log;
  std::function<void(bool error, std::string_view message)> sink;
};

// Une valeur d'un sémaphore timeline : ce qu'une soumission attend ou signale.
struct TimelinePoint {
  VkSemaphore semaphore = VK_NULL_HANDLE;
  std::uint64_t value = 0;
};

class VkContext {
 public:
  // Refus : LibraryUnavailable (chargeur ou fonction absents), DeviceUnavailable (aucun appareil,
  // UUID introuvable, aucune file compute), FeatureUnavailable (Vulkan < 1.2, timeline
  // semaphores, extension ou couche de validation absentes), VulkanCallFailed.
  static std::expected<std::unique_ptr<VkContext>, Refusal> create(const CreateOptions& options = {});
  static std::expected<std::unique_ptr<VkContext>, Refusal> adopt(AdoptedDevice adopted);

  VkContext(const VkContext&) = delete;
  VkContext& operator=(const VkContext&) = delete;
  VkContext(VkContext&&) = delete;
  VkContext& operator=(VkContext&&) = delete;
  ~VkContext();

  const InstanceFunctions& instance_functions() const { return instance_fn_; }
  const DeviceFunctions& fn() const { return device_fn_; }
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const { return get_instance_proc_addr_; }
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  // Files sur lesquelles soumettre : 1 + les files des voies en plus (lane_queue_indices, ou celles
  // créées avec l'appareil). La voie 0 est queue().
  std::size_t queue_lanes() const { return lane_queues_.size() + 1; }
  std::uint32_t queue_family() const { return queue_family_; }
  std::uint32_t queue_index() const { return queue_index_; }
  // La file est-elle partagée avec un autre module (l'affichage de libplacebo), sous son verrou ?
  // Une soumission qui y attend un point que l'hôte signalera plus tard retient tout ce qui la
  // suit, l'affichage compris.
  bool shares_queue() const { return static_cast<bool>(lock_queue_); }
  const DeviceCaps& caps() const { return caps_; }
  const DeviceUuid& uuid() const { return caps_.uuid; }
  bool owns_device() const { return owns_device_; }
  spdlog::logger& log() const { return *log_; }
  // Les tampons et images détruits (destroyed_log.h) : chaque destruction s'y note.
  DestroyedLog& destroyed() const { return destroyed_; }

  // Soumet l'enregistrement sur la file compute : il attend waits côté GPU et signale
  // signals. Ses exécutions successives s'enchaînent côté GPU, sans attente hôte.
  Done submit(Recording& recording, std::span<const TimelinePoint> waits = {},
              std::span<const TimelinePoint> signals = {});

  // Comme submit, mais l'enregistrement rejoint un lot que flush() soumet d'un seul
  // vkQueueSubmit (une soumission par image plutôt qu'une par noyau). submit, run_once,
  // wait_on_host et la destruction d'un enregistrement vident le lot d'abord : l'ordre des
  // soumissions reste celui des appels, et aucune attente ne porte sur un travail jamais soumis.
  // Un sémaphore lu hors du contexte (autre file, autre API) n'est fiable qu'après flush().
  //
  // Dans le lot, les enregistrements qui se suivent partagent un VkSubmitInfo : l'ordre de
  // soumission et la barrière finale de chacun ordonnent déjà le suivant après lui, alors une
  // attente sur un point que le groupe signale est omise, et les signaux du groupe partent à sa
  // fin (la plus grande valeur par sémaphore). Une attente d'ailleurs (slot du ring, image
  // décodée, signal de l'hôte) ouvre un groupe : les enregistrements d'avant ne l'attendent pas.
  Done enqueue(Recording& recording, std::span<const TimelinePoint> waits = {},
               std::span<const TimelinePoint> signals = {});
  // Le même sur la file de la voie lane (< queue_lanes()). Les lots de voies différentes ne
  // s'ordonnent que par leurs sémaphores ; dans une voie, comme sur une file unique.
  Done enqueue(Recording& recording, std::span<const TimelinePoint> waits,
               std::span<const TimelinePoint> signals, std::size_t lane);
  // Une soumission sans commande sur la voie lane : elle attend waits et signale signals (la
  // jointure des voies, Backend::signal).
  Done enqueue_join(std::span<const TimelinePoint> waits, std::span<const TimelinePoint> signals,
                    std::size_t lane);
  Done flush();
  // Appelé au début de flush() quand le lot n'est pas vide, sur le fil du contexte : attendre ce
  // que l'hôte écrit encore pour ce lot sur un autre fil (envois d'images), avant que
  // vkQueueSubmit rende ces écritures visibles de l'appareil (garantie d'ordre des écritures
  // hôte : celles d'avant la soumission). Un refus annule la soumission. Un seul rappel ; vide
  // pour le retirer.
  void before_flush(std::function<Done()> hook) { before_flush_ = std::move(hook); }
  // Tests : les count prochains vkQueueSubmit rendent VK_ERROR_OUT_OF_DEVICE_MEMORY sans rien soumettre,
  // comme une carte à court de mémoire un instant (A45-05).
  void fail_next_submits(std::uint32_t count) { failing_submits_ = count; }
  std::uint64_t queue_submits() const { return queue_submits_; }  // appels à vkQueueSubmit
  std::uint64_t submit_infos() const { return submit_infos_; }    // VkSubmitInfo soumis, tous appels

  // Attente hôte d'une valeur (tests, outils, envoi et lecture hôte) ; chaque appel compte dans
  // host_waits. Refus VulkanCallFailed au délai dépassé ou à la perte de l'appareil.
  Done wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout = std::chrono::seconds(10));
  std::uint64_t host_waits() const { return host_waits_; }

  // Enregistre, soumet et attend sur l'hôte un command buffer jetable : envoi et lecture hôte,
  // transitions d'images à la création. Compte une attente hôte.
  Done run_once(const std::function<void(VkCommandBuffer)>& record);

  // Fin de vie d'une ressource que le GPU lit peut-être encore (tenseur, enregistrement) :
  // attend au plus settle_timeout() que point soit atteint ; sans point, la dernière soumission de
  // la file, dont le signal couvre tout ce qui l'a précédée. Faux si l'attente échoue ou dépasse
  // le délai : le GPU est tenu pour bloqué, les fins de vie suivantes n'attendent plus, et
  // l'appelant laisse fuir sa ressource plutôt que de la rendre à un GPU qui la lit encore. Le GPU
  // bloqué est réarmé dès qu'une fin de vie trouve ses points atteints, sans attendre.
  std::chrono::nanoseconds settle_timeout() const;
  bool settle(std::optional<TimelinePoint> point = std::nullopt);
  // Le même pour plusieurs points (la dernière soumission de chaque voie) : tous atteints.
  bool settle(std::span<const TimelinePoint> points);
  // Un sémaphore détruit (TimelineSemaphore) : la dernière soumission qui le signalait est oubliée.
  void forget(VkSemaphore semaphore) const {
    for (TimelinePoint& point : last_submitted_)
      if (point.semaphore == semaphore) point = {};
  }
  bool stalled() const { return stalled_.load(std::memory_order_acquire); }
  // Un objet qu'un GPU tenu pour bloqué ne laisse pas détruire (pipelines, modules, caches : un
  // enregistrement peut encore s'en servir) : sa destruction est remise à la fin du contexte, après l'attente
  // de l'appareil. Sur un GPU qui reste bloqué, le contexte ne détruit rien non plus et la fin du processus
  // rend le tout.
  void retire(std::function<void()> destroy) const;
  std::size_t retired() const;
  // Appareil perdu (P2.8) : un appel a rendu VK_ERROR_DEVICE_LOST. Plus aucune soumission ne part, chacune
  // est refusée en DeviceLost ; seul un nouveau contexte (relancer le lecteur) en sort. Sûr depuis tout fil.
  bool lost() const { return lost_.load(std::memory_order_acquire); }
  void mark_lost() const { lost_.store(true, std::memory_order_release); }
  // Le refus de check(), l'appareil noté perdu au passage quand c'est le cas.
  Done observe(VkResult result, std::string_view call) const {
    if (result == VK_ERROR_DEVICE_LOST) mark_lost();
    return check(result, call);
  }

 private:
  VkContext() = default;
  Done create_instance(const CreateOptions& options);
  Done create_device(const CreateOptions& options);
  Done finish_setup();  // file, outils internes, journal
  Done queue_submit(std::span<const VkSubmitInfo> infos, std::size_t lane = 0);
  // Après un vkQueueSubmit refusé au groupe first d'un lot de groups groupes et count entrées : renvoie
  // les signaux promis sans leurs command buffers, ou tient le contexte pour perdu (queue.cpp).
  void recover_signals(std::size_t first, std::size_t groups, std::size_t count);

  // Une soumission du lot, ses tableaux réutilisés d'un lot à l'autre.
  struct Batched {
    Recording* recording = nullptr;  // nul : une jointure, sans commande (enqueue_join)
    std::size_t lane = 0;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<std::uint64_t> wait_values;
    std::size_t own_waits = 0;  // les premières attentes : l'exécution précédente du même enregistrement
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<std::uint64_t> signal_values;
  };
  // Un VkSubmitInfo du lot : des soumissions qui se suivent, leurs attentes et signaux fondus.
  struct Group {
    std::size_t lane = 0;
    std::size_t entry = 0;  // première entrée du lot
    std::size_t first = 0;  // dans batch_buffers_
    std::size_t count = 0;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<std::uint64_t> wait_values;
    std::vector<VkPipelineStageFlags> stages;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<std::uint64_t> signal_values;
  };
  static bool joins(const Group& group, const Batched& entry);
  static void merge(Group& group, const Batched& entry);

  std::optional<VulkanLibrary> library_;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
  InstanceFunctions instance_fn_;
  DeviceFunctions device_fn_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::uint32_t queue_family_ = 0;
  std::uint32_t queue_index_ = 0;
  std::vector<std::uint32_t> lane_indices_;  // files des voies en plus
  std::vector<VkQueue> lane_queues_;
  bool owns_instance_ = false;
  bool owns_device_ = false;
  // Couche de validation demandée : son rappel, détruit avant l'instance.
  std::unique_ptr<ValidationRelay> validation_;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger_ = nullptr;
  std::function<void()> lock_queue_;
  std::function<void()> unlock_queue_;
  DeviceCaps caps_;
  // Outils de run_once : un pool jetable et un sémaphore qui compte ses exécutions.
  VkCommandPool once_pool_ = VK_NULL_HANDLE;
  VkSemaphore once_done_ = VK_NULL_HANDLE;
  std::uint64_t once_count_ = 0;
  std::uint64_t host_waits_ = 0;
  std::uint64_t queue_submits_ = 0;
  std::uint32_t failing_submits_ = 0;  // fail_next_submits
  std::uint64_t submit_infos_ = 0;
  std::vector<Batched> batch_;
  std::size_t batched_ = 0;  // entrées de batch_ en attente
  std::function<Done()> before_flush_;
  std::vector<Group> batch_groups_;
  std::vector<VkTimelineSemaphoreSubmitInfo> batch_timelines_;
  std::vector<VkSubmitInfo> batch_infos_;
  std::vector<VkCommandBuffer> batch_buffers_;
  // Par voie : le premier signal de la dernière soumission.
  // mutable : un sémaphore détruit s'en retire (forget), settle ne l'attend plus.
  mutable std::vector<TimelinePoint> last_submitted_ = std::vector<TimelinePoint>(1);
  mutable std::mutex retired_mutex_;
  mutable std::vector<std::function<void()>> retired_;
  std::atomic<bool> stalled_{false};  // une fin de vie a attendu en vain : plus aucune n'attend
  std::chrono::milliseconds settle_floor_ = kSettleFloor;
  std::atomic<std::int64_t> longest_wait_ns_{0};  // la plus longue attente hôte réussie
  void note_wait(std::chrono::steady_clock::time_point since);
  mutable std::atomic<bool> lost_{false};
  mutable DestroyedLog destroyed_;
  std::shared_ptr<spdlog::logger> log_;
};

}  // namespace ajp::gpu::vk
