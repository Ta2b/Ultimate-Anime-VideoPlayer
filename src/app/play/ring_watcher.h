// Le veilleur d'écriture de l'anneau (chantier 6.5), actif par défaut à chaque lecture (--no-ring-watcher
// rend le sondage d'avant) : un thread qui attend les sémaphores timeline des slots écrits par gpu
// (vkWaitSemaphores, WAIT_ANY, 100 ms au plus, plus un sémaphore de réveil à lui), date chaque écriture finie
// à la sortie de l'attente et réveille main. main ne sonde plus « la tête est-elle écrite ? » chaque
// milliseconde : il dort jusqu'à son échéance ou jusqu'au réveil, et la latence d'écriture se mesure à
// l'heure exacte.
//
// Un slot se surveille de sa réception par main (watch) à la lecture de sa date (take). forget_all, avant la
// destruction des sémaphores de l'anneau, rend la main quand le thread n'en attend plus aucun.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

namespace ajp::app::play {

class RingWatcher {
 public:
  using Waker = void (*)(void*);

  // Rien sans vkWaitSemaphores ou si le thread ne démarre pas.
  static std::unique_ptr<RingWatcher> start(VkDevice device, PFN_vkGetDeviceProcAddr device_proc, Waker waker,
                                            void* context);
  ~RingWatcher();
  RingWatcher(const RingWatcher&) = delete;
  RingWatcher& operator=(const RingWatcher&) = delete;

  // Surveiller l'écriture de `value` sur `semaphore`.
  void watch(VkSemaphore semaphore, std::uint64_t value);
  // L'heure hôte où l'écriture a été vue finie, une fois ; rien si elle ne l'est pas encore (ou pas
  // surveillée).
  std::optional<std::int64_t> take(VkSemaphore semaphore, std::uint64_t value);
  // Plus aucun sémaphore surveillé, et le thread n'en attend plus : ils peuvent être détruits.
  void forget_all();

  std::uint64_t wakes() const;  // réveils de main envoyés
  // Faux une fois le thread sorti sur une erreur de vkWaitSemaphores (appareil perdu, mémoire) : main sonde
  // alors l'anneau comme sans veilleur (A10-03).
  bool alive() const;

  // Les fonctions Vulkan du veilleur ; publiques pour qu'un test en simule l'échec (start_with).
  struct Api {
    PFN_vkWaitSemaphores wait = nullptr;
    PFN_vkGetSemaphoreCounterValue counter = nullptr;
    PFN_vkSignalSemaphore signal = nullptr;
    PFN_vkCreateSemaphore create = nullptr;
    PFN_vkDestroySemaphore destroy = nullptr;
  };
  static std::unique_ptr<RingWatcher> start_with(VkDevice device, const Api& api, Waker waker, void* context);

 private:
  struct Entry {
    VkSemaphore semaphore;
    std::uint64_t value;
    std::int64_t written_us;  // 0 : pas encore écrite
  };
  RingWatcher(VkDevice device, Api api, VkSemaphore wake, Waker waker, void* context);
  void run(const std::stop_token& stop);
  void nudge();  // signale le sémaphore de réveil, sous mutex_
  void die(VkResult result);

  VkDevice device_;
  Api api_;
  VkSemaphore wake_;
  Waker waker_;
  void* context_;
  mutable std::mutex mutex_;
  std::condition_variable settled_;
  std::vector<Entry> entries_;
  std::uint64_t wake_value_ = 0;  // dernière valeur signalée du sémaphore de réveil
  std::uint64_t forget_epoch_ = 0;
  std::uint64_t seen_epoch_ = 0;  // forget_epoch_ vu par le thread hors de toute attente
  std::uint64_t wakes_ = 0;
  bool stopping_ = false;
  bool dead_ = false;  // le thread est sorti sur une erreur : plus personne n'avance seen_epoch_
  std::jthread thread_;
};

}  // namespace ajp::app::play
