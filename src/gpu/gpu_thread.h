// Thread gpu : exécute le Program installé sur chaque image décodée et écrit le ring.
//
// Il possède son backend (chargé par app avec LoadedBackend) et y charge les moteurs de chaque
// Install ; prepare n'en construit que les artefacts, avec sa propre instance. Sa boîte est
// bornée : images décodées (4 par défaut, comptées jusqu'à ce que le thread les ait
// consommées), SlotFree (la taille du ring), RingGeneration, Install, Command et Flush. Quand
// try_push refuse une image, le producteur attend un Credit dans sa propre boîte : le thread
// le lui rend par GpuLinks::credit dès qu'une image est consommée. Les images finales partent
// vers main par RingFrameSink ; sans slot libre, le thread attend un SlotFree dans sa boîte.
//
// Les commandes d'aujourd'hui ne demandent rien à gpu : pause et pas à pas se règlent par la
// contre-pression du ring, un nouveau Plan arrive par Install. L'arrêt vient du stop_token.
#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>

#include "core/frame.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/thread.h"
#include "core/types.h"
#include "gpu/loader.h"
#include "gpu/output_ring.h"
#include "gpu/program.h"
#include "gpu/worker.h"

namespace ajp::gpu {

using GpuInbox = Inbox<DecodedFrame, SlotFree, RingGeneration, Install, Command, Flush>;

struct GpuBounds {
  std::size_t frames = 4;  // images décodées dans la boîte ou gardées par le thread
  std::size_t slot_frees = kMaxRingSlots;
  std::size_t rings = 2;
  std::size_t installs = 2;
  std::size_t commands = 8;
  std::size_t flushes = 4;
};

// Consommateurs et rapporteur du thread : ils doivent lui survivre.
struct GpuLinks {
  RingFrameSink* frames = nullptr;  // vers le thread main
  std::function<void()> credit;     // pousse un Credit dans la boîte du producteur des images
  ThreadReport report;              // refus, depuis le thread gpu
};

class GpuThread final : public FrameSink {
 public:
  // Le thread prend le backend : ses moteurs et ses tenseurs meurent avec le thread, avant lui.
  // Refus : puits ou rapporteur absents (FieldMissing), bornes nulles ou images au-delà de
  // GpuWorker::kMaxBacklog (FieldOutOfRange), thread refusé par le système (ThreadUnavailable).
  static std::expected<std::unique_ptr<GpuThread>, Refusal> start(LoadedBackend backend, GpuLinks links,
                                                                  GpuBounds bounds = {});

  GpuThread(const GpuThread&) = delete;
  GpuThread& operator=(const GpuThread&) = delete;
  ~GpuThread() override = default;  // jthread : arrêt demandé, puis attente

  // FrameSink : false quand `frames` images attendent déjà ; frame reste alors intacte.
  bool try_push(DecodedFrame&& frame) override;
  bool slot_free(const SlotFree& message) { return inbox_.try_push(SlotFree{message}); }
  bool ring(RingGeneration message) { return inbox_.try_push(std::move(message)); }
  bool install(Install message) { return inbox_.try_push(std::move(message)); }
  bool command(const Command& command) { return inbox_.try_push(Command{command}); }
  bool flush(const Flush& message) { return inbox_.try_push(Flush{message}); }

  // Les statistiques publiées après le dernier message traité.
  GpuStats stats() const;

 private:
  GpuThread(LoadedBackend backend, GpuLinks links, GpuBounds bounds);
  void run(const std::stop_token& stop, const ThreadReport& report);
  void release_frames(std::size_t count);

  LoadedBackend backend_;  // avant tout ce qui tient ses moteurs ou ses tenseurs
  GpuLinks links_;
  GpuBounds bounds_;
  GpuInbox inbox_;
  mutable std::mutex mutex_;
  std::size_t held_ = 0;  // images acceptées et pas encore consommées
  bool refused_ = false;  // un try_push a échoué depuis la dernière place rendue
  GpuStats stats_;
  std::jthread thread_;  // dernier membre : démarre après la boîte, s'arrête avant elle
};

}  // namespace ajp::gpu
