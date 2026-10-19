// Ce que fait le thread gpu de chaque message, sans le thread : testable pas à pas.
//
// Le travailleur garde les images décodées reçues (au plus la borne de la boîte), le Program en
// cours et son exécuteur, le suivant (Install) et le ring de sortie. Il charge les moteurs d'un
// Install dans le backend du thread. pump() avance tant que possible :
//   - livrer à main les images écrites, dans l'ordre ;
//   - passer à la génération de ring reçue, une fois l'ancienne entièrement livrée (gpu
//     n'écrit plus l'ancienne dès qu'il reçoit la nouvelle) ;
//   - passer au Program suivant entre deux images, quand le ring a sa taille ;
//   - écrire les images finales dans les slots libres, puis seulement prendre l'image décodée
//     suivante.
// Sans slot libre, ou sans ring à la taille du Program, il s'arrête : le thread attend un
// message (SlotFree, RingGeneration, Install) dans sa boîte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "core/frame.h"
#include "core/messages.h"
#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/executor.h"
#include "gpu/output_ring.h"
#include "gpu/program.h"
#include "gpu/ref_slots.h"
#include "gpu/rife_input.h"

namespace ajp::gpu {

// Statistiques pour l'OSD, depuis le démarrage du thread.
struct GpuStats {
  std::uint64_t frames = 0;         // images décodées reçues
  std::uint64_t written = 0;        // images écrites dans le ring
  std::uint64_t interpolated = 0;   // images décidées entre deux images
  std::uint64_t repeated = 0;       // images REPEATED : rien de recalculé
  std::uint64_t scene_cuts = 0;     // coupes de scène mesurées
  std::uint64_t dropped = 0;        // images jetées : époque passée, refus, ring remplacé
  std::uint64_t lost_measures = 0;  // mesures de paire jamais relues
  // Attentes côté hôte sur le GPU. gpu n'appelle jamais Backend::wait() : le compteur prouve
  // qu'il reste à 0.
  std::uint64_t host_waits = 0;
  std::int64_t last_pair_us = 0;  // temps hôte de la dernière image décodée (une paire)
  std::int64_t max_pair_us = 0;
  std::uint64_t plan_hash = 0;  // Program en cours, 0 avant le premier Install
};

class GpuWorker {
 public:
  static constexpr std::size_t kMaxBacklog = 8;

  // backend et rife (nullptr sans noyaux d'entrée de RIFE) vivent plus longtemps que le
  // travailleur, qui détruit ses moteurs et ses tenseurs avant eux.
  GpuWorker(Backend& backend, RifeInputOps* rife, RingFrameSink& sink, ThreadReport report)
      : backend_(&backend), rife_(rife), sink_(&sink), report_(std::move(report)) {}

  // false si la file est pleine : l'appelant tient la borne de la boîte, qui est plus petite.
  bool frame(DecodedFrame frame);
  void slot_free(const SlotFree& message);
  void ring(const RingGeneration& message);
  // Install d'une ProgramRequest : les moteurs se chargent ici, dans le backend du thread.
  void install(const Install& message);
  void flush(const Flush& message);
  // Fin du flux, après la dernière image reçue : les paires qui attendaient une anticipation
  // partent au prochain pump().
  void end_of_stream();

  // Avance tant que possible.
  void pump();
  // Images décodées consommées (traitées ou jetées) depuis l'appel précédent : autant de
  // places rendues au producteur.
  std::size_t take_consumed() { return std::exchange(consumed_, 0); }

  GpuStats stats() const;
  const OutputRing* output_ring() const { return ring_ ? &*ring_ : nullptr; }

 private:
  void adopt_ring();
  void switch_program();
  bool writable() const;
  bool write_outputs();
  void process(const DecodedFrame& frame);
  void retire_executor();

  Backend* backend_;
  RifeInputOps* rife_;
  RingFrameSink* sink_;
  ThreadReport report_;
  FixedQueue<DecodedFrame, kMaxBacklog> backlog_;
  std::unique_ptr<Executor> executor_;
  std::shared_ptr<const Program> next_;
  std::optional<OutputRing> ring_;
  std::optional<OutputRing> next_ring_;  // reçu : remplace ring_ dès que ses images sont livrées
  std::optional<DecodedFrame> last_;     // la dernière source, origine d'un Program installé ensuite
  std::uint32_t epoch_ = 0;
  bool end_of_stream_ = false;
  std::size_t consumed_ = 0;
  GpuStats stats_;
  ExecutorCounters retired_;  // compteurs des exécuteurs remplacés
};

}  // namespace ajp::gpu
