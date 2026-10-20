// Thread gpu de la tranche verticale : une image décodée → une ou plusieurs sorties (RIFE : les
// images intermédiaires, puis elle) → un slot du ring et un RingFrame au thread main chacune.
// Le contrôle de flux et le protocole des slots (core/ring.h) sont ici ; ce que deviennent les
// images (interpolation, agrandissement, simple copie) est l'affaire d'une sous-classe.
//
// Il tient la place du thread du module gpu (Program, ordonnancements S0 à S3) le temps qu'il
// arrive.
//
// Contrôle de flux : au plus kMaxFrames images en vol (dans la boîte ou en attente d'un slot) ;
// try_push refuse au-delà, et chaque image sortie rend un Credit au thread source.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <expected>
#include <stop_token>
#include <thread>
#include <vector>

#include "app/play/pipeline.h"
#include "app/play/video_stage.h"
#include "core/frame.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/ring.h"

namespace ajp::app::play {

class RingStage : public VideoStage, public FrameSink {
 public:
  static constexpr std::size_t kMaxFrames = 4;

  RingStage(const RingStage&) = delete;
  RingStage& operator=(const RingStage&) = delete;
  ~RingStage() override;

  FrameSink& frames() override { return *this; }
  bool try_push(DecodedFrame&& frame) override;
  bool post(const SlotFree& free) override { return inbox_.try_push(SlotFree{free}); }
  bool post(const Flush& flush) override { return inbox_.try_push(Flush{flush}); }
  bool post(const Command& command) override { return inbox_.try_push(Command{command}); }

 protected:
  // slots : nombre de slots du ring ; generation : celle du ring.
  RingStage(VideoStageInit init, std::uint32_t slots, std::uint32_t generation);
  // Lance le thread ; à appeler à la fin du constructeur de la sous-classe.
  std::expected<void, Refusal> start(std::string_view name);
  // Arrête le thread et l'attend : le destructeur de chaque sous-classe l'appelle d'abord,
  // car le thread appelle produce() tant qu'il tourne.
  void stop();
  // Ce qu'une sortie montre : son instant et les sources qui l'encadrent.
  struct Output {
    std::int64_t pts_us = 0;
    SourceId left;
    SourceId right;
  };
  // Nombre de sorties de frame (1 à kMaxOutputs), décidé une fois quand elle arrive en tête ;
  // par défaut 1. Un refus arrête la vidéo comme un refus de produce.
  virtual std::expected<std::uint32_t, Refusal> outputs(const DecodedFrame& /*frame*/) { return 1U; }
  // Vrai : l'image en tête attend la suivante avant outputs (DistilDRBA lit l'image d'après la
  // paire), une image de retard. next() la donne pendant outputs et produce : la suivante de la
  // même époque, ou nullptr (première image d'une époque qui finit, fin du flux pas encore vue).
  // Sans suivante, la dernière image du flux reste en attente : elle n'est pas montrée.
  virtual bool wants_next() const { return false; }
  const DecodedFrame* next() const {
    return pending_.size() > 1 && pending_[1].epoch == pending_[0].epoch ? &pending_[1] : nullptr;
  }
  // Écrit la sortie output (0 ≤ output < outputs(frame)) dans le slot : attendre wait_value côté
  // GPU, signaler signal_value (impaire).
  virtual std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t output,
                                                 std::uint32_t slot, std::uint64_t wait_value,
                                                 std::uint64_t signal_value) = 0;

  static constexpr std::uint32_t kMaxOutputs = 8;

 private:
  using StageInbox = Inbox<DecodedFrame, SlotFree, Command, Flush>;
  void run(const std::stop_token& stop, const ThreadReport& report);
  void emit(const ThreadReport& report);
  void fail(const ThreadReport& report, const Refusal& refusal);

  VideoStageInit init_;
  std::uint32_t generation_;
  StageInbox inbox_;
  std::atomic<std::size_t> in_flight_{0};
  // Possédé par le thread gpu.
  std::deque<DecodedFrame> pending_;
  std::uint32_t front_outputs_ = 0;      // sorties de l'image en tête ; 0 : pas encore décidé
  std::uint32_t front_done_ = 0;         // sorties déjà écrites
  std::vector<std::uint64_t> released_;  // dernière valeur rendue par slot (0 au premier tour)
  std::vector<bool> free_;
  std::uint32_t epoch_ = 0;
  bool failed_ = false;
  std::jthread thread_;
};

}  // namespace ajp::app::play
