// Thread gpu de la tranche verticale : une image décodée → un slot du ring → un RingFrame au
// thread main. Le contrôle de flux et le protocole des slots (core/ring.h) sont ici ; ce que
// devient l'image (agrandissement, simple copie) est l'affaire d'une sous-classe.
//
// Il tient la place du thread du module gpu (Program, ordonnancements S0 à S3) le temps qu'il
// arrive : la tranche n'agrandit qu'une image à la fois, sans interpolation.
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
  // Écrit frame dans le slot : attendre wait_value côté GPU, signaler signal_value (impaire).
  virtual std::expected<void, Refusal> produce(const DecodedFrame& frame, std::uint32_t slot,
                                               std::uint64_t wait_value, std::uint64_t signal_value) = 0;

 private:
  using StageInbox = Inbox<DecodedFrame, SlotFree, Command, Flush>;
  void run(const std::stop_token& stop, const ThreadReport& report);
  void emit(const ThreadReport& report);

  VideoStageInit init_;
  std::uint32_t generation_;
  StageInbox inbox_;
  std::atomic<std::size_t> in_flight_{0};
  // Possédé par le thread gpu.
  std::deque<DecodedFrame> pending_;
  std::vector<std::uint64_t> released_;  // dernière valeur rendue par slot (0 au premier tour)
  std::vector<bool> free_;
  std::uint32_t epoch_ = 0;
  bool failed_ = false;
  std::jthread thread_;
};

}  // namespace ajp::app::play
