// Doublures des modules que la tranche verticale relie : l'écran (present + render), les
// boîtes des autres threads, l'horloge maître et le thread gpu.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "app/play/links.h"
#include "app/play/pipeline.h"
#include "app/play/video_stage.h"
#include "core/clock.h"
#include "core/frame.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/ring.h"

namespace ajp::app::play::test {

// Ce que la session a envoyé aux autres threads, dans l'ordre.
struct FakeStages final : Stages {
  std::vector<std::string> log;  // « flush 1 », « seek 5000000 », « play 1 0 », « pause »…
  std::vector<SlotFree> freed;
  std::vector<subs::RenderRequest> requests;
  bool quit_called = false;

  bool flush(std::uint32_t epoch) override;
  bool seek(const cmd::Seek& seek) override;
  bool play(std::uint32_t epoch, std::int64_t media_us) override;
  bool pause() override;
  bool slot_free(const SlotFree& free) override;
  bool request_subtitles(const subs::RenderRequest& request) override;
  void quit() override { quit_called = true; }
};

struct Shown {
  std::optional<RingFrame> frame;
  std::optional<std::int64_t> subtitles_pts;
  std::vector<std::string> osd;
};

// Écran sans fenêtre : chaque présentation avance d'une période fixe.
struct FakeScreen final : Screen {
  OutputSize output{1920, 1080};
  std::int64_t period_us = 16'667;
  std::int64_t now_us = 0;
  bool is_fullscreen = false;
  std::vector<Shown> shown;
  std::vector<RingFrame> retired;

  OutputSize size() const override { return output; }
  std::expected<ScreenTiming, Refusal> show(const ShowRequest& request) override;
  SlotFree retire(const RingFrame& frame) override;
  void set_fullscreen(bool on) override { is_fullscreen = on; }
  bool fullscreen() const override { return is_fullscreen; }
};

struct FakeClock final : MasterClock {
  ClockSnapshot value;
  ClockSnapshot snapshot() const override { return value; }
};

// Image du ring d'un pts donné, slot et valeur prêts comme gpu les poserait.
RingFrame ring_frame(std::uint32_t slot, std::int64_t pts_us, std::uint32_t epoch = 0,
                     std::uint64_t ready_value = 1);

// Thread gpu de substitution : chaque image décodée prend un slot libre et part telle
// quelle au thread main, sans calcul. Même contrôle de flux que le vrai : au plus
// kMaxFrames images en vol, un Credit par image sortie, slots rendus par SlotFree.
class PassthroughStage final : public VideoStage, public FrameSink {
 public:
  static constexpr std::size_t kMaxFrames = 4;
  static std::expected<std::unique_ptr<VideoStage>, Refusal> start(VideoStageInit init, std::uint32_t slots);

  PassthroughStage(VideoStageInit init, std::uint32_t slots);
  ~PassthroughStage() override = default;  // le std::jthread s'arrête en premier
  FrameSink& frames() override { return *this; }
  bool try_push(DecodedFrame&& frame) override;
  bool post(const SlotFree& free) override { return inbox_.try_push(SlotFree{free}); }
  bool post(const Flush& flush) override { return inbox_.try_push(Flush{flush}); }
  bool post(const Command& command) override { return inbox_.try_push(Command{command}); }

 private:
  using StageInbox = Inbox<DecodedFrame, SlotFree, Command, Flush>;
  void run(const std::stop_token& stop);
  void emit();

  VideoStageInit init_;
  StageInbox inbox_;
  std::atomic<std::size_t> in_flight_{0};
  // Possédé par le thread gpu.
  std::deque<DecodedFrame> pending_;
  std::vector<std::uint64_t> released_;  // dernière valeur rendue par slot
  std::vector<bool> free_;
  std::uint32_t epoch_ = 0;
  std::jthread thread_;  // dernier membre
};

}  // namespace ajp::app::play::test
