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
#include "app/play/ring_stage.h"
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
  std::vector<double> speeds;  // vitesses acceptées, dans l'ordre
  bool refuse_speed = false;   // boîte audio pleine : set_speed refuse
  bool quit_called = false;

  bool flush(std::uint32_t epoch) override;
  bool seek(const cmd::Seek& seek) override;
  bool play(std::uint32_t epoch, std::int64_t media_us) override;
  bool pause() override;
  bool slot_free(const SlotFree& free) override;
  bool request_subtitles(const subs::RenderRequest& request) override;
  bool audio(const Command& command) override;
  bool set_speed(double speed) override {
    if (refuse_speed) return false;
    speeds.push_back(speed);
    return true;
  }
  void quit() override { quit_called = true; }
};

struct Shown {
  std::optional<RingFrame> frame;
  std::optional<std::int64_t> subtitles_pts;
  std::vector<std::string> osd;
  bool ui = false;  // une interface était à dessiner
};

// Écran sans fenêtre : chaque présentation avance d'une période fixe.
struct FakeScreen final : Screen {
  OutputSize output{1920, 1080};
  std::int64_t period_us = 16'667;
  std::int64_t now_us = 0;
  std::int64_t queue_wait_us = 0;  // attente d'une place en file, rendue à chaque présentation
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
// quelle au thread main, sans calcul ; le contrôle de flux et les slots sont ceux de RingStage.
class PassthroughStage final : public RingStage {
 public:
  static std::expected<std::unique_ptr<VideoStage>, Refusal> start(VideoStageInit init, std::uint32_t slots);
  ~PassthroughStage() override { stop(); }

 private:
  PassthroughStage(VideoStageInit init, std::uint32_t slots) : RingStage(std::move(init), slots, 0) {}
  std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t, std::uint32_t,
                                         std::uint64_t, std::uint64_t) override {
    return Output{.pts_us = frame.pts_us, .left = frame.id, .right = frame.id};
  }
};

}  // namespace ajp::app::play::test
