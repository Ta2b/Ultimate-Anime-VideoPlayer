// Doublures des modules que la tranche verticale relie : l'écran (present + render), les
// boîtes des autres threads, l'horloge maître et le thread gpu.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
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
  int refuse_flushes = 0;      // les n prochains Flush sont refusés (boîte pleine), le Seek passe
  int refuse_clock = 0;        // les n prochains Play ou Pause sont refusés (boîte audio pleine)
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
  bool audio_lost(std::uint32_t epoch, std::int64_t media_us) override {
    log.push_back(std::format("audio perdu {} {}", epoch, media_us));
    return true;
  }
  void quit() override { quit_called = true; }
};

struct Shown {
  std::optional<RingFrame> frame;
  std::optional<std::int64_t> subtitles_pts;
  std::vector<std::string> osd;
  bool ui = false;               // une interface était à dessiner
  std::optional<PairTint> tint;  // débogage des paires (pair_marks.h)
};

// Écran sans fenêtre : chaque présentation avance d'une période fixe.
struct FakeScreen final : Screen {
  OutputSize output{1920, 1080};
  std::int64_t period_us = 16'667;
  std::int64_t now_us = 0;
  std::int64_t queue_wait_us = 0;  // temps passé dans swap_buffers, rendu à chaque présentation
  std::int64_t acquire_us = 0;     // attente d'acquisition (file FIFO pleine), rendue à chaque présentation
  bool is_fullscreen = false;
  std::vector<Shown> shown;
  std::vector<RingFrame> retired;
  std::vector<std::uint32_t> unwritten;  // slots dont l'écriture n'est pas finie (ready faux)

  OutputSize size() const override { return output; }
  std::expected<ScreenTiming, Refusal> show(const ShowRequest& request) override;
  SlotFree retire(const RingFrame& frame) override;
  bool ready(const RingFrame& frame) const override {
    return std::ranges::find(unwritten, frame.slot) == unwritten.end();
  }
  void write(std::uint32_t slot) { std::erase(unwritten, slot); }
  // Veilleur d'écriture (Screen::watch) : les slots surveillés, l'heure vue de leur écriture.
  bool waking = false;
  std::vector<std::uint32_t> watched;
  std::vector<std::pair<std::uint32_t, std::int64_t>> written_us;
  void watch(const RingFrame& frame) override { watched.push_back(frame.slot); }
  std::optional<std::int64_t> written_at(const RingFrame& frame) override {
    const auto it = std::ranges::find(written_us, frame.slot, &std::pair<std::uint32_t, std::int64_t>::first);
    if (it == written_us.end()) return std::nullopt;
    const std::int64_t at = it->second;
    written_us.erase(it);
    return at;
  }
  bool wakes_on_write() const override { return waking; }
  std::optional<Refusal> lost;  // Screen::device_lost
  std::optional<Refusal> device_lost() const override { return lost; }
  std::optional<core::VblankGrid> grid;  // Screen::vblank_grid
  std::optional<core::VblankGrid> vblank_grid() const override { return grid; }
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
