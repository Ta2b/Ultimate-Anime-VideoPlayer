// Le thread du ring (ring_stage.h) sans GPU : une sous-classe qui note ce que voit outputs() et
// peut retenir une sortie. Ce qui se teste : tout ce qui attend dans la boîte est pris avant de
// produire (l'image suivante est vue par next(), constat P7 du second audit), et une image qui
// demande plus de sorties qu'une paire n'en tient arrête l'étage en le disant (P14).
#include "app/play/ring_stage.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/links.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

// La boîte de main : garde les RingFrame reçus.
class KeptRing final : public RingSink {
 public:
  bool try_push(const RingFrame& frame) override {
    const std::scoped_lock lock(mutex_);
    frames_.push_back(frame);
    return true;
  }
  std::size_t size() const {
    const std::scoped_lock lock(mutex_);
    return frames_.size();
  }
  std::vector<RingFrame> all() const {
    const std::scoped_lock lock(mutex_);
    return frames_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<RingFrame> frames_;
};

// Une sortie par image (ou `outputs` sorties) ; la première sortie de l'image `held` attend
// release(). outputs() note, par image, si la suivante était déjà là.
class ProbeStage final : public RingStage {
 public:
  ProbeStage(VideoStageInit init, std::uint32_t slots, std::uint32_t held, std::uint32_t outputs,
             std::size_t extra_frames = 0)
      : RingStage(std::move(init), slots, 0, extra_frames), held_(held), outputs_(outputs) {
    REQUIRE(start("ring-sonde"));
  }
  ~ProbeStage() override {
    release();
    stop();
  }
  ProbeStage(const ProbeStage&) = delete;
  ProbeStage& operator=(const ProbeStage&) = delete;

  void release() {
    {
      const std::scoped_lock lock(mutex_);
      held_ = 0;
    }
    released_.notify_all();
  }
  bool holding() const { return holding_.load(); }
  void wait_for_next(bool on) { waits_.store(on ? 1 : 0); }
  void wait_for_ahead(std::size_t frames) { waits_.store(frames); }
  void same_output(int output) { same_.store(output); }
  int skipped() const { return skipped_.load(); }
  void lose_device(bool on) { lost_.store(on); }
  int health_checks() const { return health_checks_.load(); }
  // Pour chaque image passée par outputs(), dans l'ordre : vrai si next() la suivait.
  std::vector<bool> saw_next() const {
    const std::scoped_lock lock(mutex_);
    return saw_next_;
  }
  // Pour chaque image passée par outputs() : combien de suivantes de son époque next(k) donnait.
  std::vector<std::size_t> saw_ahead() const {
    const std::scoped_lock lock(mutex_);
    return saw_ahead_;
  }

 private:
  std::size_t frames_ahead() const override { return waits_.load(); }
  Fate fate(const DecodedFrame& /*frame*/, std::uint32_t output) override {
    return same_.load() == static_cast<int>(output) ? Fate::Same : Fate::Write;
  }
  void skip(const DecodedFrame& /*frame*/, std::uint32_t /*output*/) override { ++skipped_; }
  std::expected<void, Refusal> health() override {
    ++health_checks_;
    if (lost_.load()) return std::unexpected(Refusal{RefusalCode::DeviceLost, "contexte perdu (sonde)"});
    return {};
  }
  std::expected<std::uint32_t, Refusal> outputs(const DecodedFrame& /*frame*/) override {
    const std::scoped_lock lock(mutex_);
    saw_next_.push_back(next() != nullptr);
    std::size_t ahead = 0;
    while (next(ahead + 1) != nullptr) ++ahead;
    saw_ahead_.push_back(ahead);
    return outputs_;
  }
  std::expected<Output, Refusal> produce(const DecodedFrame& frame, std::uint32_t /*output*/,
                                         std::uint32_t /*slot*/, std::uint64_t /*wait_value*/,
                                         std::uint64_t /*signal_value*/) override {
    std::unique_lock lock(mutex_);
    if (frame.id.value == held_) {
      holding_.store(true);
      released_.wait(lock, [this] { return held_ == 0; });
      holding_.store(false);
    }
    return Output{.pts_us = frame.pts_us, .left = frame.id, .right = frame.id};
  }

  mutable std::mutex mutex_;
  std::condition_variable released_;
  std::uint32_t held_;
  std::uint32_t outputs_;
  std::atomic<bool> holding_{false};
  std::atomic<int> same_{-1};
  std::atomic<int> skipped_{0};
  std::atomic<std::size_t> waits_{0};
  std::atomic<bool> lost_{false};
  std::atomic<int> health_checks_{0};
  std::vector<bool> saw_next_;
  std::vector<std::size_t> saw_ahead_;
};

struct Bench {
  KeptRing ring;
  std::atomic<int> credits{0};
  std::mutex refusals_mutex;
  std::vector<Refusal> refusals;

  VideoStageInit init() {
    return VideoStageInit{.ring = &ring,
                          .credit = [this] { ++credits; },
                          .report = ThreadReport([this](const Refusal& refusal) {
                            const std::scoped_lock lock(refusals_mutex);
                            refusals.push_back(refusal);
                          })};
  }
  std::vector<Refusal> reported() {
    const std::scoped_lock lock(refusals_mutex);
    return refusals;
  }
};

DecodedFrame frame(std::uint32_t id, std::uint32_t epoch = 0) {
  return DecodedFrame{.id = SourceId{id},
                      .width = 64,
                      .height = 48,
                      .pts_us = static_cast<std::int64_t>(id) * 40'000,
                      .epoch = epoch};
}

template <class Done>
bool wait_for(Done done) {
  for (int i = 0; i < 5000; ++i) {
    if (done()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return done();
}

}  // namespace

TEST_CASE("Ring : ce qui attend dans la boîte est pris avant de produire, la suivante est vue (P7)",
          "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 8, /*held=*/1, /*outputs=*/1);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return stage.holding(); }));
  // L'étage est dans la sortie de l'image 1 : 2 et 3 attendent ensemble dans la boîte.
  REQUIRE(stage.frames().try_push(frame(2)));
  REQUIRE(stage.frames().try_push(frame(3)));
  stage.release();
  REQUIRE(wait_for([&] { return bench.ring.size() == 3; }));
  // 1 était seule ; 2 voit 3, prise de la boîte avec elle ; 3 est la dernière.
  CHECK(stage.saw_next() == std::vector<bool>{false, true, false});
  CHECK(bench.credits.load() == 3);
  CHECK(bench.reported().empty());
}

TEST_CASE("Ring : un étage qui attend la suivante produit la dernière image à la fin du flux",
          "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 8, /*held=*/0, /*outputs=*/1);
  stage.wait_for_next(true);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(stage.frames().try_push(frame(2)));
  // 1 voit 2 ; 2 attend une suivante qui ne vient pas.
  REQUIRE(wait_for([&] { return bench.ring.size() == 1; }));
  REQUIRE(stage.post(StreamEnd{.epoch = 0}));
  REQUIRE(wait_for([&] { return bench.ring.size() == 2; }));
  CHECK(stage.saw_next() == std::vector<bool>{true, false});
  CHECK(bench.reported().empty());
}

// Le Smooth cherche la fin d'un dessin tenu jusqu'à cinq images derrière la tête : l'étage en tient d'autant
// plus en vol (extra_frames), et la tête attend qu'elles soient là, ou la fin du flux.
TEST_CASE("Ring : un étage qui lit quatre images en avance les attend, puis vide le flux à sa fin",
          "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 8, /*held=*/0, /*outputs=*/1, /*extra_frames=*/3);
  stage.wait_for_ahead(4);
  for (std::uint32_t id = 1; id <= 4; ++id) REQUIRE(stage.frames().try_push(frame(id)));
  // Quatre images en vol, aucune produite : la 1 attend une quatrième derrière elle.
  std::this_thread::sleep_for(20ms);
  CHECK(bench.ring.size() == 0);
  CHECK(stage.idle());
  // Sept images en vol au plus (kMaxFrames + 3).
  for (std::uint32_t id = 5; id <= 7; ++id) REQUIRE(stage.frames().try_push(frame(id)));
  REQUIRE(wait_for([&] { return bench.ring.size() == 3; }));
  // 1, 2 et 3 voient toutes celles qui attendent derrière elles ; la 4 attend.
  CHECK(stage.saw_ahead() == std::vector<std::size_t>{6, 5, 4});
  REQUIRE(stage.post(StreamEnd{.epoch = 0}));
  REQUIRE(wait_for([&] { return bench.ring.size() == 7; }));
  CHECK(stage.saw_ahead() == std::vector<std::size_t>{6, 5, 4, 3, 2, 1, 0});
  CHECK(bench.reported().empty());
}

TEST_CASE("Ring : une image d'une autre époque derrière la tête arrête l'attente des suivantes",
          "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 8, /*held=*/0, /*outputs=*/1, /*extra_frames=*/3);
  stage.wait_for_ahead(4);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(stage.frames().try_push(frame(2)));
  REQUIRE(stage.frames().try_push(frame(3, /*epoch=*/1)));
  // 1 et 2 finissent leur époque : produites sans attendre ; 3 attend les siennes.
  REQUIRE(wait_for([&] { return bench.ring.size() == 2; }));
  CHECK(stage.saw_ahead() == std::vector<std::size_t>{1, 0});
  CHECK(bench.reported().empty());
}

TEST_CASE("Ring : plus de sorties qu'une paire n'en tient arrête l'étage en le disant (P14)", "[app][ring]") {
  STATIC_REQUIRE(RingStage::kMaxOutputs == core::kMaxInstantsPerPair);
  Bench bench;
  ProbeStage stage(bench.init(), 16, /*held=*/0, RingStage::kMaxOutputs + 1);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return !bench.reported().empty(); }));
  const std::vector<Refusal> refusals = bench.reported();
  CHECK(refusals.front().key == kFatalStageKey);  // fatal, son code gardé (A31-01)
  CHECK(refusals.front().code != RefusalCode::ThreadFailed);
  CHECK(refusals.front().detail.find("11 sorties pour une image, au plus 10") != std::string::npos);
  CHECK(bench.ring.size() == 0);
}

TEST_CASE("Ring : dix sorties pour une image, toutes produites (×10)", "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 16, /*held=*/0, RingStage::kMaxOutputs);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return bench.ring.size() == RingStage::kMaxOutputs; }));
  CHECK(bench.reported().empty());
}

TEST_CASE("Ring : une sortie Same ne prend ni slot ni RingFrame, la suivante porte l'écart voulu",
          "[app][ring]") {
  Bench bench;
  ProbeStage stage(bench.init(), 8, /*held=*/0, /*outputs=*/3);
  stage.same_output(1);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return bench.ring.size() == 2; }));
  CHECK(stage.skipped() == 1);
  CHECK(bench.reported().empty());
  const std::vector<RingFrame> kept = bench.ring.all();
  REQUIRE(kept.size() == 2);
  CHECK_FALSE(kept[0].uninterpolated);
  CHECK(kept[1].uninterpolated);
}

TEST_CASE(
    "Ring : une image qui attend un slot fait vérifier l'appareil, un contexte perdu vu en 0,3 s (11.5)",
    "[app][ring][stage]") {
  Bench bench;
  ProbeStage stage(bench.init(), 1, /*held=*/0, /*outputs=*/1);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return bench.ring.size() == 1; }));
  CHECK(stage.health_checks() == 0);  // rien n'attend : pas de vérification
  // L'image 2 attend le seul slot, qui ne revient pas : l'étage interroge l'appareil.
  REQUIRE(stage.frames().try_push(frame(2)));
  REQUIRE(wait_for([&] { return stage.health_checks() >= 2; }));
  CHECK(bench.reported().empty());
  stage.lose_device(true);
  const auto lost_at = std::chrono::steady_clock::now();
  REQUIRE(wait_for([&] { return !bench.reported().empty(); }));
  // Quelques sondes (0,3 s sur un poste) ; la borne vaut pour toutes les machines, CI chargée comprise, sans
  // branche sur l'environnement (B40-08) : elle dit que la perte se voit vite, pas au bout d'un délai long.
  CHECK(std::chrono::steady_clock::now() - lost_at <= 1500ms);
  const std::vector<Refusal> refusals = bench.reported();
  // A31-01 : l'appareil perdu reste DeviceLost (« relancer le lecteur », relance), fatal par sa clé.
  CHECK(refusals.front().code == RefusalCode::DeviceLost);
  CHECK(refusals.front().key == kFatalStageKey);
  CHECK(refusals.front().detail.find("contexte perdu (sonde)") != std::string::npos);
  CHECK(bench.ring.size() == 1);
}

}  // namespace ajp::app::play
