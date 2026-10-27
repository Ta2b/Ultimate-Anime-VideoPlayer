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

 private:
  mutable std::mutex mutex_;
  std::vector<RingFrame> frames_;
};

// Une sortie par image (ou `outputs` sorties) ; la première sortie de l'image `held` attend
// release(). outputs() note, par image, si la suivante était déjà là.
class ProbeStage final : public RingStage {
 public:
  ProbeStage(VideoStageInit init, std::uint32_t slots, std::uint32_t held, std::uint32_t outputs)
      : RingStage(std::move(init), slots, 0), held_(held), outputs_(outputs) {
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
  void wait_for_next(bool on) { waits_.store(on); }
  // Pour chaque image passée par outputs(), dans l'ordre : vrai si next() la suivait.
  std::vector<bool> saw_next() const {
    const std::scoped_lock lock(mutex_);
    return saw_next_;
  }

 private:
  bool wants_next() const override { return waits_.load(); }
  std::expected<std::uint32_t, Refusal> outputs(const DecodedFrame& /*frame*/) override {
    const std::scoped_lock lock(mutex_);
    saw_next_.push_back(next() != nullptr);
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
  std::atomic<bool> waits_{false};
  std::vector<bool> saw_next_;
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

DecodedFrame frame(std::uint32_t id) {
  return DecodedFrame{
      .id = SourceId{id}, .width = 64, .height = 48, .pts_us = static_cast<std::int64_t>(id) * 40'000};
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

TEST_CASE("Ring : plus de sorties qu'une paire n'en tient arrête l'étage en le disant (P14)", "[app][ring]") {
  STATIC_REQUIRE(RingStage::kMaxOutputs == core::kMaxInstantsPerPair);
  Bench bench;
  ProbeStage stage(bench.init(), 16, /*held=*/0, RingStage::kMaxOutputs + 1);
  REQUIRE(stage.frames().try_push(frame(1)));
  REQUIRE(wait_for([&] { return !bench.reported().empty(); }));
  const std::vector<Refusal> refusals = bench.reported();
  CHECK(refusals.front().code == RefusalCode::ThreadFailed);
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

}  // namespace ajp::app::play
