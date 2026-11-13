// Le grade avant le warp (source_grade.h) dans l'étage gpu, sur le banc sans GPU (upscale_rig.h) : un grade
// par SPAN, aucun pour un dessin tenu, et les images marquées gradées pour l'écran (RingFrame::graded).
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/source_grade.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;
namespace {

enum class Answer { Graded, Nothing, Refused };

class CountingGrader final : public SourceGrader {
 public:
  explicit CountingGrader(Answer answer) : answer_(answer) {}
  std::expected<bool, Refusal> grade(gpu::KernelOps& /*ops*/, const gpu::Tensor& /*rgb*/,
                                     bool /*debanded*/) override {
    ++calls;
    if (answer_ == Answer::Refused)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "essai"});
    return answer_ == Answer::Graded;
  }
  std::atomic<int> calls{0};

 private:
  Answer answer_;
};

// RIFE mouvement ×2, warp après SPAN, six sources dont 3 tient le dessin de 2 : SPAN sur 1, 2, 4, 5 et 6.
std::vector<RingFrame> play(Rig& rig, std::shared_ptr<SourceGrader> grader, bool warp_after_span = true) {
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = warp_after_span;
  s.grader = std::move(grader);
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(10));  // six sources, une image entre chaque paire hors du dessin tenu
  rig.stage.reset();
  return rig.sink.frames();
}

}  // namespace

TEST_CASE("Étage gpu : grade avant le warp, un grade par SPAN et toutes les images gradées", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == static_cast<int>(rig.backend.count("engine span")));
  CHECK(grader->calls == 5);  // le dessin tenu reprend le tampon gradé de 2
  for (const RingFrame& frame : frames) CHECK(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp, rien à grader, l'écran garde le grade", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Nothing);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == 5);
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp refusé, l'étage cesse de grader et la lecture continue",
          "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Refused);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == 1);
  CHECK(rig.reported().empty());
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp sans warp après SPAN, l'écran grade", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  const std::vector<RingFrame> frames = play(rig, grader, false);
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

}  // namespace ajp::app::play
