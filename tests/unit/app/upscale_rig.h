// Le banc de l'étage gpu des tests [app][stage] : l'étage UpscaleStage sur le backend factice (stage_rig.h),
// ses moteurs factices et les réglages en direct, et la forme des images reçues.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/live_settings.h"
#include "app/play/options.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "stage_rig.h"

namespace ajp::app::play::upscale_test {

inline constexpr int kWidth = 320;
inline constexpr int kHeight = 180;
inline constexpr std::int64_t kPeriodUs = 40'000;

struct Rig : test::StageRig {
  test::FakeEngine span{"span", {"input"}, {"output"}};
  test::FakeEngine rife{"rife", {"input"}, {"output"}};
  test::FakeEngine drba{"drba", {"pair01", "pair12"}, {"output"}};
  test::FakeEngine drba_motion{"drba_motion", {"pair01", "pair12"}, {"flow", "mask"}};
  test::FakeEngine motion{"rife_motion", {"input"}, {"flow", "mask"}};
  std::shared_ptr<LiveSettings> live = std::make_shared<LiveSettings>();

  // SPAN toujours ; RIFE ×num/den par les réglages en direct quand with_rife.
  UpscaleSetup setup(bool with_rife, Factor factor = {.num = 2, .den = 1}, std::uint32_t slot_count = 4) {
    live->set_rife_factor(with_rife ? factor : Factor{.num = 1, .den = 1});
    return UpscaleSetup{.backend = &backend,
                        .engine = &span,
                        .rife = with_rife ? &rife : nullptr,
                        .live = live,
                        .source_width = kWidth,
                        .source_height = kHeight,
                        .slots = slots(slot_count, kWidth * kSpanFactor, kHeight * kSpanFactor)};
  }
  void start(UpscaleSetup s, const MasterClock* clock = nullptr) {
    auto started = UpscaleStage::start(init(clock), std::move(s));
    if (!started) FAIL("refus " << name(started.error().code) << " : " << started.error().detail);
    stage = std::move(*started);
  }
  // Pousse les images id = first … last, à kPeriodUs d'écart, dans l'époque donnée.
  void feed(std::uint32_t first, std::uint32_t last, std::uint32_t epoch = 0) {
    for (std::uint32_t id = first; id <= last; ++id)
      REQUIRE(push(test::host_frame(id, kWidth, kHeight, static_cast<std::int64_t>(id) * kPeriodUs, epoch)));
  }
};

// « 3 » : la source 3 telle quelle ; « 2-3 » : une image entre les sources 2 et 3.
inline std::vector<std::string> shape_of(const std::vector<RingFrame>& frames) {
  std::vector<std::string> out;
  for (const RingFrame& f : frames) {
    out.push_back(f.left == f.right ? std::to_string(f.left.value)
                                    : std::to_string(f.left.value) + "-" + std::to_string(f.right.value));
  }
  return out;
}

inline std::vector<std::int64_t> pts_of(const std::vector<RingFrame>& frames) {
  std::vector<std::int64_t> out;
  for (const RingFrame& f : frames) out.push_back(f.pts_us);
  return out;
}

}  // namespace ajp::app::play::upscale_test
