// Le retour de présentation (app/play/placebo_feedback.h) sans GPU (C10-11) : une source de retours
// wp_presentation simulée, et present timing sans appareil, qui retombe aussitôt.
#include "app/play/placebo_feedback.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/logger.h>

#include "app/play/placebo_wayland.h"
#include "app/play/present_level.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t k144HzNs = 6'944'444;
constexpr std::int64_t k144HzUs = 6'944;

// Le compositeur simulé : chaque demande de retour est notée ; le test dit ce qui revient.
class FakeWayland final : public WaylandPresentation {
 public:
  void request(std::uint64_t id) override { requested.push_back(id); }
  void read(std::vector<WaylandDone>& out) override {
    out.insert(out.end(), coming.begin(), coming.end());
    coming.clear();
  }
  std::vector<std::uint64_t> requested;
  std::vector<WaylandDone> coming;
};

std::shared_ptr<spdlog::logger> quiet() {
  return std::make_shared<spdlog::logger>("test-feedback");
}

struct WaylandRig {
  FakeWayland* fake = nullptr;
  std::optional<PresentFeedback> feedback;
  std::int64_t now_us = 1'000'000;
  WaylandRig() {
    auto source = std::make_unique<FakeWayland>();
    fake = source.get();
    feedback.emplace(nullptr, PresentLevel::Wayland, PresentCaps{.wayland_presentation = true}, quiet(),
                     std::move(source));
  }
  // count présentations à 144 Hz ; chacune revient au tour suivant avec ce que dit le compositeur.
  void present(int count, std::uint32_t refresh_ns, std::uint32_t flags) {
    for (int i = 0; i < count; ++i) {
      feedback->prepare_wayland(now_us);
      const std::uint64_t id = fake->requested.back();
      fake->coming.push_back(WaylandDone{.id = id,
                                         .visible_us = now_us + 2 * k144HzUs,
                                         .refresh_ns = refresh_ns,
                                         .msc = id,
                                         .flags = flags});
      now_us += k144HzUs;
    }
    feedback->prepare_wayland(now_us);  // relit le dernier retour
  }
};

}  // namespace

TEST_CASE("Retour wp_presentation : chaque présentation demande son retour", "[app][present]") {
  WaylandRig rig;
  rig.present(3, k144HzNs, kWaylandKindVsync);
  REQUIRE(rig.fake->requested.size() == 4);
  CHECK(rig.fake->requested[0] == 1);
  CHECK(rig.fake->requested[3] == 4);
  CHECK(rig.feedback->last_id() == 4);
  CHECK(rig.feedback->last_read().id == 3);
  CHECK(rig.feedback->dated());
}

TEST_CASE("Retour wp_presentation : le passage en VRR retire la grille (C10-01)", "[app][present]") {
  WaylandRig rig;
  // En fenêtre, 144 Hz fixes : la période est dite et l'état change une fois.
  rig.present(10, k144HzNs, kWaylandKindVsync);
  CHECK(rig.feedback->refresh_state() == RefreshState::Fixed);
  CHECK(rig.feedback->fixed_refresh());
  CHECK(rig.feedback->refresh_duration_ns() == k144HzNs);
  CHECK(rig.feedback->take_refresh_change());
  CHECK_FALSE(rig.feedback->take_refresh_change());
  // Plein écran, KWin arme le VRR : refresh = 0. Avant, la période de 144 Hz restait et l'état Fixed aussi.
  rig.present(10, 0, kWaylandKindVsync);
  CHECK(rig.feedback->refresh_state() == RefreshState::Unknown);
  CHECK_FALSE(rig.feedback->fixed_refresh());
  CHECK(rig.feedback->refresh_duration_ns() == 0);
  CHECK(rig.feedback->take_refresh_change());
  CHECK(rig.feedback->clock().nominal_period_us() == 0.0);
  // Retour en fenêtre : fixe de nouveau.
  rig.present(10, k144HzNs, kWaylandKindVsync);
  CHECK(rig.feedback->fixed_refresh());
  CHECK(rig.feedback->take_refresh_change());
}

TEST_CASE("Retour wp_presentation : une image hors vblank ne fait pas une fréquence fixe (C10-01)",
          "[app][present]") {
  WaylandRig rig;
  rig.present(5, k144HzNs, 0x6);  // horloge matérielle et zéro copie, sans vsync
  CHECK(rig.feedback->refresh_state() == RefreshState::Unknown);
  CHECK_FALSE(rig.feedback->take_refresh_change());
}

TEST_CASE("Retour wp_presentation : une image écartée par le compositeur n'est pas datée", "[app][present]") {
  WaylandRig rig;
  rig.feedback->prepare_wayland(rig.now_us);
  rig.fake->coming.push_back(WaylandDone{.id = 1, .visible_us = 0});
  rig.feedback->prepare_wayland(rig.now_us + k144HzUs);
  CHECK(rig.feedback->last_read().id == 0);
  CHECK(rig.feedback->refresh_state() == RefreshState::Unknown);
}

#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
TEST_CASE("Retour de présentation : present timing sans fonctions retombe sur un niveau offert (C10-03)",
          "[app][present]") {
  // Sans appareil, les fonctions de present timing manquent : le niveau retombe dès la construction.
  const PresentCaps timing_only{.present_id2 = true, .timing = true};
  const PresentFeedback alone(nullptr, PresentLevel::Timing, timing_only, quiet());
  CHECK(alone.level() == PresentLevel::Open);
  const PresentCaps with_wait{.present_id2 = true, .present_wait2 = true, .timing = true};
  const PresentFeedback waits(nullptr, PresentLevel::Timing, with_wait, quiet());
  CHECK(waits.level() == PresentLevel::Wait);
  const PresentFeedback dated(nullptr, PresentLevel::Timing, timing_only, quiet(),
                              std::make_unique<FakeWayland>());
  CHECK(dated.level() == PresentLevel::Wayland);
}
#endif

}  // namespace ajp::app::play
