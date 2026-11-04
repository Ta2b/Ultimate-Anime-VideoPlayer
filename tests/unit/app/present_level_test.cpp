// Le niveau de retour de présentation (chantier 6.2, app/play/present_level.h) : la table des niveaux.
#include "app/play/present_level.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("présentation : table des niveaux selon ce qui est offert et demandé", "[app][present]") {
  const PresentCaps none{};
  const PresentCaps wayland{.wayland_presentation = true};
  const PresentCaps wait{.present_id2 = true, .present_wait2 = true};
  const PresentCaps timing{.present_id2 = true, .present_wait2 = true, .timing = true};
  const PresentCaps absolute{
      .present_id2 = true, .present_wait2 = true, .timing = true, .timing_absolute = true};
  const PresentCaps timing_without_id{.timing = true, .timing_absolute = true};

  // open : la présentation d'aujourd'hui, quoi qu'il soit offert.
  CHECK(choose_present_level(absolute, PresentWanted::Open) == PresentLevel::Open);
  // auto : le meilleur niveau.
  CHECK(choose_present_level(none, PresentWanted::Auto) == PresentLevel::Open);
  CHECK(choose_present_level(wayland, PresentWanted::Auto) == PresentLevel::Wayland);
  CHECK(choose_present_level(wait, PresentWanted::Auto) == PresentLevel::Wait);
  CHECK(choose_present_level(timing, PresentWanted::Auto) == PresentLevel::Timing);
  CHECK(choose_present_level(absolute, PresentWanted::Auto) == PresentLevel::TimingAbsolute);
  // present timing sans present_id2 : rien à relier aux images.
  CHECK(choose_present_level(timing_without_id, PresentWanted::Auto) == PresentLevel::Open);
  // Un niveau nommé n'est jamais dépassé ; absent, le meilleur en dessous.
  CHECK(choose_present_level(absolute, PresentWanted::Wait) == PresentLevel::Wait);
  CHECK(choose_present_level(absolute, PresentWanted::Wayland) == PresentLevel::Open);
  CHECK(choose_present_level(wayland, PresentWanted::Wait) == PresentLevel::Wayland);
  CHECK(choose_present_level(wait, PresentWanted::Timing) == PresentLevel::Wait);
}

TEST_CASE("présentation : --present et la forme de la swapchain", "[app][present]") {
  CHECK(parse_present("auto") == PresentWanted::Auto);
  CHECK(parse_present("timing") == PresentWanted::Timing);
  CHECK(parse_present("open") == PresentWanted::Open);
  CHECK_FALSE(parse_present("vite").has_value());
  CHECK(swapchain_shape(PresentLevel::Open).min_images == 0);
  CHECK(swapchain_shape(PresentLevel::Timing).min_images == 4);
  CHECK(swapchain_shape(PresentLevel::Timing).depth == 2);
  CHECK(name(PresentLevel::TimingAbsolute) == "timing-absolute");
}

TEST_CASE("présentation : l'état VRR lu par present timing", "[app][present]") {
  CHECK(vrr_state(0, 0) == RefreshState::Unknown);
  CHECK(vrr_state(6'068'740, UINT64_MAX) == RefreshState::Variable);
  CHECK(vrr_state(6'068'740, 6'068'740) == RefreshState::Fixed);
  CHECK(vrr_state(6'068'740, 12'137'480) == RefreshState::Unknown);  // ni l'un ni l'autre : l'ioctl décide
}

}  // namespace ajp::app::play
