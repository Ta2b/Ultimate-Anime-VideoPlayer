// État de la barre de lecture, sans ImGui ni fenêtre : visibilité, glissement sur la barre,
// volume, clics sur l'image, messages brefs et choix de recomposer l'interface.
#include "app/play/controls.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace ajp::app::play {
namespace {

using Catch::Matchers::WithinAbs;

constexpr std::int64_t kMs = 1'000;
constexpr std::int64_t kS = 1'000'000;

TEST_CASE("Le temps s'affiche comme dans un lecteur", "[app][controls]") {
  CHECK(format_clock(0, 24 * 60 * kS) == "0:00");
  CHECK(format_clock(65 * kS + 900 * kMs, 24 * 60 * kS) == "1:05");
  CHECK(format_clock(23 * 60 * kS + 59 * kS, 24 * 60 * kS) == "23:59");
  CHECK(format_clock(65 * kS, 2 * 3600 * kS) == "0:01:05");  // même forme que la durée
  CHECK(format_clock(3723 * kS, 2 * 3600 * kS) == "1:02:03");
  CHECK(format_clock(kNoPts, 0) == "--:--");
  CHECK(format_clock(-1, 0) == "--:--");
}

TEST_CASE("Les sous-titres du bas montent au-dessus de la barre visible", "[app][controls]") {
  CHECK(subtitle_lift(1400.0F, 1440.0F, 190.0F) == 150.0F);
  CHECK(subtitle_lift(1200.0F, 1440.0F, 190.0F) == 0.0F);  // déjà au-dessus
  CHECK(subtitle_lift(1400.0F, 1440.0F, 0.0F) == 0.0F);    // barre cachée
}

TEST_CASE("Fraction de la barre et temps se répondent, bornés", "[app][controls]") {
  CHECK(fraction_of(30 * kS, 120 * kS) == 0.25);
  CHECK(fraction_of(200 * kS, 120 * kS) == 1.0);
  CHECK(fraction_of(kNoPts, 120 * kS) == 0.0);
  CHECK(fraction_of(10 * kS, 0) == 0.0);  // durée inconnue
  CHECK(time_at(0.5, 120 * kS) == 60 * kS);
  CHECK(time_at(-0.2, 120 * kS) == 0);
  CHECK(time_at(1.7, 120 * kS) == 120 * kS);
  CHECK(time_at(0.5, 0) == 0);
}

TEST_CASE("La barre paraît au mouvement de la souris et s'efface après l'inactivité", "[app][controls]") {
  Controls controls;
  CHECK(controls.opacity(0, false) == 0.0F);  // jamais montrée
  controls.activity(10 * kS);
  CHECK(controls.opacity(10 * kS, false) == 0.0F);  // début du fondu
  CHECK_THAT(controls.opacity(10 * kS + kFadeUs / 2, false), WithinAbs(0.5, 1e-3));
  CHECK(controls.opacity(10 * kS + kFadeUs, false) == 1.0F);
  CHECK(controls.opacity(10 * kS + kHideAfterUs, false) == 1.0F);
  CHECK_THAT(controls.opacity(10 * kS + kHideAfterUs + kFadeUs / 2, false), WithinAbs(0.5, 1e-3));
  CHECK(controls.opacity(10 * kS + kHideAfterUs + kFadeUs, false) == 0.0F);
  CHECK(controls.cursor_hidden(10 * kS + kHideAfterUs + kFadeUs, false));
  CHECK_FALSE(controls.cursor_hidden(10 * kS + kHideAfterUs, false));
}

TEST_CASE("Un mouvement pendant l'affichage prolonge la barre sans refaire le fondu", "[app][controls]") {
  Controls controls;
  controls.activity(0);
  controls.activity(kS);
  CHECK(controls.opacity(kS, false) == 1.0F);
  CHECK(controls.opacity(kS + kHideAfterUs - 1, false) == 1.0F);
  CHECK(controls.opacity(kS + kHideAfterUs + kFadeUs, false) == 0.0F);
}

TEST_CASE("OLED : en pause, la barre s'efface après 20 s sans activité, sauf sur un menu",
          "[app][controls]") {
  Controls controls;
  controls.set_oled_care(true);
  controls.activity(0);
  CHECK(controls.opacity(kPausedHideUs - 1, true) == 1.0F);
  CHECK_THAT(controls.opacity(kPausedHideUs + kFadeUs / 2, true), WithinAbs(0.5, 1e-3));
  CHECK(controls.opacity(kPausedHideUs + kFadeUs, true) == 0.0F);
  CHECK(controls.cursor_hidden(kPausedHideUs + kFadeUs, true));
  controls.set_held(true);  // menu ouvert : la barre reste
  CHECK(controls.opacity(kPausedHideUs + kFadeUs, true) == 1.0F);
  controls.set_held(false);
  controls.activity(30 * kS);  // la souris bouge : elle revient
  CHECK(controls.opacity(30 * kS + kFadeUs, true) == 1.0F);
  controls.set_oled_care(false);  // hors OLED, la pause la garde
  CHECK(controls.opacity(30 * kS + kPausedHideUs + kFadeUs, true) == 1.0F);
}

TEST_CASE("La barre reste en pause, sur un menu ou pendant un glissement", "[app][controls]") {
  Controls controls;
  controls.activity(0);
  const std::int64_t later = kHideAfterUs + 10 * kS;
  CHECK(controls.opacity(later, false) == 0.0F);
  CHECK(controls.opacity(later, true) == 1.0F);  // pause
  controls.set_held(true);
  CHECK(controls.opacity(later, false) == 1.0F);
  controls.set_held(false);
  controls.press(0.3);
  CHECK(controls.opacity(later, false) == 1.0F);
  static_cast<void>(controls.release(0.3, 100 * kS));
  CHECK(controls.opacity(later, false) == 0.0F);
}

TEST_CASE("La souris qui quitte la fenêtre efface la barre sans attendre", "[app][controls]") {
  Controls controls;
  controls.activity(0);
  controls.pointer_left(kS);
  CHECK(controls.opacity(kS + kFadeUs, false) == 0.0F);
  CHECK_FALSE(controls.cursor_hidden(kS + kFadeUs, false));  // le curseur n'est plus à nous
  controls.activity(2 * kS);                                 // retour : nouveau fondu
  CHECK(controls.opacity(2 * kS, false) == 0.0F);
  CHECK(controls.opacity(2 * kS + kFadeUs, false) == 1.0F);
}

TEST_CASE("Glisser sur la barre fait des seeks rapides espacés, le lâcher un seek exact", "[app][controls]") {
  Controls controls;
  const std::int64_t duration = 100 * kS;
  controls.press(0.25);
  CHECK(controls.dragging());
  CHECK(controls.drag_fraction() == 0.25);

  auto first = controls.drag_to(0.3, duration, 0);
  REQUIRE(first);
  CHECK(first->target_us == 30 * kS);
  CHECK_FALSE(first->exact);
  CHECK_FALSE(controls.drag_to(0.35, duration, kDragSeekUs - 1));  // trop tôt
  CHECK(controls.drag_fraction() == 0.35);                         // la tête suit quand même
  CHECK_FALSE(controls.drag_to(0.3, duration, kDragSeekUs));       // même cible que le dernier
  auto second = controls.drag_to(0.4, duration, kDragSeekUs);
  REQUIRE(second);
  CHECK(second->target_us == 40 * kS);

  const cmd::Seek last = controls.release(0.42, duration);
  CHECK(last.target_us == 42 * kS);
  CHECK(last.exact);
  CHECK_FALSE(controls.dragging());
  CHECK_FALSE(controls.drag_fraction());
}

TEST_CASE("Sans durée connue, la barre ne seek pas en glissant", "[app][controls]") {
  Controls controls;
  CHECK_FALSE(controls.drag_to(0.5, 0, 0));
  CHECK(controls.release(0.5, 0).target_us == 0);
}

TEST_CASE("Volume : bornes, pas réguliers, muet", "[app][controls]") {
  Controls controls;
  CHECK(controls.volume() == 1.0F);
  CHECK(controls.set_volume(3.0F).gain == kMaxVolume);
  CHECK(controls.set_volume(-1.0F).gain == 0.0F);
  CHECK_THAT(controls.set_volume(0.83F).gain, WithinAbs(0.83, 1e-6));
  CHECK_THAT(controls.step_volume(kVolumeStep).gain, WithinAbs(0.9, 1e-5));  // retombe sur un pas
  CHECK_THAT(controls.step_volume(-kVolumeStep).gain, WithinAbs(0.85, 1e-5));

  CHECK(controls.toggle_mute().gain == 0.0F);
  CHECK(controls.muted());
  CHECK_THAT(controls.volume(), WithinAbs(0.85, 1e-5));  // gardé pendant le muet
  CHECK_THAT(controls.toggle_mute().gain, WithinAbs(0.85, 1e-5));
  controls.toggle_mute();
  CHECK_THAT(controls.step_volume(kVolumeStep).gain, WithinAbs(0.9, 1e-5));  // régler quitte le muet
  CHECK_FALSE(controls.muted());
}

TEST_CASE("Clic sur l'image : pause après le délai, double clic plein écran sans pause", "[app][controls]") {
  Controls controls;
  CHECK_FALSE(controls.click(0, 1));
  CHECK_FALSE(controls.pending_click(kDoubleClickUs - 1));
  CHECK(controls.pending_click(kDoubleClickUs) == Controls::VideoClick::TogglePause);
  CHECK_FALSE(controls.pending_click(kDoubleClickUs + 1));  // une seule fois

  CHECK_FALSE(controls.click(kS, 1));
  CHECK(controls.click(kS + 200 * kMs, 2) == Controls::VideoClick::ToggleFullscreen);
  CHECK_FALSE(controls.pending_click(kS + kDoubleClickUs * 2));  // le simple clic n'a pas eu lieu
}

TEST_CASE("Message bref : visible puis effacé en fondu", "[app][controls]") {
  Controls controls;
  CHECK_FALSE(controls.toast(0));
  controls.notify("Volume 80 %", kS);
  REQUIRE(controls.toast(kS));
  CHECK(*controls.toast(kS) == "Volume 80 %");
  CHECK(controls.toast_opacity(kS) == 1.0F);
  CHECK_THAT(controls.toast_opacity(kS + kToastUs - kFadeUs / 2), WithinAbs(0.5, 1e-3));
  CHECK_FALSE(controls.toast(kS + kToastUs));
}

TEST_CASE("L'interface se recompose seulement quand elle change", "[app][controls]") {
  Controls controls;
  ControlsView view{.position_us = 10 * kS, .duration_us = 100 * kS};
  CHECK_FALSE(controls.needs_frame(0, view, false));  // rien de visible, rien dessiné

  controls.activity(kS);
  CHECK(controls.needs_frame(kS, view, true));
  controls.framed(kS, view, true);
  CHECK_FALSE(controls.needs_frame(kS + kUiFrameUs - 1, view, true));  // au plus ~60 Hz
  CHECK(controls.needs_frame(kS + kUiFrameUs, view, false));           // fondu en cours

  const std::int64_t steady = kS + kFadeUs;
  controls.framed(steady, view, true);
  CHECK_FALSE(controls.needs_frame(steady + kUiFrameUs, view, false));  // rien n'a bougé
  view.position_us += kPositionStepUs;
  CHECK(controls.needs_frame(steady + kUiFrameUs, view, false));  // la position a avancé d'un pas
  controls.framed(steady + kUiFrameUs, view, true);
  view.paused = true;
  CHECK(controls.needs_frame(steady + 2 * kUiFrameUs, view, false));
  view.paused = false;
  controls.framed(steady + 2 * kUiFrameUs, view, true);

  const std::int64_t gone = kS + kHideAfterUs + kFadeUs;
  CHECK(controls.needs_frame(gone, view, false));  // une dernière fois pour l'ôter de l'écran
  controls.framed(gone, view, false);
  CHECK_FALSE(controls.needs_frame(gone + kS, view, false));
}

}  // namespace
}  // namespace ajp::app::play
