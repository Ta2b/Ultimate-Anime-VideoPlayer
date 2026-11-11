// Le niveau de retour de présentation (chantier 6.2, app/play/present_level.h) : la table des niveaux.
#include "app/play/present_level.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "app/play/queue_lanes.h"

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
  // Un niveau nommé n'est jamais dépassé ; absent, le meilleur en dessous. wait prend present timing sans
  // date visée quand rien d'autre ne date (B10-01).
  CHECK(choose_present_level(absolute, PresentWanted::Wait) == PresentLevel::Timing);
  CHECK(choose_present_level(absolute, PresentWanted::Wayland) == PresentLevel::Open);
  CHECK(choose_present_level(wayland, PresentWanted::Wait) == PresentLevel::Wayland);
  CHECK(choose_present_level(wait, PresentWanted::Timing) == PresentLevel::Wait);
}

TEST_CASE("présentation : le niveau qui date passe devant celui qui sonde (A10-02)", "[app][present]") {
  // CachyOS, KDE Wayland, le défaut --present=wait : wp_presentation doit s'ouvrir, et passer devant
  // present_wait2 qui ne date rien (sans date, pas de grille des vblanks).
  CHECK(wants_wayland(PresentWanted::Wait));
  CHECK(wants_wayland(PresentWanted::Auto));
  CHECK(wants_wayland(PresentWanted::Timing));
  CHECK(wants_wayland(PresentWanted::Wayland));
  CHECK_FALSE(wants_wayland(PresentWanted::Open));
  const PresentCaps both{.present_id2 = true, .present_wait2 = true, .wayland_presentation = true};
  CHECK(choose_present_level(both, PresentWanted::Wait) == PresentLevel::Wayland);
  CHECK(choose_present_level(both, PresentWanted::Auto) == PresentLevel::Wayland);
  CHECK(choose_present_level(both, PresentWanted::Wayland) == PresentLevel::Wayland);
  // Sans wp_presentation (X11, Windows), le sondage reste.
  const PresentCaps wait{.present_id2 = true, .present_wait2 = true};
  CHECK(choose_present_level(wait, PresentWanted::Wait) == PresentLevel::Wait);
  // Present timing date aussi : il reste devant quand il est demandé ; à --present=wait, wp_presentation
  // passe devant lui.
  const PresentCaps all{
      .present_id2 = true, .present_wait2 = true, .timing = true, .wayland_presentation = true};
  CHECK(choose_present_level(all, PresentWanted::Auto) == PresentLevel::Timing);
  CHECK(choose_present_level(all, PresentWanted::Wait) == PresentLevel::Wayland);
}

TEST_CASE("présentation : le défaut date aussi sans wp_presentation (B10-01)", "[app][present]") {
  // Windows 11 et X11, --present=wait (le défaut) : present timing offert, pas de wp_presentation. Avant, le
  // niveau restait Wait, qui sonde sans dater : aucune grille des vblanks de toute la lecture.
  const PresentCaps windows{.present_id2 = true, .present_wait2 = true, .timing = true};
  CHECK(choose_present_level(windows, PresentWanted::Wait) == PresentLevel::Timing);
  CHECK(timing_level(choose_present_level(windows, PresentWanted::Wait)));
  // Jamais de date visée à wait : TimingAbsolute reste à auto et timing.
  const PresentCaps absolute{
      .present_id2 = true, .present_wait2 = true, .timing = true, .timing_absolute = true};
  CHECK(choose_present_level(absolute, PresentWanted::Wait) == PresentLevel::Timing);
  // Sans present timing, le sondage d'avant ; muet, Timing retombe sur lui.
  const PresentCaps wait{.present_id2 = true, .present_wait2 = true};
  CHECK(choose_present_level(wait, PresentWanted::Wait) == PresentLevel::Wait);
  CHECK(fallback_level(false) == PresentLevel::Wait);
}

TEST_CASE("présentation : present timing muet retombe (A10-06)", "[app][present]") {
  CHECK_FALSE(feedback_silent(0, 0));
  CHECK_FALSE(feedback_silent(kSilentPresents - 1, 0));
  CHECK(feedback_silent(kSilentPresents, 0));
  CHECK_FALSE(feedback_silent(10 * kSilentPresents, 1));  // un seul retour : le niveau répond
  CHECK(fallback_level(true) == PresentLevel::Wayland);
  CHECK(fallback_level(false) == PresentLevel::Wait);
  CHECK(timing_level(PresentLevel::TimingAbsolute));
  CHECK_FALSE(timing_level(PresentLevel::Wayland));
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

TEST_CASE("Horloge hôte vue par Vulkan : les tics du compteur de Windows deviennent des ns (A10-07)",
          "[app][present]") {
  using ajp::app::play::host_ticks_to_ns;
  // Linux : CLOCK_MONOTONIC, déjà en ns.
  CHECK(host_ticks_to_ns(123'456'789'012ULL, 1'000'000'000ULL) == 123'456'789'012ULL);
  // Windows : 10 MHz le plus souvent, un tic vaut 100 ns.
  CHECK(host_ticks_to_ns(10'000'000ULL, 10'000'000ULL) == 1'000'000'000ULL);
  CHECK(host_ticks_to_ns(12'345'678ULL, 10'000'000ULL) == 1'234'567'800ULL);
  // Un vieux compteur à 3 579 545 Hz, après trente jours : pas de débordement, à la ns près.
  const std::uint64_t month = 3'579'545ULL * 86'400ULL * 30ULL;
  CHECK(host_ticks_to_ns(month, 3'579'545ULL) == 86'400ULL * 30ULL * 1'000'000'000ULL);
  CHECK(host_ticks_to_ns(month + 1, 3'579'545ULL) == 86'400ULL * 30ULL * 1'000'000'000ULL + 279ULL);
  CHECK(host_ticks_to_ns(42, 0) == 42);  // fréquence illisible : tel quel
}

TEST_CASE("Domaine du compteur de Windows : dates en ns ou en tics, la plus proche de l'heure hôte (B10-02)",
          "[app][present]") {
  constexpr std::uint64_t qpc = 10'000'000ULL;           // 10 MHz
  constexpr std::int64_t now_us = 3'600'000'000;         // une heure après le démarrage
  constexpr std::uint64_t shown_ns = 3'599'990'000'000;  // affichée 10 ms plus tôt, en ns
  CHECK_FALSE(host_stamp_in_ticks(shown_ns, qpc, now_us));
  CHECK(host_stamp_in_ticks(shown_ns / 100, qpc, now_us));               // la même date en tics de 100 ns
  CHECK_FALSE(host_stamp_in_ticks(shown_ns, 1'000'000'000ULL, now_us));  // Linux : déjà des ns
  // L'aller-retour pour la date visée.
  CHECK(host_ns_to_ticks(1'234'567'800ULL, qpc) == 12'345'678ULL);
  CHECK(host_ns_to_ticks(30ULL * 86'400ULL * 1'000'000'000ULL, 3'579'545ULL) ==
        3'579'545ULL * 86'400ULL * 30ULL);
  // Tronqué au tic d'avant, jamais au suivant : la date visée n'est jamais repoussée.
  CHECK(host_ns_to_ticks(host_ticks_to_ns(987'654'321ULL, 3'579'545ULL), 3'579'545ULL) + 1 >= 987'654'321ULL);
  CHECK(host_ns_to_ticks(42, 1'000'000'000ULL) == 42);
}

TEST_CASE("Surface perdue : des acquisitions qui échouent de suite finissent par le dire (B31-15)",
          "[app][present]") {
  AcquireStreak streak;
  CHECK_FALSE(streak.failed(1'000'000));
  CHECK_FALSE(streak.failed(1'000'000 + kLostSurfaceUs - 1));
  CHECK(streak.failed(1'000'000 + kLostSurfaceUs));
  streak.reset();  // une image acquise, ou la fenêtre cachée
  CHECK_FALSE(streak.failed(9'000'000));
  CHECK_FALSE(streak.failed(9'000'000 + kLostSurfaceUs / 2));
}

TEST_CASE("Voies de calcul du backend adopté : comptées sur la famille, pas sur le pool (B10-04)",
          "[app][present]") {
  // RTX : famille de calcul à 8 files, libplacebo en emploie une, le backend la deuxième. Avant, la file de
  // libplacebo (count = 1) donnait toujours 0 voie.
  CHECK(backend_lanes(8, 1) == 2);
  CHECK(backend_lanes(3, 1) == 1);
  CHECK(backend_lanes(2, 1) == 0);
  CHECK(backend_lanes(1, 0) == 0);
  CHECK(backend_lanes(0, 1) == 0);
}

}  // namespace ajp::app::play

TEST_CASE("Instantané : l'espace de la fenêtre SDR, sRGB seulement pour une fenêtre HDR (A10-18)",
          "[app][present]") {
  using ajp::app::play::snapshot_keeps_target;
  CHECK(snapshot_keeps_target(false, false));       // fenêtre SDR : son espace, avant c'était sRGB
  CHECK_FALSE(snapshot_keeps_target(false, true));  // fenêtre HDR, fichier .ppm : sRGB
  CHECK(snapshot_keeps_target(true, true));         // fenêtre HDR, fichier .pq.ppm : PQ tel quel
}
