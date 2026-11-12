#include "app/play/vrr_watch.h"

#include <optional>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("VRR relu : un changement n'est retenu qu'après deux lectures égales", "[app][display]") {
  VrrWatch watch(true);
  CHECK_FALSE(watch.observe(true));
  CHECK_FALSE(watch.observe(false));  // une lecture isolée ne change rien
  CHECK_FALSE(watch.observe(true));   // elle est oubliée
  CHECK_FALSE(watch.observe(false));
  const auto changed = watch.observe(false);
  REQUIRE(changed);
  CHECK_FALSE(*changed);
  CHECK_FALSE(watch.armed());
  CHECK_FALSE(watch.observe(std::nullopt));  // lecture absente : ignorée
  CHECK_FALSE(watch.observe(false));
}

TEST_CASE("VRR relu : present timing prime sur l'ioctl et bascule tout de suite", "[app][present]") {
  VrrWatch watch(false);
  // L'ioctl dit armé une fois : à confirmer ; present timing dit armé : retenu à la première lecture.
  CHECK(watch.observe(std::optional<bool>(true), std::optional<bool>(false)) == std::optional<bool>(true));
  CHECK(watch.armed());
  // Même état : rien ; present timing contredit l'ioctl : present timing gagne.
  CHECK_FALSE(watch.observe(std::optional<bool>(true), std::optional<bool>(false)));
  CHECK(watch.observe(std::optional<bool>(false), std::optional<bool>(true)) == std::optional<bool>(false));
  // Sans present timing : l'ioctl, avec sa confirmation.
  CHECK_FALSE(watch.observe(std::nullopt, std::optional<bool>(true)));
  CHECK(watch.observe(std::nullopt, std::optional<bool>(true)) == std::optional<bool>(true));
}

TEST_CASE("NVAPI : permis n'est pas engagé tant que le bit « engagé » n'est pas vu (C11-08)",
          "[app][display]") {
  // Bit 0 seul : G-SYNC permis, peut-être « plein écran seulement » sur une fenêtre. Avant, « armé » : le
  // plancher et l'Auto se réglaient pour un VRR qui n'était pas là. Inconnu : la présentation datée le dit.
  CHECK_FALSE(nv_vrr_armed(kNvVrrEnabled, false).has_value());
  CHECK(nv_vrr_armed(0, false) == std::optional<bool>(false));
  CHECK(nv_vrr_armed(kNvVrrInMode, true) == std::optional<bool>(false));  // engagé sans être permis : non
  // Le bit 4 posé dit l'engagement ; déjà vu dans ce processus, son absence dit « non engagé ».
  CHECK(nv_vrr_armed(kNvVrrEnabled | kNvVrrInMode, false) == std::optional<bool>(true));
  CHECK(nv_vrr_armed(kNvVrrEnabled | kNvVrrInMode, true) == std::optional<bool>(true));
  CHECK(nv_vrr_armed(kNvVrrEnabled, true) == std::optional<bool>(false));
  // Une lecture inconnue ne change rien à l'état suivi.
  VrrWatch watch(false);
  CHECK_FALSE(watch.observe(nv_vrr_armed(kNvVrrEnabled, false)).has_value());
  CHECK_FALSE(watch.observe(nv_vrr_armed(kNvVrrEnabled, false)).has_value());
  CHECK_FALSE(watch.armed());
}

}  // namespace ajp::app::play
