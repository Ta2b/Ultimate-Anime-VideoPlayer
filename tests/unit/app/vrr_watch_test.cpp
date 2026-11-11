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

TEST_CASE("NVAPI : le bit « engagé » n'est exigé qu'une fois vu posé (B11-02)", "[app][display]") {
  // Bit 0 seul : G-SYNC permis. Un pilote qui ne pose jamais le bit 4 laissait « non armé » pour toujours.
  CHECK(nv_vrr_armed(kNvVrrEnabled, false));
  CHECK_FALSE(nv_vrr_armed(0, false));
  CHECK_FALSE(nv_vrr_armed(kNvVrrInMode, true));  // engagé sans être permis : incohérent, non
  // Le bit 4 déjà vu posé : il dit l'engagement.
  CHECK(nv_vrr_armed(kNvVrrEnabled | kNvVrrInMode, true));
  CHECK_FALSE(nv_vrr_armed(kNvVrrEnabled, true));
}

}  // namespace ajp::app::play
