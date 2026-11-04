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

}  // namespace ajp::app::play
