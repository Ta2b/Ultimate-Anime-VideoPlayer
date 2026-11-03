#include "app/play/vrr_watch.h"

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

}  // namespace ajp::app::play
