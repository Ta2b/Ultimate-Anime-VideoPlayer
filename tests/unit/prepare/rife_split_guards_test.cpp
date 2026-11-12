// Les garde-fous de la découpe de RIFE contre un graphe qui lui ressemble sans être le même : colle
// entrelacée d'un autre ordre de canaux (C04-01), Resize avec antialias (C04-02).
#include <catch2/catch_test_macros.hpp>

#include "prepare/rife_split.h"
#include "rife_fake.h"

namespace ajp::prepare {
namespace {

TEST_CASE("C04-01 : une colle entrelacée d'un autre ordre de canaux est refusée", "[prepare][rife_fused]") {
  Fake lite;
  lite.reduce = {32, 16, 8, 4, 1};
  lite.casts = true;
  lite.interleaved = true;
  REQUIRE(analyze_rife(fake_rife(lite)).has_value());
  lite.interleaved_order_swapped = true;  // le GridSample warpe Concat(images A et B, encodages)
  CHECK_FALSE(analyze_rife(fake_rife(lite)).has_value());
}

TEST_CASE("C04-02 : un Resize avec antialias est refusé (RIFE)", "[prepare][rife_fused]") {
  REQUIRE(analyze_rife(fake_rife(Fake{})).has_value());
  Fake filtered;
  filtered.antialias = true;
  CHECK_FALSE(analyze_rife(fake_rife(filtered)).has_value());
}

}  // namespace
}  // namespace ajp::prepare
