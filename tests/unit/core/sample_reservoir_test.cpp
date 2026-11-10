// A33-05 : les mesures au-delà de la capacité ne sont plus ignorées.
#include "core/sample_reservoir.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

namespace ajp::core {

TEST_CASE("Réservoir : sous la capacité, tout est gardé", "[core][profil]") {
  SampleReservoir<float> reservoir(8);
  for (int i = 0; i < 5; ++i) reservoir.add(static_cast<float>(i));
  CHECK(reservoir.kept().size() == 5);
  CHECK(reservoir.seen() == 5);
  CHECK(reservoir.total() == 10.0);
}

TEST_CASE("Réservoir : au-delà, la fin de la lecture compte autant que le début", "[core][profil]") {
  // Une première moitié à 1 ms, une seconde à 3 ms (la chauffe) : l'échantillon voit les deux.
  SampleReservoir<float> reservoir(1000);
  for (int i = 0; i < 100000; ++i) reservoir.add(i < 50000 ? 1.0F : 3.0F);
  CHECK(reservoir.kept().size() == 1000);
  CHECK(reservoir.seen() == 100000);
  CHECK(reservoir.total() == 200000.0);
  const auto late = std::ranges::count(reservoir.kept(), 3.0F);
  CHECK(late > 400);
  CHECK(late < 600);
}

}  // namespace ajp::core
