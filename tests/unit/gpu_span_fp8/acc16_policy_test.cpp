// La règle de l'accumulateur f16 du SPAN FP8 face à ses débordements (acc16_policy.h).
#include "gpu/span_fp8/acc16_policy.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {

TEST_CASE("acc16 : des débordements rares gardent le f16", "[span_fp8][acc16]") {
  Acc16Policy policy;
  std::int64_t overflowed = 0;
  // Une image débordée tous les 130 lancements : jamais trois dans une fenêtre de 240.
  for (int i = 1; i <= 2000; ++i) {
    if (i % 130 == 0) ++overflowed;
    CHECK(policy.poll(overflowed) == Acc16Change::None);
  }
  CHECK(policy.f16());
}

TEST_CASE("acc16 : trois débordements rapprochés passent au f32, puis le f16 revient", "[span_fp8][acc16]") {
  Acc16Policy policy;
  CHECK(policy.poll(0) == Acc16Change::None);
  CHECK(policy.poll(1) == Acc16Change::None);
  CHECK(policy.poll(2) == Acc16Change::None);
  CHECK(policy.poll(3) == Acc16Change::ToF32);
  CHECK_FALSE(policy.f16());
  // Les débordements vus en retard pendant le f32 ne relancent rien.
  std::int64_t launches = 4;
  CHECK(policy.poll(4) == Acc16Change::None);
  ++launches;
  for (; launches < 4 + Acc16Policy::kCooldown - 1; ++launches) CHECK(policy.poll(4) == Acc16Change::None);
  CHECK(policy.poll(4) == Acc16Change::ToF16);
  CHECK(policy.f16());
  // De retour en f16, il faut de nouveau trois images débordées pour repasser au f32.
  CHECK(policy.poll(5) == Acc16Change::None);
  CHECK(policy.poll(6) == Acc16Change::None);
  CHECK(policy.poll(7) == Acc16Change::ToF32);
}

TEST_CASE("acc16 : trois images débordées vues d'un coup comptent toutes", "[span_fp8][acc16]") {
  Acc16Policy policy;
  CHECK(policy.poll(3) == Acc16Change::ToF32);
}

TEST_CASE("acc16 : trois débordements étalés au-delà de la fenêtre gardent le f16", "[span_fp8][acc16]") {
  Acc16Policy policy;
  std::int64_t overflowed = 0;
  for (std::int64_t i = 1; i <= 3 * Acc16Policy::kWindow; ++i) {
    if (i % (Acc16Policy::kWindow / 2 + 1) == 0) ++overflowed;
    CHECK(policy.poll(overflowed) == Acc16Change::None);
  }
}

}  // namespace ajp::gpu::span_fp8
