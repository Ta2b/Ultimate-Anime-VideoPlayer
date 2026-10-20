// L'appel de DistilDRBA pour un instant de la paire (kernels/spec/distil_drba.h) : les fenêtres
// [idx − 0,5 ; idx + 0,5) d'infer.py, et u dans ]0, 1[ comme rife_input l'accepte.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

TEST_CASE("DistilDRBA : centré sur a avant le milieu, sur b à partir du milieu", "[kernels][drba]") {
  // t = 1/4 : tau = 1,25 autour de a.
  CHECK_FALSE(drba_call(0.25F).on_b);
  CHECK_THAT(drba_call(0.25F).u, WithinAbs(0.625, 1e-6));
  // t = 1/2 : la fenêtre de b commence, tau = 0,5.
  CHECK(drba_call(0.5F).on_b);
  CHECK_THAT(drba_call(0.5F).u, WithinAbs(0.25, 1e-6));
  // t = 3/4 : tau = 0,75 autour de b.
  CHECK(drba_call(0.75F).on_b);
  CHECK_THAT(drba_call(0.75F).u, WithinAbs(0.375, 1e-6));
  for (const float t : {0.01F, 0.1F, 0.333F, 0.499F, 0.5F, 0.667F, 0.9F, 0.99F}) {
    INFO("t = " << t);
    CHECK(drba_call(t).u > 0.0F);
    CHECK(drba_call(t).u < 1.0F);
  }
}
