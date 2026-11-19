// La référence du débandage de la source (kernels/ref/deband.h) : celle du noyau CUDA et du shader Vulkan.
#include "kernels/ref/deband.h"

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

using namespace kirari::kernels;

TEST_CASE("Débandage : le tirage reste dans le rayon et ne dépend que du pixel", "[kernels][deband]") {
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const ref::DebandOffset o = ref::deband_offset(x, y);
      CHECK(o.dx * o.dx + o.dy * o.dy <= 17 * 17);
      const ref::DebandOffset again = ref::deband_offset(x, y);
      CHECK((again.dx == o.dx && again.dy == o.dy));
    }
  }
  // Le hachage, au mot près de deband_kernels.cuh : quelques valeurs fixées.
  CHECK(ref::deband_uniform(0, 0, 1) >= 0.0F);
  CHECK(ref::deband_uniform(0, 0, 1) < 1.0F);
  CHECK(ref::deband_uniform(3, 5, 1) != ref::deband_uniform(5, 3, 1));
  CHECK(ref::deband_uniform(3, 5, 1) != ref::deband_uniform(3, 5, 2));
}

TEST_CASE("Débandage : la règle prend la moyenne sous le seuil, garde la valeur au-delà",
          "[kernels][deband]") {
  CHECK(ref::deband_rule(0.5F, 0.501F, 0.499F, 0.502F, 0.498F) ==
        0.25F * (0.501F + 0.499F + 0.502F + 0.498F));
  CHECK(ref::deband_rule(0.5F, 0.6F, 0.6F, 0.6F, 0.6F) == 0.5F);
}

TEST_CASE("Débandage : un dégradé en paliers se lisse, un trait net reste", "[kernels][deband]") {
  constexpr int kW = 64;
  constexpr int kH = 32;
  RgbImage image(kW, kH);
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      // Paliers de 1/1000 tous les 8 pixels, sous le seuil ; un trait vertical de 0,2 en x = 40.
      const float step = 0.3F + (0.001F * static_cast<float>(x / 8));
      for (PlaneF& p : image.planes) p.at(x, y) = x == 40 ? 0.5F : step;
    }
  }
  const auto out = ref::deband(image);
  REQUIRE(out);
  int smoothed = 0;
  for (int y = 0; y < kH; ++y) {
    CHECK(out->planes[0].at(40, y) == 0.5F);  // le trait ne prend aucune moyenne
    for (int x = 0; x < kW; ++x) {
      if (x == 40) continue;
      const float before = image.planes[1].at(x, y);
      const float after = out->planes[1].at(x, y);
      CHECK(std::fabs(after - before) <= ref::kDebandThreshold);
      if (after != before) ++smoothed;
    }
  }
  CHECK(smoothed > kW * kH / 2);
  // Une image impaire ou vide est refusée comme ailleurs dans la référence.
  CHECK_FALSE(ref::deband(RgbImage(0, 0)));
}
