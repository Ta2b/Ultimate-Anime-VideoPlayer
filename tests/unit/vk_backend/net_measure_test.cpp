// Mesures communes des runtimes de fabricant (net_measure.h), sans appareil : l'image de test
// (taille, bornes, structure d'image plutôt que bruit, plans distincts, décalage d'une entrée à
// l'autre), la médiane qui écarte un lancement retardé, l'écart entre deux sorties, et le choix de
// l'unité par tolérance et justesse, tel qu'OpenVINO (GPU, NPU) et Core ML (trois répartitions)
// l'appellent.
#include "gpu/vulkan/backend/net_measure.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ajp::gpu::vk::net;

namespace {

// Écart moyen entre voisins horizontaux d'un plan de width colonnes.
double neighbour_step(const std::vector<float>& image, std::size_t width, std::size_t pixels) {
  double sum = 0.0;
  std::size_t count = 0;
  for (std::size_t i = 0; i + 1 < pixels; ++i) {
    if ((i + 1) % width == 0) continue;
    sum += std::abs(static_cast<double>(image[i + 1]) - static_cast<double>(image[i]));
    ++count;
  }
  return sum / static_cast<double>(count);
}

}  // namespace

TEST_CASE("Mesure des runtimes : image de test", "[vk_backend][measure]") {
  constexpr std::int64_t kHeight = 96;
  constexpr std::int64_t kWidth = 160;
  constexpr std::array<std::int64_t, 4> kDims{1, 3, kHeight, kWidth};
  const std::vector<float> image = probe_image(kDims);
  constexpr auto kPixels = static_cast<std::size_t>(kHeight * kWidth);
  REQUIRE(image.size() == 3 * kPixels);
  for (const float v : image) {
    REQUIRE(v >= 0.0F);
    REQUIRE(v <= 1.0F);
  }
  CHECK(probe_image(kDims) == image);

  SECTION("une image, pas du bruit : des aplats, des traits et des bords francs") {
    // Le motif haché d'avant changeait de valeur à chaque pixel (écart moyen ~1/3).
    CHECK(neighbour_step(image, kWidth, kPixels) < 0.1);
    std::size_t lines = 0;
    std::size_t edges = 0;
    for (std::size_t i = 0; i + 1 < kPixels; ++i) {
      if (image[i] < 0.05F) ++lines;
      if (std::abs(image[i + 1] - image[i]) > 0.15F) ++edges;
    }
    CHECK(lines > kPixels / 20);
    CHECK(edges > kPixels / 20);
  }
  SECTION("les plans diffèrent") {
    const std::vector<float> first(image.begin(), image.begin() + kPixels);
    const std::vector<float> second(image.begin() + kPixels, image.begin() + (2 * kPixels));
    CHECK(first != second);
  }
  SECTION("shift décale l'image de 2 pixels vers la droite") {
    const std::vector<float> moved = probe_image(kDims, 1);
    CHECK(moved != image);
    for (std::size_t y = 0; y < static_cast<std::size_t>(kHeight); ++y) {
      for (std::size_t x = 0; x + 2 < static_cast<std::size_t>(kWidth); ++x) {
        const std::size_t at = (y * kWidth) + x;
        REQUIRE(moved[at + 2] == image[at]);
      }
    }
  }
  SECTION("moins de deux dimensions : une valeur médiane ; dimension nulle : rien") {
    CHECK(probe_image(std::array<std::int64_t, 1>{4}) == std::vector<float>(4, 0.5F));
    CHECK(probe_image({}) == std::vector<float>{0.5F});
    CHECK(probe_image(std::array<std::int64_t, 4>{1, 3, 0, 8}).empty());
    CHECK(probe_image(std::array<std::int64_t, 2>{-1, 8}).empty());
  }
}

TEST_CASE("Mesure des runtimes : médiane et écart", "[vk_backend][measure]") {
  CHECK(median(std::vector<double>{}) == 0.0);
  CHECK(median(std::vector<double>{3.0}) == 3.0);
  CHECK(median(std::vector<double>{4.0, 1.0, 3.0}) == 3.0);
  CHECK(median(std::vector<double>{4.0, 1.0, 3.0, 2.0}) == Catch::Approx(2.5));
  // Un lancement retardé par l'hôte ne déplace pas la médiane.
  std::vector<double> runs(kUnitTimedRuns, 10.0);
  runs[3] = 80.0;
  CHECK(median(runs) == 10.0);

  CHECK(mean_error(std::vector<float>{}, std::vector<float>{}) == 0.0);
  CHECK(mean_error(std::vector<float>{0.5F, 0.25F}, std::vector<float>{0.5F, 0.75F}) == Catch::Approx(0.25));
  CHECK(std::isinf(mean_error(std::vector<float>{1.0F}, std::vector<float>{1.0F, 2.0F})));
}

TEST_CASE("Mesure des runtimes : choix de l'unité", "[vk_backend][measure]") {
  using Measured = std::optional<UnitMeasure>;
  constexpr std::size_t kGpu = 0;
  constexpr std::size_t kNpu = 1;

  SECTION("OpenVINO : le NPU jusqu'à 1,25 fois le GPU") {
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 12.5, .error = 0.004}}}, kNpu) ==
          kNpu);
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 12.6, .error = 0.004}}}, kNpu) ==
          kGpu);
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 6.0, .error = 0.004}}}, kNpu) ==
          kNpu);
  }
  SECTION("une unité trop imprécise est écartée, même rapide") {
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 5.0, .error = 0.02}}}, kNpu) ==
          kGpu);
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 5.0, .error = kUnitMaxMeanError}}},
                      kNpu) == kNpu);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(choose_unit(std::array{Measured{{.ms = 10.0}}, Measured{{.ms = 5.0, .error = nan}}}, kNpu) == kGpu);
  }
  SECTION("Core ML : la plus rapide des justes quand le Neural Engine est trop lent ou absent") {
    constexpr std::size_t kAne = 2;
    const std::array slow{Measured{{.ms = 9.0}}, Measured{{.ms = 7.0, .error = 0.001}},
                          Measured{{.ms = 9.0, .error = 0.001}}};
    CHECK(choose_unit(slow, kAne) == 1U);
    const std::array missing{Measured{{.ms = 9.0}}, Measured{{.ms = 7.0, .error = 0.001}}, Measured{}};
    CHECK(choose_unit(missing, kAne) == 1U);
    const std::array close{Measured{{.ms = 9.0}}, Measured{{.ms = 7.0, .error = 0.001}},
                           Measured{{.ms = 8.0, .error = 0.001}}};
    CHECK(choose_unit(close, kAne) == kAne);
    // La référence en échec : la deuxième répartition l'est devenue.
    const std::array no_reference{Measured{}, Measured{{.ms = 7.0}}, Measured{{.ms = 8.0, .error = 0.001}}};
    CHECK(choose_unit(no_reference, kAne) == kAne);
  }
  SECTION("rien de juste : aucun choix") {
    CHECK_FALSE(choose_unit(std::array<Measured, 2>{}, kNpu));
    CHECK_FALSE(choose_unit(std::array{Measured{{.ms = 1.0, .error = 1.0}}}, 0));
    CHECK(choose_unit(std::array{Measured{{.ms = 4.0}}}, 5) == 0U);
  }
}
