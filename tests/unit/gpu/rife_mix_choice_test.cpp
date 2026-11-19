// Le choix du mélange du RIFE natif et sa référence (rife_mix_choice.h), sur une chaîne factice : une sortie
// exacte plus le bruit de chaque morceau. Le bruit d'un morceau de TensorRT dépend des options des morceaux
// d'avant (à travers des blocs FP16, tout changement en amont redistribue ses arrondis) ; celui d'un morceau
// natif ne dépend que de lui. Les dB de chaque morceau f32 contre TensorRT sur la même entrée (pieces_db)
// se mesurent ailleurs : le test les donne.
#include "gpu/cuda/backend/rife_mix_choice.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace mix = kirari::gpu::cuda::rife_mix;

namespace {

constexpr std::size_t kPlane = 4096;
constexpr std::size_t kPieces = 3;  // l'encodeur, deux blocs
// Les options : 0 TensorRT, 1 natif f16 (le plus rapide), 2 natif f32 (la référence native).
constexpr int kF16 = 1;
constexpr int kF32 = 2;

double sigma_of(double db) {
  return std::pow(10.0, -db / 20.0);
}

struct Fake {
  double trt_sigma = sigma_of(43.6) / std::sqrt(3.0);  // la chaîne de TensorRT à 43,6 dB du calcul exact
  double f16_sigma = sigma_of(56.0);
  double f32_sigma = sigma_of(75.0);
  std::size_t wrong_piece = kPieces;  // un morceau natif faux : un biais sur toute sa sortie
  double wrong_bias = 0.0;
  bool nan_native = false;  // le natif f16 rend des NaN

  std::vector<float> run(const mix::Options& options) const {
    std::vector<float> out(3 * kPlane);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = 0.25F + 0.5F * static_cast<float>(i % 97) / 97.0F;
    std::uint32_t upstream = 1;
    for (std::size_t p = 0; p < options.size(); ++p) {
      const int o = options[p];
      const double sigma = o == 0 ? trt_sigma : o == kF16 ? f16_sigma : f32_sigma;
      const std::uint32_t seed =
          o == 0 ? upstream * 7919U + static_cast<std::uint32_t>(p)
                 : 100003U * static_cast<std::uint32_t>(o) + static_cast<std::uint32_t>(p);
      std::mt19937 rng(seed);
      std::normal_distribution<double> noise(0.0, sigma);
      const double bias = o != 0 && p == wrong_piece ? wrong_bias : 0.0;
      for (float& v : out) v += static_cast<float>(noise(rng) + bias);
      if (o == kF16 && nan_native) out[5] = std::numeric_limits<float>::quiet_NaN();
      upstream = upstream * 31U + static_cast<std::uint32_t>(o) + 1U;
    }
    return out;
  }
  mix::Chain chain() const {
    return [this](const mix::Options& o) -> std::expected<std::vector<float>, kirari::Refusal> {
      return run(o);
    };
  }
};

// TensorRT 1 ms par morceau, le natif f16 0,8, le natif f32 1,2.
mix::Grid times() {
  return mix::Grid(kPieces, std::vector<double>{1.0, 0.8, 1.2});
}

// Le choix complet, comme try_native : la garde, puis le mélange contre la référence qu'elle retient.
struct Outcome {
  mix::Reference reference;
  mix::Mix mix;
};

Outcome choose(const Fake& fake, const std::vector<double>& pieces_db, double min_db = 45.0) {
  const std::vector<float> trt = fake.run(mix::Options(kPieces, 0));
  const std::vector<float> exact = fake.run(mix::Options(kPieces, kF32));
  Outcome out{.reference = mix::reference_of(pieces_db, mix::group_db(exact, trt, kPlane, false), min_db)};
  const bool native = out.reference.native;
  const mix::Judge judge{.want = native ? &exact : &trt,
                         .plane = kPlane,
                         .bar_db = out.reference.bar_db,
                         .base = native ? kF32 : 0};
  auto got = mix::choose_mix(times(), judge, fake.chain());
  REQUIRE(got.has_value());
  out.mix = std::move(*got);
  return out;
}

const std::vector<double> kGoodPieces{65.0, 70.0, 72.0};

}  // namespace

TEST_CASE("RIFE natif : un TensorRT bruité laisse passer un natif juste contre le natif f32", "[rife_mix]") {
  const Fake fake;
  const Outcome old_rule = [&] {
    // L'ancienne règle : la référence TensorRT, 45 dB. Un encodeur différent du sien y redistribue les
    // arrondis des blocs et se lit au niveau de leur bruit.
    const std::vector<float> trt = fake.run(mix::Options(kPieces, 0));
    const mix::Judge judge{.want = &trt, .plane = kPlane, .bar_db = {45.0}, .base = 0};
    auto got = mix::choose_mix(times(), judge, fake.chain());
    REQUIRE(got.has_value());
    return Outcome{.mix = std::move(*got)};
  }();
  CHECK(old_rule.mix.options[0] == 0);

  const Outcome now = choose(fake, kGoodPieces);
  REQUIRE(now.reference.native);
  // La barre suit TensorRT, qui n'atteint pas 45 dB du calcul exact : au moins aussi proche que lui.
  CHECK(now.reference.bar_db.front() < 45.0);
  CHECK(now.reference.bar_db.front() > 43.0);
  CHECK(now.mix.options == mix::Options{kF16, kF16, kF16});
  CHECK(now.mix.db >= now.reference.bar_db.front());
}

TEST_CASE("RIFE natif : un natif faux ne devient jamais sa propre référence", "[rife_mix]") {
  Fake fake;
  fake.wrong_piece = 1;  // le bloc 0, f16 comme f32
  fake.wrong_bias = sigma_of(43.0);
  // La chaîne de TensorRT reste à plus de kChainFloorDb du natif f32 (le biais s'y noie dans son bruit) :
  // seul le plancher des morceaux voit le défaut, sur la même entrée.
  REQUIRE(mix::worst_psnr(fake.run(mix::Options(kPieces, kF32)), fake.run(mix::Options(kPieces, 0)), kPlane,
                          false) >= mix::kChainFloorDb);
  const Outcome now = choose(fake, {65.0, 43.0, 72.0});
  CHECK_FALSE(now.reference.native);
  CHECK(now.reference.bar_db == std::vector<double>{45.0});
  CHECK(now.mix.options[1] == 0);
  if (!mix::all_trt(now.mix.options)) {
    const std::vector<float> trt = fake.run(mix::Options(kPieces, 0));
    CHECK(mix::worst_psnr(trt, now.mix.out, kPlane, false) >= 45.0);
  }
}

TEST_CASE("RIFE natif : la garde de la référence native", "[rife_mix]") {
  SECTION("un morceau sous le plancher") {
    const auto ref = mix::reference_of({65.0, 59.9, 72.0}, {43.6}, 45.0);
    CHECK_FALSE(ref.native);
    CHECK(ref.bar_db == std::vector<double>{45.0});
  }
  SECTION("la chaîne de TensorRT trop loin du natif f32") {
    const auto ref = mix::reference_of(kGoodPieces, {39.9}, 45.0);
    CHECK_FALSE(ref.native);
    CHECK(ref.bar_db == std::vector<double>{45.0});
  }
  SECTION("des valeurs non finies ou absentes") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(mix::reference_of({65.0, nan, 72.0}, {43.6}, 45.0).native);
    CHECK_FALSE(mix::reference_of(kGoodPieces, {50.0, nan}, 45.0).native);
    CHECK_FALSE(mix::reference_of(kGoodPieces, {nan, 50.0}, 45.0).native);
    CHECK_FALSE(mix::reference_of(kGoodPieces, {}, 45.0).native);
    CHECK_FALSE(mix::reference_of({}, {50.0}, 45.0).native);
    CHECK_FALSE(mix::reference_of(kGoodPieces, {-std::numeric_limits<double>::infinity()}, 45.0).native);
  }
  SECTION("native : la barre de chaque groupe est la porte, ou TensorRT s'il ne l'atteint pas") {
    const auto ref = mix::reference_of(kGoodPieces, {49.3, 43.6}, 45.0);
    REQUIRE(ref.native);
    CHECK(ref.bar_db == std::vector<double>{45.0, 43.6});
  }
}

TEST_CASE("RIFE natif : TensorRT à 49 dB, un natif à 44 dB du calcul exact reste refusé", "[rife_mix]") {
  Fake fake;
  fake.trt_sigma = sigma_of(49.3) / std::sqrt(3.0);
  fake.f16_sigma = sigma_of(44.0 + 10.0 * std::log10(3.0));  // trois morceaux f16 : 44 dB sur la chaîne
  const Outcome now = choose(fake, kGoodPieces);
  REQUIRE(now.reference.native);
  CHECK(now.reference.bar_db == std::vector<double>{45.0});
  CHECK(now.mix.options != mix::Options{kF16, kF16, kF16});
  if (!mix::all_trt(now.mix.options)) CHECK(now.mix.db >= 45.0);
}

TEST_CASE("RIFE natif : un natif qui rend des NaN n'est jamais retenu", "[rife_mix]") {
  Fake fake;
  fake.nan_native = true;
  const Outcome now = choose(fake, kGoodPieces);
  for (const int o : now.mix.options) CHECK(o != kF16);
}
