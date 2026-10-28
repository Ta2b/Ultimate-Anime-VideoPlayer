// Rééchantillonnage séparable : tables de poids de l'hôte identiques à la référence, passes GPU
// contre ref::resize en float et en float16, bords en miroir et refus.
#include "kernels/ref/resample.h"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/weights.h"
#include "kernels/ref/half.h"
#include "kernels/spec.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

void check_resize(Rig& rig, int w, int h, int dw, int dh, k::Filter filter, Precision in, Precision out) {
  INFO(w << "x" << h << " vers " << dw << "x" << dh << ", filtre " << static_cast<int>(filter)
         << ", précisions " << static_cast<int>(in) << " → " << static_cast<int>(out));
  const k::RgbImage image = as_stored(synthetic_rgb(w, h, static_cast<std::uint32_t>(w * h + dw)), in);
  const auto want = k::ref::resize(image, dw, dh, filter);
  REQUIRE(want);
  TensorBuffer src = upload_rgb(rig, image, in);
  TensorBuffer dst =
      tensor_target(rig, static_cast<std::uint32_t>(dw), static_cast<std::uint32_t>(dh), 3, out);
  record_and_run(rig, [&](Recorder& r) { return rig.kernels().resize(r, src.view, dst.view, filter); });
  CHECK(max_abs_diff(download_rgb(rig, dst), as_stored(*want, out)) <= k::tolerance::kRgbF16);
}

}  // namespace

TEST_CASE("poids : tables de l'hôte identiques au bit près à la référence", "[vk_compute][resample]") {
  int compared = 0;
  for (const int src : {1, 2, 3, 7, 11, 64, 100, 1080}) {
    for (const int dst : {1, 2, 5, 22, 128, 1920}) {
      for (const double shift : {0.0, 0.25, -0.5, 0.5, 1.0, -1.0}) {
        for (const k::Filter filter : {k::Filter::Spline36, k::Filter::Bilinear, k::Filter::Area}) {
          INFO(src << " vers " << dst << ", décalage " << shift << ", filtre " << static_cast<int>(filter));
          const auto ours = compute_weights(src, dst, shift, filter);
          const auto want = k::ref::compute_weights(src, dst, shift, filter);
          REQUIRE(ours.has_value() == want.has_value());
          if (!want) continue;
          REQUIRE(ours->taps == static_cast<std::uint32_t>(want->taps));
          REQUIRE(ours->src == static_cast<std::uint32_t>(want->src));
          REQUIRE(ours->dst == static_cast<std::uint32_t>(want->dst));
          CHECK(ours->start == want->start);
          REQUIRE(ours->w.size() == want->w.size());
          for (std::size_t i = 0; i < want->w.size(); ++i) {
            CHECK(std::bit_cast<std::uint32_t>(ours->w[i]) == std::bit_cast<std::uint32_t>(want->w[i]));
          }
          ++compared;
        }
      }
    }
  }
  CHECK(compared > 300);
}

TEST_CASE("poids : décalages, dimensions et Area hors contrat refusés", "[vk_compute][resample]") {
  require_refusal(compute_weights(0, 4, 0.0, k::Filter::Spline36), RefusalCode::InputMismatch);
  require_refusal(compute_weights(4, k::kMaxDimension + 1, 0.0, k::Filter::Spline36),
                  RefusalCode::InputMismatch);
  require_refusal(compute_weights(4, 8, 1.5, k::Filter::Spline36), RefusalCode::FieldOutOfRange);
  require_refusal(compute_weights(4, 8, std::numeric_limits<double>::quiet_NaN(), k::Filter::Bilinear),
                  RefusalCode::FieldOutOfRange);
  require_refusal(compute_weights(4, 8, 0.0, k::Filter::Area), RefusalCode::InputMismatch);
  require_refusal(compute_weights(8, 4, 0.25, k::Filter::Area), RefusalCode::FieldOutOfRange);
}

TEST_CASE("rééchantillonnage : Spline36, bilinéaire et Area contre la référence, en float",
          "[vk_compute][resample]") {
  Rig rig = make_rig();
  check_resize(rig, 22, 14, 44, 28, k::Filter::Spline36, Precision::F32, Precision::F32);
  check_resize(rig, 22, 14, 34, 20, k::Filter::Spline36, Precision::F32, Precision::F32);
  check_resize(rig, 22, 14, 34, 20, k::Filter::Bilinear, Precision::F32, Precision::F32);
  check_resize(rig, 44, 28, 22, 14, k::Filter::Spline36, Precision::F32, Precision::F32);
  check_resize(rig, 44, 28, 22, 14, k::Filter::Area, Precision::F32, Precision::F32);
  check_resize(rig, 64, 36, 22, 14, k::Filter::Area, Precision::F32, Precision::F32);
  check_resize(rig, 30, 18, 30, 18, k::Filter::Spline36, Precision::F32, Precision::F32);
}

TEST_CASE("rééchantillonnage : bords en miroir sur de petites images", "[vk_compute][resample]") {
  Rig rig = make_rig();
  // 2 × 2 vers 8 × 8 : chaque prise de Spline36 sort de l'image et revient par le miroir.
  check_resize(rig, 2, 2, 8, 8, k::Filter::Spline36, Precision::F32, Precision::F32);
  check_resize(rig, 2, 2, 6, 4, k::Filter::Bilinear, Precision::F32, Precision::F32);
  check_resize(rig, 8, 8, 2, 2, k::Filter::Spline36, Precision::F32, Precision::F32);
  check_resize(rig, 8, 6, 2, 2, k::Filter::Area, Precision::F32, Precision::F32);
}

TEST_CASE("rééchantillonnage : tenseurs float16 en entrée, en sortie ou des deux côtés",
          "[vk_compute][resample]") {
  Rig rig = make_rig();
  require_f16(rig);
  check_resize(rig, 22, 14, 44, 28, k::Filter::Spline36, Precision::F16, Precision::F16);
  check_resize(rig, 22, 14, 44, 28, k::Filter::Spline36, Precision::F16, Precision::F32);
  check_resize(rig, 44, 28, 22, 14, k::Filter::Area, Precision::F32, Precision::F16);
}

TEST_CASE("rééchantillonnage : tailles impaires et plans hors contrat refusés", "[vk_compute][resample]") {
  Rig rig = make_rig();
  TensorBuffer even = tensor_target(rig, 22, 14, 3, Precision::F32);
  TensorBuffer odd = tensor_target(rig, 21, 14, 3, Precision::F32);
  TensorBuffer big = tensor_target(rig, 44, 28, 3, Precision::F32);
  TensorBuffer one = tensor_target(rig, 44, 28, 1, Precision::F32);
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_refusal(rig.kernels().resize(r, odd.view, big.view, k::Filter::Spline36),
                  RefusalCode::InputMismatch);
  require_refusal(rig.kernels().resize(r, big.view, odd.view, k::Filter::Spline36),
                  RefusalCode::InputMismatch);
  require_refusal(rig.kernels().resize(r, even.view, big.view, k::Filter::Area), RefusalCode::InputMismatch);
  require_refusal(rig.kernels().resize(r, even.view, one.view, k::Filter::Spline36),
                  RefusalCode::InputMismatch);
  require_refusal(rig.kernels().resize(r, even.view, big.view, static_cast<k::Filter>(5)),
                  RefusalCode::InputMismatch);
  CHECK(require_value(r.finish())->dispatches() == 0);
  // Un seul plan se rééchantillonne aussi.
  const std::array plane{synthetic_plane(22, 14, 3, 0.0F, 1.0F)};
  TensorBuffer src = upload_planes(rig, plane, Precision::F32);
  TensorBuffer dst = tensor_target(rig, 44, 28, 1, Precision::F32);
  record_and_run(
      rig, [&](Recorder& rec) { return rig.kernels().resize(rec, src.view, dst.view, k::Filter::Spline36); });
  const auto want = k::ref::resize(plane[0], 44, 28, k::Filter::Spline36);
  REQUIRE(want);
  CHECK(max_abs_diff(download_planes(rig, dst)[0], *want) <= k::tolerance::kRgbF16);
}
