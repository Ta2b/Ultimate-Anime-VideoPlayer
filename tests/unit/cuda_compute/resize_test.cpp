// Tables de poids et rééchantillonnage séparable sur l'appareil, contre la référence CPU.
#include "gpu/cuda/compute/resize.h"

#include <cmath>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "kernels/ref/half.h"
#include "kernels/ref/resample.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Filter;

namespace {

struct Case {
  int sw, sh, dw, dh;
  Filter filter;
};

constexpr Case kCases[] = {
    {96, 54, 144, 82, Filter::Spline36},   // agrandissement non entier
    {160, 90, 64, 36, Filter::Spline36},   // réduction : le noyau s'étire
    {160, 90, 64, 36, Filter::Bilinear},   //
    {160, 90, 64, 36, Filter::Area},       //
    {162, 94, 54, 94, Filter::Area},       // réduction d'un seul côté
    {2, 2, 16, 16, Filter::Spline36},      // le miroir sert à chaque prise
    {66, 38, 66, 38, Filter::Spline36},    // même taille
    {128, 64, 130, 66, Filter::Bilinear},  // presque la même taille
};

}  // namespace

TEST_CASE("les tables de poids égalent celles de la référence", "[cuda][resize]") {
  for (const Filter f : {Filter::Spline36, Filter::Bilinear, Filter::Area}) {
    for (const auto& [src, dst] :
         {std::pair{1, 1}, std::pair{7, 3}, std::pair{960, 1920}, std::pair{1920, 960}, std::pair{1080, 720},
          std::pair{33, 32}, std::pair{2, 16384}}) {
      for (const double shift : {0.0, 0.25, -0.5}) {
        CAPTURE(f, src, dst, shift);
        const auto ours = compute_weights(src, dst, shift, f);
        const auto ref = kernels::ref::compute_weights(src, dst, shift, f);
        REQUIRE(ours.has_value() == ref.has_value());
        if (!ours) continue;
        CHECK(ours->taps == ref->taps);
        CHECK(ours->start == ref->start);
        CHECK(ours->w == ref->w);
      }
    }
  }
}

TEST_CASE("les tables de poids refusent ce que refuse la référence", "[cuda][resize]") {
  CHECK(compute_weights(0, 4, 0.0, Filter::Spline36).error().code == RefusalCode::ShapeInvalid);
  CHECK(compute_weights(4, 16385, 0.0, Filter::Spline36).error().code == RefusalCode::ShapeInvalid);
  CHECK(compute_weights(4, 8, 1.5, Filter::Spline36).error().code == RefusalCode::FieldOutOfRange);
  CHECK(compute_weights(4, 8, std::nan(""), Filter::Bilinear).error().code == RefusalCode::FieldOutOfRange);
  CHECK(compute_weights(4, 8, 0.0, Filter::Area).error().code == RefusalCode::ShapeInvalid);
  CHECK(compute_weights(8, 4, 0.25, Filter::Area).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("le rééchantillonnage égale la référence au bit", "[cuda][resize]") {
  AJP_CUDA_RIG(rig);
  const Case c = GENERATE(from_range(kCases));
  const Elem in_elem = GENERATE(Elem::F32, Elem::F16);
  const Elem out_elem = GENERATE(Elem::F32, Elem::F16);
  CAPTURE(c.sw, c.sh, c.dw, c.dh, c.filter, in_elem, out_elem);
  kernels::RgbImage image = random_rgb(c.sw, c.sh, -0.1F, 1.1F, 3U);
  if (in_elem == Elem::F16) image = kernels::ref::round_to_half(image);
  const DeviceTensor in = upload_rgb(rig, image, in_elem);
  DeviceTensor out = alloc_tensor(rig, out_elem, c.dw, c.dh);
  auto plan = ResizePlan::create(
      *rig.pool,
      {.src_width = c.sw, .src_height = c.sh, .dst_width = c.dw, .dst_height = c.dh, .filter = c.filter},
      rig.s());
  REQUIRE(plan);
  REQUIRE(plan->run(in.tensor, out.tensor, rig.s()));
  auto expected = kernels::ref::resize(image, c.dw, c.dh, c.filter);
  REQUIRE(expected);
  if (out_elem == Elem::F16) expected = kernels::ref::round_to_half(*expected);
  CHECK(max_diff(download_rgb(rig, out), *expected) == 0.0F);
}

TEST_CASE("le rééchantillonnage refuse les tailles hors contrat", "[cuda][resize]") {
  AJP_CUDA_RIG(rig);
  const auto refused = [&](ResizeSpec spec) {
    auto plan = ResizePlan::create(*rig.pool, spec, rig.s());
    REQUIRE_FALSE(plan);
    return plan.error().code;
  };
  CHECK(refused({.src_width = 63, .src_height = 64, .dst_width = 128, .dst_height = 128}) ==
        RefusalCode::ShapeInvalid);
  CHECK(refused({.src_width = 64, .src_height = 64, .dst_width = 0, .dst_height = 128}) ==
        RefusalCode::ShapeInvalid);
  CHECK(
      refused(
          {.src_width = 64, .src_height = 64, .dst_width = 128, .dst_height = 128, .filter = Filter::Area}) ==
      RefusalCode::ShapeInvalid);
  CHECK(refused({.src_width = 64, .src_height = 64, .dst_width = 32, .dst_height = 32, .channels = 0}) ==
        RefusalCode::ShapeInvalid);

  auto plan = ResizePlan::create(
      *rig.pool, {.src_width = 64, .src_height = 32, .dst_width = 32, .dst_height = 16}, rig.s());
  REQUIRE(plan);
  DeviceTensor in = alloc_tensor(rig, Elem::F16, 64, 32);
  DeviceTensor out = alloc_tensor(rig, Elem::F16, 32, 16);
  CHECK(plan->run(out.tensor, in.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
  DeviceTensor four = alloc_tensor(rig, Elem::F16, 64, 32, 4);
  CHECK(plan->run(four.tensor, out.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
}
