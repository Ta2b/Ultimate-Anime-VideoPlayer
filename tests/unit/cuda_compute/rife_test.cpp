// Entrée et sortie de RIFE sur l'appareil, contre kernels/ref/rife.h : remplissage virtuel en
// noir studio, conversion propre à RIFE, plans constants, instant, recadrages.
#include "gpu/cuda/compute/rife.h"

#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "gpu/cuda/compute/yuv.h"
#include "kernels/ref/half.h"
#include "kernels/ref/rife.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Format;
using kernels::Range;

namespace {

struct Size {
  int width;
  int height;
};

// Remplissage impair à droite et en bas, remplissage pair des deux côtés, aucun remplissage.
constexpr Size kSizes[] = {{100, 50}, {130, 66}, {128, 64}};

kernels::PlaneF rounded(const kernels::PlaneF& p, Elem elem) {
  if (elem == Elem::F32) return p;
  kernels::PlaneF out = p;
  for (float& v : out.data) v = kernels::ref::round_to_half(v);
  return out;
}

}  // namespace

TEST_CASE("l'entrée de RIFE égale rife_input de la référence au bit", "[cuda][rife]") {
  AJP_CUDA_RIG(rig);
  const Size size = GENERATE(from_range(kSizes));
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::Yuv444p16);
  const Range range = GENERATE(Range::Limited, Range::Full);
  const Elem elem = GENERATE(Elem::F16, Elem::F32);
  CAPTURE(size.width, size.height, format, range, elem);
  const kernels::YuvImage a = random_yuv(format, size.width, size.height, 41U);
  const kernels::YuvImage b = random_yuv(format, size.width, size.height, 43U);
  const DeviceYuv da = upload_yuv(rig, a);
  const DeviceYuv db = upload_yuv(rig, b);
  const kernels::RifeGeometry g = kernels::rife_geometry(size.width, size.height);

  auto plan =
      IngestPlan::create(*rig.pool, rife_ingest_spec(format, size.width, size.height, range), rig.s());
  REQUIRE(plan);
  CHECK(plan->output_width() == g.padded_width);
  CHECK(plan->output_height() == g.padded_height);
  DeviceTensor input = alloc_tensor(rig, elem, g.padded_width, g.padded_height, kernels::kRifeInputChannels);
  REQUIRE(plan->run(da.planes, channels(input.tensor, kernels::kRifeChannelA, 3), rig.s()));
  REQUIRE(plan->run(db.planes, channels(input.tensor, kernels::kRifeChannelB, 3), rig.s()));
  REQUIRE(rife_constants(input.tensor, rig.s()));
  REQUIRE(rife_time(input.tensor, 0.375F, rig.s()));

  const auto pa = kernels::ref::rife_prepare(a, range);
  const auto pb = kernels::ref::rife_prepare(b, range);
  REQUIRE(pa);
  REQUIRE(pb);
  const auto expected = kernels::ref::rife_input(*pa, *pb, 0.375F);
  REQUIRE(expected);
  const std::vector<kernels::PlaneF> got = download_planes(rig, input);
  REQUIRE(got.size() == expected->size());
  for (std::size_t c = 0; c < got.size(); ++c) {
    CAPTURE(c);
    CHECK(max_diff(got[c], rounded((*expected)[c], elem)) == 0.0F);
  }
}

TEST_CASE("la sortie de RIFE égale rife_crop et rife_finish de la référence", "[cuda][rife]") {
  AJP_CUDA_RIG(rig);
  const Size size = GENERATE(from_range(kSizes));
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::Yuv444p16);
  CAPTURE(size.width, size.height, format);
  const kernels::RifeGeometry g = kernels::rife_geometry(size.width, size.height);
  const kernels::RgbImage padded = random_rgb(g.padded_width, g.padded_height, -0.05F, 1.05F, 47U);
  const DeviceTensor dp = upload_rgb(rig, padded, Elem::F32);

  // RGB : la fenêtre seule, en F16.
  DeviceTensor crop = alloc_tensor(rig, Elem::F16, size.width, size.height);
  REQUIRE(rife_crop(dp.tensor, crop.tensor, rig.s()));
  const auto expected_rgb = kernels::ref::rife_crop(padded, g);
  REQUIRE(expected_rgb);
  CHECK(max_diff(download_rgb(rig, crop), kernels::ref::round_to_half(*expected_rgb)) == 0.0F);

  // YUV : conversion de RIFE sur l'image remplie, puis la fenêtre.
  auto plan = OutputPlan::create(*rig.pool, rife_output_spec(format, Range::Limited, g), rig.s());
  REQUIRE(plan);
  DeviceYuv out = alloc_yuv(rig, format, size.width, size.height);
  REQUIRE(plan->run(dp.tensor, out.planes, rig.s()));
  const auto expected_yuv = kernels::ref::rife_finish(padded, format, Range::Limited, g);
  REQUIRE(expected_yuv);
  CHECK(download_yuv(rig, out) == *expected_yuv);
}

TEST_CASE("l'entrée de RIFE refuse les tailles et instants hors contrat", "[cuda][rife]") {
  AJP_CUDA_RIG(rig);
  DeviceTensor input = alloc_tensor(rig, Elem::F16, 128, 64, kernels::kRifeInputChannels);
  for (const float t : {-0.1F, 1.5F, std::numeric_limits<float>::quiet_NaN()}) {
    CHECK(rife_time(input.tensor, t, rig.s()).error().code == RefusalCode::FieldOutOfRange);
  }
  DeviceTensor unaligned = alloc_tensor(rig, Elem::F16, 100, 64, kernels::kRifeInputChannels);
  CHECK(rife_constants(unaligned.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
  DeviceTensor three = alloc_tensor(rig, Elem::F16, 128, 64, 3);
  CHECK(rife_constants(three.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
  // La sortie remplie doit mesurer rife_geometry de la fenêtre.
  DeviceTensor out = alloc_tensor(rig, Elem::F16, 100, 70);
  CHECK(rife_crop(three.tensor, out.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
}
