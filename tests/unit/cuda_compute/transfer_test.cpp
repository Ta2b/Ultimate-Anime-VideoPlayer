// Fonctions de transfert et primaires sur l'appareil, contre kernels/ref/transfer.h. powf, logf
// et expf de CUDA s'écartent de quelques ulp de la bibliothèque C : chaque côté tient
// tolerance::kTransferRelative de la formule exacte, d'où une borne double entre eux.
#include "gpu/cuda/compute/transfer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "kernels/ref/transfer.h"
#include "kernels/spec/tolerance.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Rgb;

namespace {

constexpr int kW = 67;
constexpr int kH = 33;

struct Case {
  const char* name;
  TransferSpec spec;
  float lo;  // entrée
  float hi;
  float peak;  // crête de la sortie, pour la borne absolue
};

// La référence pour un cas : la même chaîne que le noyau, fonction par fonction.
Rgb reference(const TransferSpec& s, Rgb c) {
  const kernels::ref::Bt1886Display display =
      kernels::ref::bt1886_display(s.bt1886_white, s.bt1886_black).value_or(kernels::ref::Bt1886Display{});
  const auto decode = [&s, &display](float v) {
    switch (s.decode) {
      case Curve::Srgb:
        return kernels::ref::srgb_eotf(v);
      case Curve::Bt1886:
        return kernels::ref::bt1886_eotf(v, display);
      case Curve::Pq:
        return kernels::ref::pq_eotf(v);
      case Curve::Hlg:
        return kernels::ref::hlg_inverse_oetf(v);
      case Curve::Linear:
        break;
    }
    return v;
  };
  const auto encode = [&s, &display](float v) {
    switch (s.encode) {
      case Curve::Srgb:
        return kernels::ref::srgb_inverse_eotf(v);
      case Curve::Bt1886:
        return kernels::ref::bt1886_inverse_eotf(v, display);
      case Curve::Pq:
        return kernels::ref::pq_inverse_eotf(v);
      case Curve::Hlg:
        return kernels::ref::hlg_oetf(v);
      case Curve::Linear:
        break;
    }
    return v;
  };
  c = {.r = decode(c.r), .g = decode(c.g), .b = decode(c.b)};
  if (s.hlg_ootf_peak) c = kernels::ref::hlg_ootf(c, *kernels::ref::hlg_display(*s.hlg_ootf_peak));
  c = {.r = c.r * s.gain, .g = c.g * s.gain, .b = c.b * s.gain};
  if (s.primaries) c = kernels::ref::convert_primaries(c, *s.primaries);
  return {.r = encode(c.r), .g = encode(c.g), .b = encode(c.b)};
}

const Case kCases[] = {
    {"EOTF PQ", {.decode = Curve::Pq}, 0.0F, 1.0F, 10000.0F},
    {"PQ inverse", {.encode = Curve::Pq}, 0.0F, 10000.0F, 1.0F},
    {"EOTF sRGB", {.decode = Curve::Srgb}, -0.1F, 1.1F, 1.0F},
    {"sRGB inverse", {.encode = Curve::Srgb}, -0.1F, 1.1F, 1.0F},
    {"EOTF BT.1886",
     {.decode = Curve::Bt1886, .bt1886_white = 100.0F, .bt1886_black = 0.1F},
     0.0F,
     1.0F,
     100.0F},
    {"BT.1886 inverse",
     {.encode = Curve::Bt1886, .bt1886_white = 100.0F, .bt1886_black = 0.1F},
     0.0F,
     100.0F,
     1.0F},
    {"HLG inverse", {.decode = Curve::Hlg}, 0.0F, 1.0F, 1.0F},
    {"OETF HLG", {.encode = Curve::Hlg}, 0.0F, 1.0F, 1.0F},
    {"HLG vers PQ", {.decode = Curve::Hlg, .encode = Curve::Pq, .hlg_ootf_peak = 1000.0F}, 0.0F, 1.0F, 1.0F},
    {"BT.709 vers BT.2020 en PQ",
     {.decode = Curve::Bt1886,
      .encode = Curve::Pq,
      .bt1886_white = 1.0F,
      .gain = 203.0F,
      .primaries = kernels::kBt709ToBt2020},
     0.0F,
     1.0F,
     1.0F},
};

}  // namespace

TEST_CASE("les fonctions de transfert suivent la référence", "[cuda][transfer]") {
  AJP_CUDA_RIG(rig);
  const Case c = GENERATE(from_range(kCases));
  CAPTURE(c.name);
  const kernels::RgbImage in = random_rgb(kW, kH, c.lo, c.hi, 101U);
  const DeviceTensor din = upload_rgb(rig, in, Elem::F32);
  DeviceTensor out = alloc_tensor(rig, Elem::F32, kW, kH);
  auto params = make_transfer(c.spec);
  REQUIRE(params);
  REQUIRE(transfer(din.tensor, *params, out.tensor, rig.s()));
  const kernels::RgbImage got = download_rgb(rig, out);
  double worst = 0.0;
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      const Rgb e = reference(c.spec, in.at(x, y));
      const Rgb g = got.at(x, y);
      for (const auto& [ge, ee] : {std::pair{g.r, e.r}, std::pair{g.g, e.g}, std::pair{g.b, e.b}}) {
        const double bound = 2.0 * (kernels::tolerance::kTransferRelative * std::fabs(ee) +
                                    kernels::tolerance::kTransferAbsolute * c.peak);
        worst = std::max(worst, std::fabs(static_cast<double>(ge) - ee) / bound);
        REQUIRE(std::fabs(static_cast<double>(ge) - ee) <= bound);
      }
    }
  }
  INFO("pire écart, en part de la borne : " << worst);
  CHECK(worst <= 1.0);
}

TEST_CASE("les fonctions de transfert refusent un écran invalide", "[cuda][transfer]") {
  CHECK(make_transfer({.hlg_ootf_peak = 300.0F}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(make_transfer({.decode = Curve::Bt1886, .bt1886_white = 0.1F, .bt1886_black = 0.1F}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(make_transfer({.gain = std::nanf("")}).error().code == RefusalCode::FieldOutOfRange);
  const auto hlg = make_transfer({.hlg_ootf_peak = 2000.0F});
  REQUIRE(hlg);
  const auto display = kernels::ref::hlg_display(2000.0F);
  REQUIRE(display);
  CHECK(hlg->hlg_gamma == display->gamma);
}
