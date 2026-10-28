// Fonctions de transfert et primaires : le noyau en float contre la formule en double de
// kernels/spec/transfer.h (|f − d| ≤ rel |d| + abs × crête), la variante float16 contre la
// référence float rangée en float16.
#include "kernels/ref/transfer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

constexpr float kHlgPeak = 600.0F;
constexpr float kWhite = 100.0F;
constexpr float kBlack = 0.1F;

struct Op {
  TransferOp op;
  const char* name;
  double peak;  // crête de la sortie, pour la borne absolue
  float lo;     // domaine d'entrée parcouru (au-delà de ses bornes : l'entrée y est ramenée)
  float hi;
};

const std::array kOps{
    Op{TransferOp::PqEotf, "PQ EOTF", 10000.0, -0.1F, 1.1F},
    Op{TransferOp::PqInverseEotf, "PQ inverse", 1.0, -10.0F, 10500.0F},
    Op{TransferOp::HlgOetf, "HLG OETF", 1.0, -0.1F, 1.1F},
    Op{TransferOp::HlgInverseOetf, "HLG inverse", 1.0, -0.1F, 1.1F},
    Op{TransferOp::HlgOotf, "HLG OOTF", kHlgPeak, -0.05F, 1.0F},
    Op{TransferOp::SrgbEotf, "sRGB EOTF", 1.0, -0.1F, 1.1F},
    Op{TransferOp::SrgbInverseEotf, "sRGB inverse", 1.0, -0.1F, 1.1F},
    Op{TransferOp::Bt1886Eotf, "BT.1886 EOTF", kWhite, -0.1F, 1.1F},
    Op{TransferOp::Bt1886InverseEotf, "BT.1886 inverse", 1.0, 0.0F, 110.0F},
    Op{TransferOp::Bt709ToBt2020, "709 vers 2020", 1.0, -0.1F, 1.1F},
    Op{TransferOp::Bt2020ToBt709, "2020 vers 709", 1.0, -0.1F, 1.1F},
};

double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

double pq_eotf(double e) {
  namespace pq = k::pq;
  const double p = std::pow(clamp(e, 0, 1), 1.0 / pq::kM2);
  return std::pow(std::max(p - pq::kC1, 0.0) / (pq::kC2 - pq::kC3 * p), 1.0 / pq::kM1) * pq::kPeakNits;
}

double pq_inverse(double nits) {
  namespace pq = k::pq;
  const double ym = std::pow(clamp(nits, 0, pq::kPeakNits) / pq::kPeakNits, pq::kM1);
  return std::pow((pq::kC1 + pq::kC2 * ym) / (1.0 + pq::kC3 * ym), pq::kM2);
}

double hlg_oetf(double e) {
  namespace hlg = k::hlg;
  e = clamp(e, 0, 1);
  return e <= hlg::kKnee ? std::sqrt(3.0 * e) : hlg::kA * std::log(12.0 * e - hlg::kB) + hlg::kC;
}

double hlg_inverse(double v) {
  namespace hlg = k::hlg;
  v = clamp(v, 0, 1);
  return v <= 0.5 ? v * v / 3.0 : (std::exp((v - hlg::kC) / hlg::kA) + hlg::kB) / 12.0;
}

double srgb_eotf(double v) {
  namespace s = k::srgb;
  v = clamp(v, 0, 1);
  return v <= s::kEncodedKnee ? v / s::kSlope : std::pow((v + s::kOffset) / (1.0 + s::kOffset), s::kGamma);
}

double srgb_inverse(double l) {
  namespace s = k::srgb;
  l = clamp(l, 0, 1);
  return l <= s::kLinearKnee ? l * s::kSlope : (1.0 + s::kOffset) * std::pow(l, 1.0 / s::kGamma) - s::kOffset;
}

struct Bt1886 {
  double a;
  double b;
};

Bt1886 bt1886() {
  const double g = k::bt1886::kGamma;
  const double wg = std::pow(double{kWhite}, 1.0 / g);
  const double bg = std::pow(double{kBlack}, 1.0 / g);
  return {.a = std::pow(wg - bg, g), .b = bg / (wg - bg)};
}

// La sortie en double d'un pixel ; les opérations par composante ignorent les deux autres.
std::array<double, 3> exact(TransferOp op, std::array<double, 3> c) {
  std::array<double, 3> out{};
  const Bt1886 d = bt1886();
  const double g = k::bt1886::kGamma;
  for (std::size_t i = 0; i < 3; ++i) {
    switch (op) {
      case TransferOp::PqEotf:
        out[i] = pq_eotf(c[i]);
        break;
      case TransferOp::PqInverseEotf:
        out[i] = pq_inverse(c[i]);
        break;
      case TransferOp::HlgOetf:
        out[i] = hlg_oetf(c[i]);
        break;
      case TransferOp::HlgInverseOetf:
        out[i] = hlg_inverse(c[i]);
        break;
      case TransferOp::SrgbEotf:
        out[i] = srgb_eotf(c[i]);
        break;
      case TransferOp::SrgbInverseEotf:
        out[i] = srgb_inverse(c[i]);
        break;
      case TransferOp::Bt1886Eotf:
        out[i] = d.a * std::pow(std::max(clamp(c[i], 0, 1) + d.b, 0.0), g);
        break;
      case TransferOp::Bt1886InverseEotf:
        out[i] = std::pow(clamp(c[i], kBlack, kWhite) / d.a, 1.0 / g) - d.b;
        break;
      default:
        break;
    }
  }
  if (op == TransferOp::HlgOotf) {
    const k::Coefficients kc = k::coefficients(k::Matrix::Bt2020Ncl);
    const double kr = kc.kr;
    const double kb = kc.kb;
    const double ys = kr * c[0] + (1.0 - kr - kb) * c[1] + kb * c[2];
    const double gamma = k::hlg::kGammaAt1000 + k::hlg::kGammaSlope * std::log10(kHlgPeak / 1000.0);
    const double gain = ys > 0.0 ? kHlgPeak * std::pow(ys, gamma - 1.0) : 0.0;
    for (std::size_t i = 0; i < 3; ++i) out[i] = gain * c[i];
  }
  if (op == TransferOp::Bt709ToBt2020 || op == TransferOp::Bt2020ToBt709) {
    const k::Mat3& m = op == TransferOp::Bt709ToBt2020 ? k::kBt709ToBt2020 : k::kBt2020ToBt709;
    for (int r = 0; r < 3; ++r) {
      out[static_cast<std::size_t>(r)] = m.at(r, 0) * c[0] + m.at(r, 1) * c[1] + m.at(r, 2) * c[2];
    }
  }
  return out;
}

bool propagates_nan(TransferOp op) {
  return op == TransferOp::HlgOotf || op == TransferOp::Bt709ToBt2020 || op == TransferOp::Bt2020ToBt709;
}

// La borne basse du domaine d'entrée de chaque courbe.
double lower_bound(TransferOp op) {
  return op == TransferOp::Bt1886InverseEotf ? kBlack : 0.0;
}

// La référence float pour un pixel, rangée comme le port.
std::array<float, 3> reference(TransferOp op, std::array<float, 3> c) {
  namespace r = k::ref;
  const r::HlgDisplay hlg = *r::hlg_display(kHlgPeak);
  const r::Bt1886Display d = *r::bt1886_display(kWhite, kBlack);
  const k::Rgb rgb{.r = c[0], .g = c[1], .b = c[2]};
  k::Rgb out{};
  switch (op) {
    case TransferOp::HlgOotf:
      out = r::hlg_ootf(rgb, hlg);
      break;
    case TransferOp::Bt709ToBt2020:
      out = r::convert_primaries(rgb, k::kBt709ToBt2020);
      break;
    case TransferOp::Bt2020ToBt709:
      out = r::convert_primaries(rgb, k::kBt2020ToBt709);
      break;
    default: {
      std::array<float, 3> each{};
      for (std::size_t i = 0; i < 3; ++i) {
        switch (op) {
          case TransferOp::PqEotf:
            each[i] = r::pq_eotf(c[i]);
            break;
          case TransferOp::PqInverseEotf:
            each[i] = r::pq_inverse_eotf(c[i]);
            break;
          case TransferOp::HlgOetf:
            each[i] = r::hlg_oetf(c[i]);
            break;
          case TransferOp::HlgInverseOetf:
            each[i] = r::hlg_inverse_oetf(c[i]);
            break;
          case TransferOp::SrgbEotf:
            each[i] = r::srgb_eotf(c[i]);
            break;
          case TransferOp::SrgbInverseEotf:
            each[i] = r::srgb_inverse_eotf(c[i]);
            break;
          case TransferOp::Bt1886Eotf:
            each[i] = r::bt1886_eotf(c[i], d);
            break;
          default:
            each[i] = r::bt1886_inverse_eotf(c[i], d);
            break;
        }
      }
      return each;
    }
  }
  return {out.r, out.g, out.b};
}

// Une grille du domaine, les coudes des courbes, et NaN (ramené à la borne basse).
k::RgbImage inputs(const Op& op, Precision p) {
  constexpr int kWidth = 97;
  k::RgbImage image = synthetic_rgb(kWidth, 5, static_cast<std::uint32_t>(op.op) + 1U, op.lo, op.hi);
  const std::array knees{0.0F,  1.0F, k::hlg::kKnee, 0.5F, k::srgb::kEncodedKnee, k::srgb::kLinearKnee,
                         op.lo, op.hi};
  for (std::size_t i = 0; i < knees.size(); ++i) {
    for (k::PlaneF& plane : image.planes) plane.at(static_cast<int>(i), 0) = knees[i];
  }
  for (k::PlaneF& plane : image.planes) plane.at(kWidth - 1, 4) = std::numeric_limits<float>::quiet_NaN();
  return as_stored(image, p);
}

k::RgbImage run_op(Rig& rig, const Op& op, const k::RgbImage& in, Precision p) {
  TensorBuffer src = upload_rgb(rig, in, p);
  TensorBuffer dst = tensor_target(rig, src.view.width, src.view.height, 3, p);
  const DisplayParams display{.hlg_peak = kHlgPeak, .bt1886_white = kWhite, .bt1886_black = kBlack};
  record_and_run(rig,
                 [&](Recorder& r) { return rig.kernels().transfer(r, op.op, src.view, dst.view, display); });
  return download_rgb(rig, dst);
}

std::array<float, 3> pixel(const k::RgbImage& image, int x, int y) {
  return {image.planes[0].at(x, y), image.planes[1].at(x, y), image.planes[2].at(x, y)};
}

}  // namespace

TEST_CASE("transfert : chaque opération en float contre la formule en double", "[vk_compute][transfer]") {
  Rig rig = make_rig();
  for (const Op& op : kOps) {
    INFO(op.name);
    const k::RgbImage in = inputs(op, Precision::F32);
    const k::RgbImage out = run_op(rig, op, in, Precision::F32);
    double worst = 0.0;
    for (int y = 0; y < in.height(); ++y) {
      for (int x = 0; x < in.width(); ++x) {
        const std::array<float, 3> c = pixel(in, x, y);
        std::array<double, 3> input{c[0], c[1], c[2]};
        if (std::isnan(c[0])) {
          // NaN : ramené à la borne basse par les courbes, propagé par les produits.
          if (propagates_nan(op.op)) {
            const std::array<float, 3> f = pixel(out, x, y);
            CHECK((std::isnan(f[0]) && std::isnan(f[1]) && std::isnan(f[2])));
            continue;
          }
          input.fill(lower_bound(op.op));
        }
        const std::array<double, 3> d = exact(op.op, input);
        const std::array<float, 3> f = pixel(out, x, y);
        for (std::size_t i = 0; i < 3; ++i) {
          const double bound =
              k::tolerance::kTransferRelative * std::fabs(d[i]) + k::tolerance::kTransferAbsolute * op.peak;
          const double error = std::fabs(f[i] - d[i]);
          worst = std::max(worst, error / bound);
          if (error > bound) FAIL_CHECK("entrée " << c[i] << " : " << f[i] << " contre " << d[i]);
        }
      }
    }
    INFO("pire écart : " << worst << " de la borne");
    CHECK(worst <= 1.0);
  }
}

TEST_CASE("transfert : variante float16 contre la référence rangée en float16", "[vk_compute][transfer]") {
  Rig rig = make_rig();
  require_f16(rig);
  for (const Op& op : kOps) {
    INFO(op.name);
    const k::RgbImage in = inputs(op, Precision::F16);
    const k::RgbImage out = run_op(rig, op, in, Precision::F16);
    for (int y = 0; y < in.height(); ++y) {
      for (int x = 0; x < in.width(); ++x) {
        const std::array<float, 3> want = reference(op.op, pixel(in, x, y));
        const std::array<float, 3> got = pixel(out, x, y);
        for (std::size_t i = 0; i < 3; ++i) {
          // Un pas de float16 de part et d'autre de la valeur, plus la borne des courbes.
          const double stored = k::ref::round_to_half(want[i]);
          const double bound = std::fabs(stored) / 1024.0 +
                               k::tolerance::kTransferRelative * std::fabs(stored) +
                               k::tolerance::kTransferAbsolute * op.peak;
          if (std::fabs(got[i] - stored) > bound) FAIL_CHECK(got[i] << " contre " << stored);
        }
      }
    }
  }
}

TEST_CASE("transfert : écrans, opérations et tenseurs hors contrat refusés", "[vk_compute][transfer]") {
  Rig rig = make_rig();
  KernelRunner& kr = rig.kernels();
  TensorBuffer a = tensor_target(rig, 8, 8, 3, Precision::F32);
  TensorBuffer b = tensor_target(rig, 8, 8, 3, Precision::F32);
  TensorBuffer c = tensor_target(rig, 8, 4, 3, Precision::F32);
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_refusal(kr.transfer(r, TransferOp::HlgOotf, a.view, b.view, {.hlg_peak = 300.0F}),
                  RefusalCode::FieldOutOfRange);
  require_refusal(kr.transfer(r, TransferOp::HlgOotf, a.view, b.view,
                              {.hlg_peak = std::numeric_limits<float>::quiet_NaN()}),
                  RefusalCode::FieldOutOfRange);
  require_refusal(
      kr.transfer(r, TransferOp::Bt1886Eotf, a.view, b.view, {.bt1886_white = 0.1F, .bt1886_black = 0.1F}),
      RefusalCode::FieldOutOfRange);
  require_refusal(kr.transfer(r, TransferOp::Bt1886InverseEotf, a.view, b.view, {.bt1886_black = -1.0F}),
                  RefusalCode::FieldOutOfRange);
  require_refusal(kr.transfer(r, static_cast<TransferOp>(42), a.view, b.view), RefusalCode::InputMismatch);
  require_refusal(kr.transfer(r, TransferOp::PqEotf, a.view, c.view), RefusalCode::InputMismatch);
  CHECK(require_value(r.finish())->dispatches() == 0);
  // Les paramètres d'écran de l'hôte sont ceux de la référence.
  const HlgDisplay hlg = require_value(hlg_display(kHlgPeak));
  CHECK(hlg.gamma == k::ref::hlg_display(kHlgPeak)->gamma);
  const Bt1886Display d = require_value(bt1886_display(kWhite, kBlack));
  CHECK(d.a == k::ref::bt1886_display(kWhite, kBlack)->a);
  CHECK(d.b == k::ref::bt1886_display(kWhite, kBlack)->b);
}
