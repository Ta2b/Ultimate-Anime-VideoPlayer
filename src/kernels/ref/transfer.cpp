#include "kernels/ref/transfer.h"

#include <cmath>
#include <format>

namespace ajp::kernels::ref {
namespace {

// Borne une entrée à son domaine ; NaN donne la borne basse, comme fmaxf.
float bounded(float v, float lo, float hi) {
  if (!(v > lo)) return lo;
  return v < hi ? v : hi;
}

float clamp01(float v) {
  return bounded(v, 0.0F, 1.0F);
}

}  // namespace

float pq_eotf(float signal) {
  const float p = std::pow(clamp01(signal), 1.0F / pq::kM2);
  const float y = std::pow(std::fmax(p - pq::kC1, 0.0F) / (pq::kC2 - pq::kC3 * p), 1.0F / pq::kM1);
  return y * pq::kPeakNits;
}

float pq_inverse_eotf(float nits) {
  const float ym = std::pow(bounded(nits, 0.0F, pq::kPeakNits) / pq::kPeakNits, pq::kM1);
  return std::pow((pq::kC1 + pq::kC2 * ym) / (1.0F + pq::kC3 * ym), pq::kM2);
}

float hlg_oetf(float scene) {
  const float e = clamp01(scene);
  if (e <= hlg::kKnee) return std::sqrt(3.0F * e);
  return hlg::kA * std::log(12.0F * e - hlg::kB) + hlg::kC;
}

float hlg_inverse_oetf(float signal) {
  const float v = clamp01(signal);
  if (v <= 0.5F) return v * v / 3.0F;
  return (std::exp((v - hlg::kC) / hlg::kA) + hlg::kB) / 12.0F;
}

Result<HlgDisplay> hlg_display(float peak_nits) {
  if (!(peak_nits >= hlg::kMinPeak && peak_nits <= hlg::kMaxPeak)) {
    return fail(ErrorCode::ParameterOutOfRange,
                std::format("pic HLG {} cd/m² hors de [{}, {}]", peak_nits, hlg::kMinPeak, hlg::kMaxPeak));
  }
  return HlgDisplay{.peak = peak_nits,
                    .gamma = hlg::kGammaAt1000 + hlg::kGammaSlope * std::log10(peak_nits / 1000.0F)};
}

Rgb hlg_ootf(const Rgb& scene, const HlgDisplay& display) {
  const Coefficients k = coefficients(Matrix::Bt2020Ncl);
  const float ys = k.kr * scene.r + (1.0F - k.kr - k.kb) * scene.g + k.kb * scene.b;
  // Une scène sans lumière (ou hors gamut, Ys ≤ 0) s'affiche noire.
  const float gain = ys > 0.0F ? display.peak * std::pow(ys, display.gamma - 1.0F) : 0.0F;
  return {.r = gain * scene.r, .g = gain * scene.g, .b = gain * scene.b};
}

float srgb_eotf(float signal) {
  const float v = clamp01(signal);
  if (v <= srgb::kEncodedKnee) return v / srgb::kSlope;
  return std::pow((v + srgb::kOffset) / (1.0F + srgb::kOffset), srgb::kGamma);
}

float srgb_inverse_eotf(float light) {
  const float l = clamp01(light);
  if (l <= srgb::kLinearKnee) return l * srgb::kSlope;
  return (1.0F + srgb::kOffset) * std::pow(l, 1.0F / srgb::kGamma) - srgb::kOffset;
}

Result<Bt1886Display> bt1886_display(float white, float black) {
  if (!(black >= 0.0F && white > black && std::isfinite(white))) {
    return fail(ErrorCode::ParameterOutOfRange,
                std::format("BT.1886 : blanc {} et noir {} invalides", white, black));
  }
  const float wg = std::pow(white, 1.0F / bt1886::kGamma);
  const float bg = std::pow(black, 1.0F / bt1886::kGamma);
  return Bt1886Display{
      .white = white, .black = black, .a = std::pow(wg - bg, bt1886::kGamma), .b = bg / (wg - bg)};
}

float bt1886_eotf(float signal, const Bt1886Display& display) {
  return display.a * std::pow(std::fmax(clamp01(signal) + display.b, 0.0F), bt1886::kGamma);
}

float bt1886_inverse_eotf(float light, const Bt1886Display& display) {
  const float l = bounded(light, display.black, display.white);
  return std::pow(l / display.a, 1.0F / bt1886::kGamma) - display.b;
}

Rgb convert_primaries(const Rgb& linear, const Mat3& matrix) {
  const auto m = [&matrix](int row, int column) { return static_cast<float>(matrix.at(row, column)); };
  return {.r = m(0, 0) * linear.r + m(0, 1) * linear.g + m(0, 2) * linear.b,
          .g = m(1, 0) * linear.r + m(1, 1) * linear.g + m(1, 2) * linear.b,
          .b = m(2, 0) * linear.r + m(2, 1) * linear.g + m(2, 2) * linear.b};
}

}  // namespace ajp::kernels::ref
