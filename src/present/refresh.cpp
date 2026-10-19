#include "present/refresh.h"

#include <cmath>
#include <format>

#include "core/cadence.h"
#include "core/debug_flags.h"

namespace ajp::present {

namespace {

double relative_gap(double measured, double nominal) {
  return std::abs(measured - nominal) / nominal;
}

}  // namespace

std::expected<core::Ratio, Refusal> nominal_refresh(double measured_hz) {
  // Écrit en positif pour qu'un NaN soit refusé lui aussi.
  if (!(measured_hz >= core::limits::kRefreshHz.lo && measured_hz <= core::limits::kRefreshHz.hi)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("rafraîchissement {} Hz hors de [1, 1000]", measured_hz)});
  }
  const auto integer = std::llround(measured_hz);
  const auto ntsc = std::llround(measured_hz * 1.001);
  const double ntsc_hz = static_cast<double>(ntsc) * 1000.0 / 1001.0;
  const double integer_gap = relative_gap(measured_hz, static_cast<double>(integer));
  const double ntsc_gap = relative_gap(measured_hz, ntsc_hz);
  if (integer_gap <= ntsc_gap && integer_gap <= refresh::kNominalTolerance) {
    return core::Ratio{integer, 1};
  }
  if (ntsc_gap <= refresh::kNominalTolerance) return core::reduced({ntsc * 1000, 1001});
  const auto millihertz = std::llround(measured_hz * static_cast<double>(refresh::kMeasuredDen));
  return core::reduced({millihertz, refresh::kMeasuredDen});
}

std::expected<core::Ratio, Refusal> nominal_refresh(core::Ratio measured) {
  if (measured.num <= 0 || measured.den <= 0) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("rafraîchissement {}/{} Hz non positif", measured.num, measured.den)});
  }
  return nominal_refresh(measured.value());
}

}  // namespace ajp::present
