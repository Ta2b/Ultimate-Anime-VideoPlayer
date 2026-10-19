#include "export/hdr10.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <string_view>

namespace ajp::exporter {
namespace {

// Constantes de SMPTE ST 2084.
constexpr double kM1 = 2610.0 / 16384.0;
constexpr double kM2 = 2523.0 / 4096.0 * 128.0;
constexpr double kC1 = 3424.0 / 4096.0;
constexpr double kC2 = 2413.0 / 4096.0 * 32.0;
constexpr double kC3 = 2392.0 / 4096.0 * 32.0;
constexpr double kPqPeak = 10000.0;

constexpr std::size_t kCodes = std::size_t{std::numeric_limits<std::uint16_t>::max()} + 1;

std::expected<void, Refusal> check(bool ok, std::string_view field, double value) {
  if (ok) return {};
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                 std::format("export.mastering.{} : {} hors de sa plage", field, value)});
}

std::expected<void, Refusal> check_xy(const Chromaticity& c, std::string_view field) {
  const bool ok = c.x >= 0.0 && c.x <= 1.0 && c.y >= 0.0 && c.y <= 1.0;
  if (ok) return {};
  return std::unexpected(
      Refusal{RefusalCode::FieldOutOfRange,
              std::format("export.mastering.{} : ({}, {}) hors de [0, 1]", field, c.x, c.y)});
}

std::uint32_t round_nits(double nits) {
  return static_cast<std::uint32_t>(std::lround(nits));
}

}  // namespace

std::expected<void, Refusal> validate_mastering(const Hdr10Mastering& m) {
  for (const auto& [c, field] : {std::pair{m.red, "red"}, std::pair{m.green, "green"},
                                 std::pair{m.blue, "blue"}, std::pair{m.white, "white"}}) {
    if (auto ok = check_xy(c, field); !ok) return ok;
  }
  if (auto ok =
          check(m.max_nits >= kMinMasteringPeak && m.max_nits <= kMaxMasteringPeak, "max_nits", m.max_nits);
      !ok) {
    return ok;
  }
  return check(m.min_nits >= 0.0 && m.min_nits <= kMaxMasteringBlack && m.min_nits < m.max_nits, "min_nits",
               m.min_nits);
}

double pq_to_nits(double signal) {
  const double e = std::pow(std::clamp(signal, 0.0, 1.0), 1.0 / kM2);
  const double y = std::pow(std::max(e - kC1, 0.0) / (kC2 - kC3 * e), 1.0 / kM1);
  return kPqPeak * y;
}

LightMeter::LightMeter() : nits_(kCodes) {
  for (std::size_t code = 0; code < kCodes; ++code) {
    nits_[code] = pq_to_nits(static_cast<double>(code) / static_cast<double>(kCodes - 1));
  }
}

void LightMeter::add(const HostFrame& frame) {
  if (frame.width <= 0 || frame.height <= 0) return;
  const std::size_t row_values = frame.stride / sizeof(std::uint16_t);
  const auto width = static_cast<std::size_t>(frame.width);
  std::uint16_t frame_max = 0;
  double sum = 0.0;
  for (std::size_t y = 0; y < static_cast<std::size_t>(frame.height); ++y) {
    const auto row = frame.rgba.subspan(y * row_values, width * 4);
    double row_sum = 0.0;
    for (std::size_t x = 0; x < width; ++x) {
      const std::uint16_t brightest = std::max(std::max(row[x * 4], row[x * 4 + 1]), row[x * 4 + 2]);
      frame_max = std::max(frame_max, brightest);
      row_sum += nits_[brightest];
    }
    sum += row_sum;
  }
  const double pixels = static_cast<double>(width) * static_cast<double>(frame.height);
  add(FrameLight{.max_nits = nits_[frame_max], .average_nits = sum / pixels});
}

void LightMeter::add(const FrameLight& light) {
  max_nits_ = std::max(max_nits_, light.max_nits);
  max_average_nits_ = std::max(max_average_nits_, light.average_nits);
}

LightLevel LightMeter::level() const {
  return LightLevel{.max_cll = round_nits(max_nits_), .max_fall = round_nits(max_average_nits_)};
}

}  // namespace ajp::exporter
