#include "app/play/export_job.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <numeric>
#include <system_error>

#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

int even(double value) {
  return std::max(2, static_cast<int>(std::lround(value / 2.0)) * 2);
}

// « 48 », « 47.952 » : trois décimales au plus, sans zéros de fin.
std::string rate_text(media::Rational rate) {
  if (rate.num <= 0 || rate.den <= 0) return {};
  if (rate.num % rate.den == 0) return std::to_string(rate.num / rate.den);
  std::string text = std::format("{:.3f}", static_cast<double>(rate.num) / static_cast<double>(rate.den));
  while (text.back() == '0') text.pop_back();
  if (text.back() == '.') text.pop_back();
  return text;
}

}  // namespace

PlaySettings export_chain(const PlaySettings& s) {
  PlaySettings out = s;
  out.true_hdr = false;
  out.hdr_native = false;
  out.grade = false;
  out.smooth_motion = false;
  return out;
}

ExportSize chain_output(const PlaySettings& chain, int source_width, int source_height) {
  const WorkSize work = work_size(source_width, source_height, chain.work_height);
  const int factor = chain.upscale ? kSpanFactor : 1;
  return {.width = work.width * factor, .height = work.height * factor};
}

ExportSize export_size(ExportSize chain, ExportSize source, int height) {
  if (height == kExportSourceHeight) return {.width = even(source.width), .height = even(source.height)};
  if (height == kExportChainHeight || chain.height <= 0)
    return {.width = even(chain.width), .height = even(chain.height)};
  const double aspect = static_cast<double>(chain.width) / static_cast<double>(chain.height);
  return {.width = even(aspect * height), .height = even(height)};
}

media::Rational export_rate(media::Rational source, Factor factor) {
  if (source.num <= 0 || source.den <= 0) return {};
  media::Rational out = source;
  if (factor.num > factor.den && factor.den > 0) {
    out.num = source.num * factor.num;
    out.den = source.den * factor.den;
  }
  const int divisor = std::gcd(out.num, out.den);
  return {.num = out.num / divisor, .den = out.den / divisor};
}

std::filesystem::path default_export_path(const std::filesystem::path& source, ExportSize size,
                                          media::Rational rate) {
  const std::string fps = rate_text(rate);
  const std::string base = std::format("{} [ajp {}x{}{}]", prepare::utf8_path(source.stem()), size.width,
                                       size.height, fps.empty() ? std::string() : std::format(" {}fps", fps));
  const std::filesystem::path dir = source.parent_path();
  std::filesystem::path path = dir / prepare::path_from_utf8(base + ".mkv");
  std::error_code error;
  for (int n = 2; std::filesystem::exists(path, error) && n < 1000; ++n)
    path = dir / prepare::path_from_utf8(std::format("{} ({}).mkv", base, n));
  return path;
}

std::string export_label(exporter::Codec codec, exporter::Quality quality) {
  if (codec == exporter::Codec::Lossless) return std::string(exporter::label(codec));
  std::string level(exporter::label(quality));  // « Très haute » → « très haute »
  level[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(level[0])));
  return std::format("{} · qualité {}", exporter::label(codec), level);
}

}  // namespace ajp::app::play
