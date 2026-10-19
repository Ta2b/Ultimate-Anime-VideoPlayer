#include "app/play/build_progress.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>

namespace ajp::app::play {
namespace {

constexpr double kPixelExponent = 0.8;
constexpr std::int64_t kReferencePixels = std::int64_t{1280} * 720;
constexpr std::size_t kMaxLogs = 128;               // journaux lus au plus
constexpr std::uintmax_t kMaxLogBytes = 4U << 20U;  // un journal plus gros est ignoré
constexpr std::string_view kBuiltIn = "Engine built in ";

// La valeur d'une option « --nom=valeur » de la ligne de commande, jusqu'à l'option suivante.
std::optional<std::string_view> option(std::string_view line, std::string_view name) {
  const std::size_t at = line.find(name);
  if (at == std::string_view::npos) return std::nullopt;
  std::string_view value = line.substr(at + name.size());
  const std::size_t next = value.find(" --");
  if (next != std::string_view::npos) value = value.substr(0, next);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\r')) value.remove_suffix(1);
  return value;
}

// « input:1x3x720x1280 » : hauteur × largeur, les deux dernières dimensions.
std::optional<std::int64_t> shape_pixels(std::string_view shape) {
  const std::size_t colon = shape.rfind(':');
  if (colon != std::string_view::npos) shape = shape.substr(colon + 1);
  const std::size_t comma = shape.find(',');  // plusieurs entrées : la première
  if (comma != std::string_view::npos) shape = shape.substr(0, comma);
  const std::size_t last = shape.rfind('x');
  if (last == std::string_view::npos || last == 0) return std::nullopt;
  const std::size_t before = shape.rfind('x', last - 1);
  const std::string_view hs = shape.substr(before == std::string_view::npos ? 0 : before + 1,
                                           last - (before == std::string_view::npos ? 0 : before + 1));
  const std::string_view ws = shape.substr(last + 1);
  std::int64_t h = 0;
  std::int64_t w = 0;
  if (std::from_chars(hs.data(), hs.data() + hs.size(), h).ec != std::errc{} ||
      std::from_chars(ws.data(), ws.data() + ws.size(), w).ec != std::errc{} || h <= 0 || w <= 0)
    return std::nullopt;
  return h * w;
}

// Plus de pixels allongent la construction ; moins ne la raccourcissent guère (le temps fixe du
// choix des tactiques domine : SPAN 64 s en 540p contre 60 s en 720p, RIFE 34 s en 960x576 contre
// 23 s en 1280x768 sur la 4070 Ti).
double scaled(double seconds, std::int64_t from_pixels, std::int64_t to_pixels) {
  if (from_pixels <= 0 || to_pixels <= from_pixels) return seconds;
  return seconds *
         std::pow(static_cast<double>(to_pixels) / static_cast<double>(from_pixels), kPixelExponent);
}

}  // namespace

BuildProgress::BuildProgress(double expected_seconds, std::int64_t started_us)
    : expected_s_(expected_seconds > 1.0 ? expected_seconds : 1.0), started_us_(started_us) {}

void BuildProgress::marker(float fraction) {
  // Les repères du milieu (plan mémoire, sérialisation) reviennent à chaque sous-moteur Myelin :
  // seuls ceux du début et de la fin sont sûrs.
  if (fraction > 0.10F && fraction < 0.95F) return;
  marker_ = std::max(marker_, std::min(fraction, 0.99F));
}

double BuildProgress::elapsed_seconds(std::int64_t now_us) const {
  return static_cast<double>(std::max<std::int64_t>(now_us - started_us_, 0)) * 1e-6;
}

float BuildProgress::fraction(std::int64_t now_us) const {
  if (finished_) return 1.0F;
  const double p = elapsed_seconds(now_us) / expected_s_;
  // Linéaire jusqu'à 90 % de la durée attendue, puis de plus en plus lent, sans atteindre 99 %.
  const double shaped = p < 0.9 ? p : 0.9 + 0.09 * (1.0 - std::exp(-(p - 0.9) * 4.0));
  return std::max(marker_, static_cast<float>(std::min(shaped, 0.99)));
}

std::optional<PastBuild> read_build_log(std::string_view text) {
  const std::string_view first = text.substr(0, text.find('\n'));
  const auto onnx = option(first, "--onnx=");
  auto shape = option(first, "--optShapes=");
  if (!shape) shape = option(first, "--shapes=");
  const std::size_t built = text.rfind(kBuiltIn);
  if (!onnx || !shape || built == std::string_view::npos) return std::nullopt;
  const auto pixels = shape_pixels(*shape);
  std::string_view number = text.substr(built + kBuiltIn.size());
  number = number.substr(0, number.find(' '));
  double seconds = 0.0;
  if (!pixels || std::from_chars(number.data(), number.data() + number.size(), seconds).ec != std::errc{} ||
      !(seconds > 0.0))
    return std::nullopt;
  return PastBuild{.onnx = std::filesystem::path(std::string(*onnx)), .pixels = *pixels, .seconds = seconds};
}

double expected_build_seconds(const std::filesystem::path& cache_dir, const std::filesystem::path& onnx,
                              std::int64_t pixels) {
  std::optional<PastBuild> nearest;
  double nearest_distance = 0.0;
  std::error_code error;
  std::size_t read = 0;
  for (const auto& entry : std::filesystem::directory_iterator(cache_dir, error)) {
    if (read >= kMaxLogs) break;
    const std::filesystem::path& path = entry.path();
    if (!path.string().ends_with(".build.log") || !entry.is_regular_file(error)) continue;
    if (entry.file_size(error) > kMaxLogBytes || error) continue;
    ++read;
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    const auto past = read_build_log(text.str());
    if (!past || (past->onnx != onnx && past->onnx.filename() != onnx.filename())) continue;
    const double distance = std::abs(
        std::log(static_cast<double>(std::max<std::int64_t>(pixels, 1)) / static_cast<double>(past->pixels)));
    if (!nearest || distance < nearest_distance) {
      nearest = past;
      nearest_distance = distance;
    }
  }
  if (!nearest) return scaled(kDefaultBuildSeconds, kReferencePixels, pixels);
  return scaled(nearest->seconds, nearest->pixels, pixels);
}

}  // namespace ajp::app::play
