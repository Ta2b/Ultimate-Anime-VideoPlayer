#include "app/play/grade_taste.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include "app/play/i18n.h"
#include "core/charconv.h"
#include "render/adaptive_rules.h"
#include "render/curve.h"

namespace ajp::app::play {
namespace {

namespace rules = render::adaptive_rules;

constexpr double kApprovalWeight = 0.25;
constexpr double kHalfLifeS = 90.0 * 86'400.0;
// Les seuils sombre / claire de la courbe d'avant par défaut (GradeCurve::lo, hi), en part du blanc.
constexpr double kDarkLo = 0.13;
constexpr double kBrightHi = 0.54;
constexpr double kGlowGap = 0.4;  // un éclat : p99,5 au moins à 0,4 au-dessus de la médiane

constexpr std::array<std::string_view, kSceneClasses> kNames{"sombre", "eclat", "claire", "ciel"};

double age_weight(std::int64_t unix_s, std::int64_t now_unix) {
  const double age = std::max<double>(0.0, static_cast<double>(now_unix - unix_s));
  return std::exp2(-age / kHalfLifeS);
}

bool number(std::string_view text, double& out) {
  const auto [end, error] = core::from_chars(text.data(), text.data() + text.size(), out);
  return error == std::errc{} && end == text.data() + text.size();
}

bool number(std::string_view text, std::int64_t& out) {
  const auto [end, error] = core::from_chars(text.data(), text.data() + text.size(), out);
  return error == std::errc{} && end == text.data() + text.size();
}

char kind_letter(LearnSample::Kind kind) {
  switch (kind) {
    case LearnSample::Kind::Calibration:
      return 'f';
    case LearnSample::Kind::Approval:
      return 'k';
    case LearnSample::Kind::Correction:
      break;
  }
  return 'c';
}

}  // namespace

std::string_view name(SceneClass scene) {
  return kNames.at(static_cast<std::size_t>(scene));
}

GradeTaste clamped(GradeTaste taste) {
  for (double& e : taste.e) e = std::clamp(e, -render::kCurveMaxE, render::kCurveMaxE);
  taste.toe = std::clamp(taste.toe, -render::kCurveMaxToe, render::kCurveMaxToe);
  taste.vibrance = std::clamp(taste.vibrance, 1.0, render::kCurveMaxVibrance);
  return taste;
}

GradeTaste migrate(const GradeCurve& curve) {
  const GradeCurve defaults;
  const auto e_of = [](double gray, double neutral) {
    return gray > 0.0 && neutral > 0.0 ? std::log2(gray / neutral) / 2.0 : 0.0;
  };
  GradeTaste taste;
  taste.e.at(static_cast<std::size_t>(SceneClass::Dark)) = e_of(curve.gray, defaults.gray);
  taste.e.at(static_cast<std::size_t>(SceneClass::Glow)) = e_of(curve.gray_glow, defaults.gray_glow);
  taste.e.at(static_cast<std::size_t>(SceneClass::Bright)) = e_of(curve.gray_bright, defaults.gray_bright);
  taste.e.at(static_cast<std::size_t>(SceneClass::Sky)) = e_of(curve.gray_bright, defaults.gray_bright);
  taste.toe = (curve.shadows - defaults.shadows) / 2.0;
  taste.vibrance = 1.0 + (curve.colors - defaults.colors) * 0.75;
  return clamped(taste);
}

SceneClass class_of(const SceneMix& mix) {
  if (mix.tv >= 0.5) return SceneClass::Sky;
  const double dark = (1.0 - mix.t) * (1.0 - mix.glow);
  const double glow = (1.0 - mix.t) * mix.glow;
  if (mix.t >= dark && mix.t >= glow) return SceneClass::Bright;
  return glow > dark ? SceneClass::Glow : SceneClass::Dark;
}

SceneClass class_of(const media::ShotStats& stats, double series_mean) {
  if (stats.blue >= rules::kBlueHi || (stats.blue >= rules::kBlueLo && stats.chroma >= rules::kChromaHi))
    return SceneClass::Sky;
  const double x = stats.mean / series_ratio(series_mean > 0.0 ? series_mean : -1.0);
  const double t = std::clamp((x - kDarkLo) / (kBrightHi - kDarkLo), 0.0, 1.0);
  if (t >= 0.5) return SceneClass::Bright;
  if (stats.p50 < rules::kGlowSceneHi && stats.bright >= rules::kGlowShareLo &&
      stats.p995 >= stats.p50 + kGlowGap)
    return SceneClass::Glow;
  return SceneClass::Dark;
}

std::expected<TasteReport, std::string> learn_taste(std::span<const TasteSample> samples,
                                                    const GradeTaste& now, std::int64_t now_unix) {
  TasteReport report;
  std::array<double, kSceneClasses> e_sum{};
  double toe_sum = 0.0;
  double vibrance_sum = 0.0;
  double weight = 0.0;
  for (const TasteSample& s : samples) {
    const bool approval = s.kind == LearnSample::Kind::Approval;
    ++(approval ? report.approvals : report.corrections);
    const double w = (approval ? kApprovalWeight : 1.0) * age_weight(s.unix_s, now_unix);
    const auto c = static_cast<std::size_t>(s.scene);
    report.support.at(c) += w;
    e_sum.at(c) += w * s.taste.e_of(s.scene);
    toe_sum += w * s.taste.toe;
    vibrance_sum += w * s.taste.vibrance;
    weight += w;
  }
  if (report.corrections < 2)
    return std::unexpected(std::string(
        i18n::tr("At least two corrections are needed: set a few scenes with \"This scene\" first.")));
  for (std::size_t c = 0; c < kSceneClasses; ++c)
    report.taste.e.at(c) = (kTastePrior * now.e.at(c) + e_sum.at(c)) / (kTastePrior + report.support.at(c));
  report.taste.toe = (kTastePrior * now.toe + toe_sum) / (kTastePrior + weight);
  report.taste.vibrance = (kTastePrior * now.vibrance + vibrance_sum) / (kTastePrior + weight);
  report.taste = clamped(report.taste);
  return report;
}

std::string_view taste_header() {
  return "v2\tunix_s\tgenre\tclasse\te\tpied\tvibrance\n";
}

std::string taste_line(const TasteSample& s) {
  return std::format("v2\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n", s.unix_s, kind_letter(s.kind), name(s.scene),
                     s.taste.e_of(s.scene), s.taste.toe, s.taste.vibrance);
}

TasteRead parse_taste_samples(std::string_view text) {
  TasteRead out;
  bool first = true;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) continue;
    if (std::exchange(first, false) && line == taste_header().substr(0, taste_header().size() - 1)) continue;
    std::array<std::string_view, 7> fields{};
    std::size_t count = 0;
    while (count < fields.size()) {
      const std::size_t tab = line.find('\t');
      fields.at(count++) = line.substr(0, tab);
      if (tab == std::string_view::npos) break;
      line = line.substr(tab + 1);
    }
    TasteSample s;
    double e = 0.0;
    const auto scene = std::ranges::find(kNames, count == fields.size() ? fields[3] : std::string_view{});
    const bool read = count == fields.size() && fields[0] == "v2" && fields[2].size() == 1 &&
                      scene != kNames.end() && number(fields[1], s.unix_s) && number(fields[4], e) &&
                      number(fields[5], s.taste.toe) && number(fields[6], s.taste.vibrance);
    const char kind = read ? fields[2].front() : '?';
    if (kind != 'c' && kind != 'f' && kind != 'k') {
      ++out.rejected;
      continue;
    }
    s.kind = kind == 'f'   ? LearnSample::Kind::Calibration
             : kind == 'k' ? LearnSample::Kind::Approval
                           : LearnSample::Kind::Correction;
    s.scene = static_cast<SceneClass>(scene - kNames.begin());
    s.taste.e.at(static_cast<std::size_t>(s.scene)) = e;
    s.taste = clamped(s.taste);
    out.samples.push_back(s);
  }
  return out;
}

TasteRead read_taste_samples(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  std::ostringstream text;
  text << in.rdbuf();
  return parse_taste_samples(text.str());
}

void append_taste_sample(const std::filesystem::path& file, const TasteSample& sample) {
  std::error_code error;
  const bool fresh = !std::filesystem::exists(file, error);
  std::ofstream out(file, std::ios::app | std::ios::binary);
  if (!out) return;
  if (fresh) out << taste_header();
  out << taste_line(sample);
}

std::filesystem::path taste_log_of(const std::filesystem::path& learning_log) {
  if (learning_log.empty()) return {};
  return learning_log.parent_path() / "adaptive-learning-v2.tsv";
}

}  // namespace ajp::app::play
