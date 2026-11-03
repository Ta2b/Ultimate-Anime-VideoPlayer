#include "app/play/grade_learning.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <iterator>
#include <sstream>
#include <system_error>

#include "app/play/i18n.h"
#include "core/charconv.h"
#include "render/grade_params.h"

namespace ajp::app::play {
namespace {

using render::grade_limits::kMiddleGray;
using render::grade_limits::kShadows;
using render::grade_limits::kVibrance;

double unit(double v) {
  return std::clamp(v, 0.0, 1.0);
}

double smooth(double x) {
  x = unit(x);
  return x * x * (3.0 - 2.0 * x);
}

// Les plages des curseurs : l'écart d'une famille se compte en part de la sienne.
constexpr std::array<double, 3> kScale{kShadows.hi - kShadows.lo, kVibrance.hi - kVibrance.lo,
                                       kMiddleGray.hi - kMiddleGray.lo};

// Les poids des bouts de courbe d'une famille dans une scène (sombre, claire, éclat ; et ciel pour les
// couleurs). Couleurs : la vibrance de la scène claire est reparamétrée en bout à part, v × (1 − coupe), pour
// que la courbe reste linéaire en ses bouts.
std::array<double, 4> features(std::size_t family, const SceneMix& m) {
  const double dark = (1.0 - m.glow) * (1.0 - m.t);
  const double bright = (1.0 - m.glow) * m.t;
  if (family == 1) return {(1.0 - m.tv) * dark, (1.0 - m.tv) * bright, (1.0 - m.tv) * m.glow, m.tv};
  return {dark, bright, m.glow, 0.0};
}

std::size_t anchor_count(std::size_t family) {
  return family == 1 ? 4 : 3;
}

double value_of(const SceneLook& look, std::size_t family) {
  return family == 0 ? look.shadows : family == 1 ? look.colors : look.gray;
}

std::array<double, 4> anchors_of(const GradeCurve& c, std::size_t family) {
  if (family == 0) return {c.shadows, c.shadows_bright, c.shadows_glow, 0.0};
  if (family == 1) return {c.colors, c.colors * (1.0 - unit(c.colors_cut)), c.colors_glow, c.colors_sky};
  return {c.gray, c.gray_bright, c.gray_glow, 0.0};
}

// Les bouts d'une famille ramenés dans les bornes de leurs curseurs (couleurs : la scène claire entre 0 et la
// scène sombre, une coupe de 0 à 1).
std::array<double, 4> bounded(std::array<double, 4> a, std::size_t family) {
  const core::Range<double> range = family == 0 ? kShadows : family == 1 ? kVibrance : kMiddleGray;
  for (double& v : a) v = std::clamp(v, range.lo, range.hi);
  if (family == 1) a[1] = std::clamp(a[1], 0.0, a[0]);
  return a;
}

void put_anchors(GradeCurve& c, std::size_t family, const std::array<double, 4>& a) {
  if (family == 0) {
    c.shadows = a[0];
    c.shadows_bright = a[1];
    c.shadows_glow = a[2];
  } else if (family == 1) {
    if (a[0] > 1e-6) c.colors_cut = unit(1.0 - a[1] / a[0]);
    c.colors = a[0];
    c.colors_glow = a[2];
    c.colors_sky = a[3];
  } else {
    c.gray = a[0];
    c.gray_bright = a[1];
    c.gray_glow = a[2];
  }
}

// Système linéaire symétrique défini positif de taille n (au plus 4), par élimination de Gauss.
std::array<double, 4> solve(std::array<std::array<double, 4>, 4> m, std::array<double, 4> b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t pivot = i;
    for (std::size_t r = i + 1; r < n; ++r)
      if (std::abs(m[r][i]) > std::abs(m[pivot][i])) pivot = r;
    std::swap(m[i], m[pivot]);
    std::swap(b[i], b[pivot]);
    if (std::abs(m[i][i]) < 1e-12) continue;
    for (std::size_t r = i + 1; r < n; ++r) {
      const double f = m[r][i] / m[i][i];
      for (std::size_t c = i; c < n; ++c) m[r][c] -= f * m[i][c];
      b[r] -= f * b[i];
    }
  }
  std::array<double, 4> x{};
  for (std::size_t i = n; i-- > 0;) {
    double sum = b[i];
    for (std::size_t c = i + 1; c < n; ++c) sum -= m[i][c] * x[c];
    x[i] = std::abs(m[i][i]) < 1e-12 ? 0.0 : sum / m[i][i];
  }
  return x;
}

// Un échantillon pondéré : son poids dans chaque famille (0 : n'en dit rien).
struct Weighted {
  const LearnSample* sample = nullptr;
  std::array<double, 3> w{};
};

// La régularisation vers la courbe en cours, en poids de correction : un bout que rien ne contredit ne bouge
// pas, un bout que contredit une seule correction en prend les deux tiers.
constexpr double kPrior = 0.25;
// Écart (part de la plage) au-delà duquel une correction compte moins (Huber) : une main qui a glissé ne
// tord pas toute la courbe.
constexpr double kHuber = 0.12;

struct Fit {
  GradeCurve curve;
  double loss = 0.0;
};

// Les meilleurs bouts sous les seuils lo et hi, et l'écart qu'il reste (régularisation comprise).
Fit fit_at(std::span<const Weighted> data, std::span<const double> huber, const GradeCurve& now, double lo,
           double hi) {
  Fit fit{.curve = now};
  fit.curve.lo = lo;
  fit.curve.hi = hi;
  for (std::size_t family = 0; family < 3; ++family) {
    const std::size_t n = anchor_count(family);
    std::array<std::array<double, 4>, 4> m{};
    std::array<double, 4> b{};
    double yy = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
      const double w = data[i].w[family] * huber[i];
      if (w <= 0.0) continue;
      const std::array<double, 4> f = features(family, scene_mix(data[i].sample->scene, lo, hi));
      const double y = value_of(data[i].sample->look, family);
      for (std::size_t r = 0; r < n; ++r) {
        b[r] += w * f[r] * y;
        for (std::size_t c = 0; c < n; ++c) m[r][c] += w * f[r] * f[c];
      }
      yy += w * y * y;
    }
    const std::array<double, 4> a0 = anchors_of(now, family);
    std::array<std::array<double, 4>, 4> mp = m;
    std::array<double, 4> bp = b;
    for (std::size_t r = 0; r < n; ++r) {
      mp[r][r] += kPrior;
      bp[r] += kPrior * a0[r];
    }
    const std::array<double, 4> a = bounded(solve(mp, bp, n), family);
    double loss = yy;
    for (std::size_t r = 0; r < n; ++r) {
      loss += kPrior * (a[r] - a0[r]) * (a[r] - a0[r]) - 2.0 * a[r] * b[r];
      for (std::size_t c = 0; c < n; ++c) loss += a[r] * m[r][c] * a[c];
    }
    const double scale = kScale[family];
    fit.loss += loss / (scale * scale);
    put_anchors(fit.curve, family, a);
  }
  // Les seuils ne bougent que si les corrections le paient : un déplacement de 5 points du seuil sombre (10
  // du clair) coûte autant qu'une correction d'un dixième de la plage.
  const double dlo = (lo - now.lo) / 5.0;
  const double dhi = (hi - now.hi) / 10.0;
  fit.loss += 0.002 * (dlo * dlo + dhi * dhi);
  return fit;
}

Fit best_fit(std::span<const Weighted> data, std::span<const double> huber, const GradeCurve& now,
             bool search) {
  Fit best = fit_at(data, huber, now, now.lo, now.hi);
  if (!search) return best;
  const double lo_from = std::max(1.0, std::round(now.lo) - 10.0);
  const double lo_to = std::min(60.0, std::round(now.lo) + 10.0);
  for (double lo = lo_from; lo <= lo_to; lo += 1.0) {
    const double hi_from = std::max(lo + 8.0, std::round(now.hi) - 20.0);
    const double hi_to = std::min(95.0, std::round(now.hi) + 20.0);
    for (double hi = hi_from; hi <= hi_to; hi += 2.0) {
      Fit fit = fit_at(data, huber, now, lo, hi);
      if (fit.loss < best.loss) best = fit;
    }
  }
  return best;
}

// L'écart d'un échantillon à une courbe, en part de la plage, sur les familles qu'il corrige (toutes pour
// une approbation).
double residual(const LearnSample& s, const GradeCurve& c, std::uint8_t families) {
  const SceneMix m = scene_mix(s.scene, c.lo, c.hi);
  const SceneLook look = look_at(c, m.t, m.glow, m.tv);
  double sum = 0.0;
  int count = 0;
  for (std::size_t family = 0; family < 3; ++family) {
    if ((families & (1U << family)) == 0) continue;
    const double r = (value_of(s.look, family) - value_of(look, family)) / kScale[family];
    sum += r * r;
    ++count;
  }
  return count == 0 ? 0.0 : std::sqrt(sum / count);
}

bool corrects(const LearnSample& s) {
  return s.kind != LearnSample::Kind::Approval && s.touched != 0;
}

char kind_letter(LearnSample::Kind kind) {
  switch (kind) {
    case LearnSample::Kind::Correction:
      return 'c';
    case LearnSample::Kind::Calibration:
      return 'f';
    case LearnSample::Kind::Approval:
      return 'k';
  }
  return 'c';
}

template <typename T>
bool number(std::string_view text, T& out) {
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = core::from_chars(
      text.data(), end, out);  // NOLINT(bugprone-suspicious-stringview-data-usage) : bornée par end
  return ec == std::errc{} && ptr == end;
}

}  // namespace

double series_ratio(double series_luma) {
  if (series_luma < 0.0) return 1.0;
  return std::clamp(series_luma / kSeriesRef, render::adaptive_rules::kRatioMin,
                    render::adaptive_rules::kRatioMax);
}

GradeCurve curve_of(const PlaySettings& s) {
  const render::GradeSettings& g = s.grade_values;
  return GradeCurve{.lo = g.adaptive_lo,
                    .hi = g.adaptive_hi,
                    .shadows = g.shadows,
                    .shadows_bright = g.adaptive_shadows,
                    .shadows_glow = g.adaptive_glow_shadows,
                    .colors = g.vibrance_rtx,
                    .colors_cut = g.adaptive_vibrance_cut,
                    .colors_glow = g.adaptive_glow_vibrance,
                    .colors_sky = g.adaptive_vibrance,
                    .gray = static_cast<double>(s.true_hdr_gray),
                    .gray_bright = g.adaptive_gray,
                    .gray_glow = g.adaptive_glow_gray};
}

void set_curve(PlaySettings& s, const GradeCurve& c) {
  render::GradeSettings& g = s.grade_values;
  g.adaptive_lo = c.lo;
  g.adaptive_hi = c.hi;
  g.shadows = c.shadows;
  g.adaptive_shadows = c.shadows_bright;
  g.adaptive_glow_shadows = c.shadows_glow;
  g.vibrance_rtx = c.colors;
  g.adaptive_vibrance_cut = c.colors_cut;
  g.adaptive_glow_vibrance = c.colors_glow;
  g.adaptive_vibrance = c.colors_sky;
  s.true_hdr_gray = static_cast<int>(std::lround(c.gray));
  g.adaptive_gray = c.gray_bright;
  g.adaptive_glow_gray = c.gray_glow;
}

SceneLook look_at(const GradeCurve& c, double t, double glow, double tv) {
  t = unit(t);
  glow = unit(glow);
  tv = unit(tv);
  const double keep = 1.0 - unit(c.colors_cut) * t;
  return SceneLook{
      .shadows = std::lerp(std::lerp(c.shadows, c.shadows_bright, t), c.shadows_glow, glow),
      .colors = std::lerp(std::lerp(c.colors * keep, c.colors_glow, glow), c.colors_sky, tv),
      .gray = std::lerp(std::lerp(c.gray, c.gray_bright, t), c.gray_glow, glow),
  };
}

std::uint8_t touched_families(const GradeCurve& a, const GradeCurve& b) {
  std::uint8_t out = 0;
  if (a.shadows != b.shadows || a.shadows_bright != b.shadows_bright || a.shadows_glow != b.shadows_glow)
    out |= kShadowsFamily;
  if (a.colors != b.colors || a.colors_cut != b.colors_cut || a.colors_glow != b.colors_glow ||
      a.colors_sky != b.colors_sky)
    out |= kColorsFamily;
  if (a.gray != b.gray || a.gray_bright != b.gray_bright || a.gray_glow != b.gray_glow) out |= kGrayFamily;
  return out;
}

void correct_scene(GradeCurve& c, double t, double glow, double tv, Family family, double wanted) {
  t = unit(t);
  glow = unit(glow);
  tv = unit(tv);
  std::array<double*, 3> anchors{};
  std::array<double, 3> f{};
  core::Range<double> range = kShadows;
  if (family == kShadowsFamily) {
    anchors = {&c.shadows, &c.shadows_bright, &c.shadows_glow};
    f = {(1.0 - glow) * (1.0 - t), (1.0 - glow) * t, glow};
  } else if (family == kColorsFamily) {
    // La coupe reste : la scène sombre porte la scène claire avec elle, comme le fait le curseur.
    anchors = {&c.colors, &c.colors_glow, &c.colors_sky};
    f = {(1.0 - tv) * (1.0 - glow) * (1.0 - unit(c.colors_cut) * t), (1.0 - tv) * glow, tv};
    range = kVibrance;
  } else {
    anchors = {&c.gray, &c.gray_bright, &c.gray_glow};
    f = {(1.0 - glow) * (1.0 - t), (1.0 - glow) * t, glow};
    range = kMiddleGray;
  }
  wanted = std::clamp(wanted, range.lo, range.hi);
  const std::size_t index = family == kShadowsFamily ? 0 : family == kColorsFamily ? 1 : 2;
  // Le plus petit changement (au sens des moindres carrés) qui rend la valeur voulue : chaque bout bouge en
  // proportion de sa part ; un bout en butée laisse le reste aux autres.
  for (int round = 0; round < 4; ++round) {
    const double delta = wanted - value_of(look_at(c, t, glow, tv), index);
    if (std::abs(delta) < 1e-9) return;
    double norm = 0.0;
    std::array<bool, 3> free{};
    for (std::size_t k = 0; k < 3; ++k) {
      free[k] = f[k] > 1e-9 && (delta > 0.0 ? *anchors[k] < range.hi : *anchors[k] > range.lo);
      if (free[k]) norm += f[k] * f[k];
    }
    if (norm < 1e-12) return;
    for (std::size_t k = 0; k < 3; ++k)
      if (free[k]) *anchors[k] = std::clamp(*anchors[k] + f[k] * delta / norm, range.lo, range.hi);
  }
}

SceneMix scene_mix(const LearnScene& scene, double lo, double hi) {
  using render::grade_limits::forced_bright;
  using render::grade_limits::forced_glow;
  if (scene.force >= 0.0)
    return SceneMix{.t = forced_bright(scene.force) ? 1.0 : 0.0,
                    .glow = forced_glow(scene.force) ? 1.0 : 0.0};
  const double from = lo / 100.0;
  const double to = std::max(hi, lo + 1.0) / 100.0;
  const double t = smooth((scene.x - from) / (to - from));
  return SceneMix{.t = t, .glow = unit(scene.glow_seen) * (1.0 - t), .tv = unit(scene.tv)};
}

std::expected<LearnReport, std::string> learn_grade(std::span<const LearnSample> samples,
                                                    const GradeCurve& now, std::int64_t now_unix) {
  LearnReport report;
  for (const LearnSample& s : samples) {
    if (corrects(s)) ++report.corrections;
    if (s.kind == LearnSample::Kind::Approval) ++report.approvals;
  }
  if (report.corrections < 2)
    return std::unexpected(std::string(
        i18n::tr("At least two corrections are needed: set a few scenes with \"This scene\" first.")));
  // Le poids de chaque échantillon : une correction compte entier, un réglage de calibration une fois et
  // demie, une approbation (une scène regardée longtemps sans rien changer) un dixième ; une famille qu'une
  // correction ne touche pas vaut approbation. Le temps l'efface (demi-vie de 60 jours).
  std::vector<Weighted> data;
  data.reserve(samples.size());
  for (const LearnSample& s : samples) {
    const double days = std::max<double>(static_cast<double>(now_unix - s.unix_s), 0.0) / 86'400.0;
    const double recency = std::exp2(-days / 60.0);
    const double full = s.kind == LearnSample::Kind::Calibration  ? 1.5
                        : s.kind == LearnSample::Kind::Correction ? 1.0
                                                                  : 0.1;
    Weighted w{.sample = &s};
    for (std::size_t family = 0; family < 3; ++family) {
      const bool touched = corrects(s) && (s.touched & (1U << family)) != 0;
      w.w[family] = (touched ? full : 0.1) * recency;
    }
    data.push_back(w);
  }
  // Une correction plus récente remplace ce qui la précède : l'approbation silencieuse d'avant, sur le même
  // genre de scène, ne compte plus ; une correction d'avant sur presque la même scène compte moitié moins
  // (l'utilisateur affine).
  for (const LearnSample& later : samples) {
    if (!corrects(later)) continue;
    const SceneMix ml = scene_mix(later.scene, now.lo, now.hi);
    for (Weighted& w : data) {
      const LearnSample& s = *w.sample;
      if (s.unix_s >= later.unix_s) continue;
      const SceneMix ms = scene_mix(s.scene, now.lo, now.hi);
      const double gap = std::abs(ms.t - ml.t) + std::abs(ms.glow - ml.glow) + std::abs(ms.tv - ml.tv);
      const bool touched_before = corrects(s);
      if (gap > (touched_before ? 0.08 : 0.25)) continue;
      if (touched_before && std::abs(s.scene.x - later.scene.x) > 0.03) continue;
      for (std::size_t family = 0; family < 3; ++family) {
        if ((later.touched & (1U << family)) == 0) continue;
        const bool touched = touched_before && (s.touched & (1U << family)) != 0;
        w.w[family] *= touched ? 0.5 : 0.0;
      }
    }
  }
  // Les seuils ne se cherchent qu'avec assez de corrections pour les situer ; deux passes, la seconde
  // modère les corrections que la première explique mal (Huber).
  const bool search = report.corrections >= 4;
  std::vector<double> huber(data.size(), 1.0);
  Fit fit = best_fit(data, huber, now, search);
  for (std::size_t i = 0; i < data.size(); ++i) {
    const double r = residual(*data[i].sample, fit.curve,
                              corrects(*data[i].sample) ? data[i].sample->touched : kAllFamilies);
    huber[i] = r <= kHuber ? 1.0 : kHuber / r;
  }
  fit = best_fit(data, huber, now, search);
  report.curve = fit.curve;
  // Le bilan : l'écart moyen des corrections avant et après, et leur poids dans chaque genre de scène.
  int counted = 0;
  for (const LearnSample& s : samples) {
    if (!corrects(s)) continue;
    report.error_before += residual(s, now, s.touched);
    report.error_after += residual(s, fit.curve, s.touched);
    ++counted;
    const SceneMix m = scene_mix(s.scene, fit.curve.lo, fit.curve.hi);
    report.support_dark += (1.0 - m.glow) * (1.0 - m.t);
    report.support_bright += (1.0 - m.glow) * m.t;
    report.support_glow += m.glow;
  }
  report.error_before /= counted;
  report.error_after /= counted;
  return report;
}

std::string_view learn_header() {
  return "unix_s\tgenre\ttouche\tx\teclat\tciel\tforce\tnoirs\tcouleurs\tgris\n";
}

std::string learn_line(const LearnSample& s) {
  return std::format("{}\t{}\t{}\t{:.5f}\t{:.4f}\t{:.4f}\t{:.1f}\t{:.4f}\t{:.4f}\t{:.2f}\n", s.unix_s,
                     kind_letter(s.kind), static_cast<int>(s.touched), s.scene.x, s.scene.glow_seen,
                     s.scene.tv, s.scene.force, s.look.shadows, s.look.colors, s.look.gray);
}

std::vector<LearnSample> parse_learn_samples(std::string_view text) {
  std::vector<LearnSample> out;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    std::array<std::string_view, 10> fields{};
    std::size_t count = 0;
    while (count < fields.size()) {
      const std::size_t tab = line.find('\t');
      fields[count++] = line.substr(0, tab);
      if (tab == std::string_view::npos) break;
      line = line.substr(tab + 1);
    }
    if (count != fields.size() || fields[1].size() != 1) continue;  // l'en-tête, ou une ligne abîmée
    LearnSample s;
    int touched = 0;
    const char kind = fields[1].front();
    s.kind = kind == 'f'   ? LearnSample::Kind::Calibration
             : kind == 'k' ? LearnSample::Kind::Approval
                           : LearnSample::Kind::Correction;
    if (kind != 'c' && kind != 'f' && kind != 'k') continue;
    if (!number(fields[0], s.unix_s) || !number(fields[2], touched) || !number(fields[3], s.scene.x) ||
        !number(fields[4], s.scene.glow_seen) || !number(fields[5], s.scene.tv) ||
        !number(fields[6], s.scene.force) || !number(fields[7], s.look.shadows) ||
        !number(fields[8], s.look.colors) || !number(fields[9], s.look.gray))
      continue;
    s.touched = static_cast<std::uint8_t>(touched & kAllFamilies);
    out.push_back(s);
  }
  return out;
}

std::vector<LearnSample> read_learn_samples(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  std::ostringstream text;
  text << in.rdbuf();
  std::vector<LearnSample> all = parse_learn_samples(text.str());
  // Les 5000 derniers : plus de 40 heures de lecture, le reste a de toute façon presque tout perdu de poids.
  constexpr std::size_t kKept = 5000;
  if (all.size() > kKept) all.erase(all.begin(), all.end() - static_cast<std::ptrdiff_t>(kKept));
  return all;
}

void append_learn_sample(const std::filesystem::path& file, const LearnSample& sample) {
  std::error_code error;
  const bool fresh = !std::filesystem::exists(file, error);
  std::ofstream out(file, std::ios::app | std::ios::binary);
  if (!out) return;
  if (fresh) out << learn_header();
  out << learn_line(sample);
}

}  // namespace ajp::app::play
