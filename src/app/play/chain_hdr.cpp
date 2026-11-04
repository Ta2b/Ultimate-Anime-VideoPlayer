// Le HDR de la chaîne : le grade de chaque sortie et le vrai noir OLED (chain.h). Les actions de
// l'utilisateur sont dans chain.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <system_error>
#include <utility>

#include "app/play/chain.h"
#include "app/play/chain_adaptive_log.h"
#include "app/play/chain_shots.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "core/clock.h"
#include "core/log.h"
#include "render/adaptive_rules.h"
#include "render/display_model.h"
#include "render/looks.h"

namespace ajp::app::play {
namespace {

// Le grade HDR de la chaîne : route native si le HDR adaptatif est choisi (le SDR converti en PQ par
// libplacebo, aspect TrueHDR émulé), aucun en SDR. Pic : celui de l'écran (1000 nits s'il ne le dit pas).
// Le vrai noir OLED est voulu : forcé, ou « auto » sur un écran détecté OLED.
bool black_wanted(const PlaySettings& s, const std::optional<DisplayFacts>& display) {
  switch (s.oled_black) {
    case OledBlack::On:
      return true;
    case OledBlack::Off:
      return false;
    case OledBlack::Auto:
      return display && display->oled;
  }
  return false;
}

// Une dalle OLED : détectée, ou affirmée par un vrai noir forcé.
bool oled_panel(const PlaySettings& s, const std::optional<DisplayFacts>& display) {
  return (display && display->oled) || s.oled_black == OledBlack::On;
}

}  // namespace

double glow_weight(const gpu::KernelOps::SceneStats& stats, const render::GradeSettings& g) {
  const auto smooth = [](double edge0, double edge1, double v) {
    const double x = std::clamp((v - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
  };
  const double p50 = static_cast<double>(stats.p50);
  const double spread = static_cast<double>(stats.p995) - p50;
  const double bright = static_cast<double>(stats.bright);
  const double lo = g.adaptive_lo / 100.0;
  const double hi = std::max(g.adaptive_hi / 100.0, lo + 0.01);
  // L'éclat : au moins 0,2 % de points au-dessus de 80 %, guère plus de 12 % (au-delà, une scène claire), et
  // un pic à plus de 0,4 au-dessus du fond.
  namespace rules = render::adaptive_rules;
  return smooth(0.35, 0.60, spread) * smooth(rules::kGlowShareLo, rules::kGlowShareHi, bright) *
         (1.0 - smooth(rules::kGlowSceneLo, rules::kGlowSceneHi, bright)) * (1.0 - smooth(lo, hi, p50));
}

render::GradeChoice grade_of(const PlaySettings& s, int auto_peak, const std::optional<DisplayFacts>& display,
                             const std::optional<render::BlackThresholds>& black, bool source_hdr) {
  const int peak = s.true_hdr_peak > 0 ? s.true_hdr_peak : auto_peak > 0 ? auto_peak : 1000;
  render::GradeSettings g = s.grade_values;
  g.contrast = s.true_hdr_contrast;
  g.saturation = s.true_hdr_saturation;
  g.middle_gray = s.true_hdr_gray;
  // Dalle OLED : détectée, ou affirmée par un vrai noir forcé. Son limiteur suit la moyenne
  // d'image que l'écran annonce, quand il l'annonce.
  g.oled = oled_panel(s, display);
  // La moyenne d'image que l'écran annonce, sinon celle du modèle d'une dalle OLED (H1 : L_max plein écran
  // de la courbe QD-OLED, ramenée au pic visé).
  std::optional<double> average = display ? display->frame_average_nits : std::nullopt;
  if (!average && g.oled)
    average = render::DisplayModel{.panel_peak_nits = static_cast<double>(peak), .oled = true}
                  .frame_average_content_nits();
  if (average) {
    if (const auto abl = render::abl_range(*average)) {
      g.abl_lo = abl->lo;
      g.abl_hi = abl->hi;
    }
  }
  g.true_black = black_wanted(s, display) && black.has_value();
  if (g.true_black) {
    g.black_lo = black->lo;
    g.black_hi = black->hi;
  }
  const core::HdrMode route = s.hdr_native ? core::HdrMode::Native : core::HdrMode::Off;
  render::GradeChoice choice;
  for (const bool pq : {true, false}) {
    auto decision = render::decide_grade(g, render::GradeContext{.route = route,
                                                                 .source_hdr = source_hdr,
                                                                 .output_pq = pq,
                                                                 .enabled = s.grade,
                                                                 .peak_nits = static_cast<double>(peak)});
    if (!decision) {
      log_channel("app")->warn("grade : {}", decision.error().detail);
      return {};
    }
    if (decision->mode != render::GradeMode::Neutral) {
      render::GradeParams params = decision->params;
      // Une LUT sans nom est sautée (intensité 0) ; une seule LUT d'HDR met toute la chaîne en codes PQ.
      const std::vector<LookLayer> slots = look_slots(s);
      constexpr std::array<double render::GradeParams::*, render::kLookSlots> kMixes{
          &render::GradeParams::look_mix,   &render::GradeParams::look_mix2,
          &render::GradeParams::look_mix3,  &render::GradeParams::look_mix4,
          &render::GradeParams::look_mix5,  &render::GradeParams::look_mix6,
          &render::GradeParams::look_mix7,  &render::GradeParams::look_mix8,
          &render::GradeParams::look_mix9,  &render::GradeParams::look_mix10,
          &render::GradeParams::look_mix11, &render::GradeParams::look_mix12,
          &render::GradeParams::look_mix13, &render::GradeParams::look_mix14,
          &render::GradeParams::look_mix15};
      bool any_pq = false;
      for (std::size_t i = 0; i < slots.size(); ++i) {
        params.*kMixes[i] = slots[i].key.empty() ? 0.0 : slots[i].mix;
        any_pq = any_pq || (!slots[i].key.empty() && render::look_is_pq(slots[i].key, ""));
      }
      // Une pile qui contient une LUT d'HDR est étalonnée pour des codes PQ : sur une sortie SDR elle
      // donnerait des couleurs fausses, elle est coupée en entier.
      if (!pq && any_pq) {
        for (std::size_t i = 0; i < slots.size(); ++i) params.*kMixes[i] = 0.0;
      }
      params.look_pq = pq && any_pq ? 1.0 : 0.0;
      (pq ? choice.pq : choice.sdr) = params;
    }
  }
  return choice;
}

std::vector<std::string> apply_looks(PlaceboScreen& screen, const PlaySettings& s,
                                     const std::filesystem::path& luts_dir) {
  const std::vector<LookLayer> keys = look_slots(s);
  std::array<std::optional<render::CubeLut>, render::kLookSlots> loaded;
  std::array<const render::CubeLut*, render::kLookSlots> luts{};
  std::vector<std::string> failed;
  for (std::size_t i = 0; i < render::kLookSlots; ++i) {
    if (i >= keys.size() || keys[i].key.empty()) continue;
    auto lut = load_look(keys[i].key, luts_dir);
    if (!lut) {
      log_channel("app")->warn("look : {}", lut.error().detail);
      failed.push_back(keys[i].key);
      continue;
    }
    log_channel("app")->info("look {} : {} ({} points)", i + 1, keys[i].key, lut->size);
    luts[i] = &loaded[i].emplace(std::move(*lut));
  }
  if (auto set = screen.set_looks(luts); !set) {
    log_channel("app")->warn("look : {}", set.error().detail);
    failed.push_back(set.error().detail);
  }
  return failed;
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s) {
  std::string text = screen.output_description();
  if (s.hdr_native && screen.hdr_output()) {
    text += i18n::tr(" · adaptive HDR");
  }
  if (s.grade) text += i18n::tr_format(" · grade {}", grade_label(s.grade_preset));
  return text;
}

// Soins OLED : la barre s'efface aussi en pause sur une dalle OLED ; la cadence fixe vaut pour
// tout écran (un VRR à fréquence constante).
void apply_oled_care(Chain& to, const PlaySettings& s) {
  to.ui.controls().set_oled_care(s.oled_care && oled_panel(s, to.display));
  to.session.set_steady_cadence(s.steady_cadence);
}

// Le grade de la chaîne avec ses seuils du vrai noir.
void set_grade(Chain& to, const PlaySettings& wanted, bool from_measured_frame) {
  PlaySettings s = wanted;
  s.grade_values.scene_light = to.scene_light;
  if (to.adaptive_shot != 0 && s.grade_values.adaptive) {
    // Le shader du grade reçoit la position du plan analysé à la place de la sienne (render/grade_params.h).
    s.grade_values.adaptive_shot = render::AdaptiveShot{
        .t = to.adaptive_t, .tv = to.adaptive_tv, .glow = to.adaptive_glow, .series = to.adaptive_bs};
  } else {
    s = with_adaptive(s, to.adaptive_t, to.adaptive_tv, to.adaptive_bs, to.adaptive_glow);
  }
  const render::GradeChoice choice = grade_of(s, to.auto_peak, to.display, to.black, to.source_hdr);
  to.live.set_deband_source(wanted.grade);
  // En pause, rien ne se présente de lui-même : le grade passe tout de suite.
  const RingFrame* newest =
      from_measured_frame && !to.session.view().paused ? to.session.newest_frame() : nullptr;
  if (newest != nullptr) {
    to.screen.set_grade_after(choice, *newest);
  } else {
    to.screen.set_grade(choice);
  }
}

PlaySettings with_adaptive(const PlaySettings& s, double t, double tv, double bs, double glow) {
  const render::GradeSettings& g = s.grade_values;
  if (!g.adaptive || !(t > 0.0 || tv > 0.0 || bs > 0.0 || glow > 0.0)) return s;
  bs = std::clamp(bs, 0.0, 1.0);
  t = std::clamp(t, 0.0, 1.0);
  glow = std::clamp(glow, 0.0, 1.0);
  const double vibrance_t_smooth = std::clamp(tv, 0.0, 1.0);
  PlaySettings out = s;
  render::GradeSettings& o = out.grade_values;
  // La scène sombre avec éclat (glow) glisse en plus vers ses propres noirs, couleurs et gris moyen.
  o.shadows = std::lerp(std::lerp(g.shadows, g.adaptive_shadows, t), g.adaptive_glow_shadows, glow);
  // Les scènes claires perdent une part de la vibrance (couleurs chaudes sursaturées), puis la scène bleue
  // saturée glisse vers sa propre valeur.
  const double keep = 1.0 - std::clamp(g.adaptive_vibrance_cut, 0.0, 1.0) * t;
  o.vibrance = std::lerp(std::lerp(g.vibrance * keep, g.adaptive_glow_vibrance, glow), g.adaptive_vibrance,
                         vibrance_t_smooth);
  o.vibrance_rtx = std::lerp(std::lerp(g.vibrance_rtx * keep, g.adaptive_glow_vibrance, glow),
                             g.adaptive_vibrance, vibrance_t_smooth);
  const double gray = std::lerp(std::lerp(static_cast<double>(s.true_hdr_gray), g.adaptive_gray, t),
                                g.adaptive_glow_gray, glow);
  out.true_hdr_gray = static_cast<int>(std::lround(gray));
  // Série claire par nature (luminosité moyenne de l'épisode haute) : plus de contraste et de couleur pour
  // que l'image reste percutante, que ses scènes ordinaires ne soient pas toutes traitées comme « claires ».
  o.punch_base += 0.12 * bs;
  o.chroma_base += 0.06 * bs;
  out.true_hdr_contrast = std::min(out.true_hdr_contrast + static_cast<int>(std::lround(15.0 * bs)), 200);
  return out;
}

namespace {

// La lumière de la scène pour le grade en trois passes : envoyée quand elle change vraiment (3 % de
// luminance moyenne, 2 points de part claire), au plus tous les dixièmes de seconde, sauf à une coupe.
void light_tick(Chain& to, const std::optional<ShotScene>& scene, std::int64_t now_us, bool cut) {
  if (!scene || !to.running.grade) return;
  const render::SceneLight& light = scene->light;
  if (to.scene_light) {
    const render::SceneLight& was = *to.scene_light;
    const bool moved = std::abs(light.apl_nits - was.apl_nits) > std::max(0.03 * was.apl_nits, 0.25) ||
                       std::abs(light.lit - was.lit) > 0.02;
    if (!moved || (!cut && now_us - to.scene_light_at_us < 100'000)) return;
  }
  to.scene_light = light;
  to.scene_light_at_us = now_us;
  set_grade(to, to.running, true);
}

}  // namespace

void adaptive_tick(Chain& to, std::int64_t now_us) {
  const render::GradeSettings& g = to.running.grade_values;
  // Le lecteur décide de la scène : le plan de la pré-analyse (chain_shots.h) s'il est connu, sinon la scène
  // vue à l'ingestion ; le grade d'avant (--legacy-grade) la mesure lui-même faute des deux.
  std::optional<ShotScene> shot;
  feed_plan_shots(to);
  if (g.adaptive && g.adaptive_force < 0.0) shot = next_shot(to, now_us);
  light_tick(to, shot ? shot : next_live(to), now_us, shot && shot->serial != to.adaptive_shot);
  if (!g.adaptive || !shot) {
    to.ui.set_adaptive_scene({});
    to.learn.seen.reset();
    to.learn.pending.reset();
    const bool had_shot = std::exchange(to.adaptive_shot, 0) != 0;
    if (had_shot || to.adaptive_t != 0.0 || to.adaptive_tv != 0.0 || to.adaptive_bs != 0.0 ||
        to.adaptive_glow != 0.0) {
      to.adaptive_bs = 0.0;  // coupé en cours de route : retour aux curseurs
      to.adaptive_glow = 0.0;
      to.adaptive_glow_seen = 0.0;
      to.adaptive_t = 0.0;
      to.adaptive_tv = 0.0;
      to.adaptive_blue = 0.0;
      to.adaptive_chroma = 0.0;
      to.adaptive_luma = -1.0;
      to.adaptive_cut = true;
      set_grade(to, to.running);
    }
    return;
  }
  const gpu::KernelOps::SceneStats* stats = &shot->stats;
  if (to.running.adaptive_record) adaptive_record(to, *stats, now_us);
  const std::optional<float> luma = stats->mean;
  const double glow_now = glow_weight(*stats, g);
  // Une scène sombre avec éclat se classe sur son fond (la médiane), pas sur la moyenne que l'éclat relève.
  const double scene = std::lerp(static_cast<double>(*luma), static_cast<double>(stats->p50), glow_now);
  // Le plan analysé est connu à l'image près : un nouveau plan se traite dès le tour où il arrive, sans
  // attendre le dixième de seconde des tours ordinaires. Ses valeurs (un fondu : celles de l'image) ne se
  // lissent pas.
  const bool new_shot = shot->serial != to.adaptive_shot;
  to.adaptive_shot = shot->serial;
  if (now_us - to.adaptive_at_us < 100'000 && !new_shot) return;
  learn_tick(to, now_us);
  const std::int64_t dt = to.adaptive_at_us == 0 ? 0 : now_us - to.adaptive_at_us;
  to.adaptive_at_us = now_us;
  to.adaptive_luma = scene;
  to.adaptive_cut = true;
  to.adaptive_blue = static_cast<double>(stats->blue);
  to.adaptive_chroma = static_cast<double>(stats->chroma);
  to.adaptive_glow_seen = glow_now;
  // La luminosité de la SÉRIE : la moyenne de l'épisode, lissée sur une à deux minutes (moyenne cumulée au
  // début, puis exponentielle), amorcée à celle de Re:Zero (0,30). Une série claire par nature (Slime : plus
  // de 0,5) ne doit pas voir toutes ses scènes ordinaires traitées comme « claires » : les seuils glissent
  // avec elle, et la série claire reçoit en plus du contraste et de la couleur (with_adaptive). Une scène
  // plus claire que sa série reste « claire ».
  constexpr double kSeriesSeedUs = 20'000'000.0;
  constexpr double kSeriesTauUs = 90'000'000.0;
  to.series_us += static_cast<double>(dt);
  if (to.series_luma < 0.0) to.series_luma = kSeriesRef;
  // La moyenne des plans analysés, dès qu'elle couvre assez de la vidéo ; sinon, celle de la lecture.
  if (const std::optional<double> analyzed = shots_series_luma(to)) {
    to.series_luma = *analyzed;
  } else {
    to.series_luma +=
        std::min(static_cast<double>(dt) / (std::min(to.series_us, kSeriesTauUs) + kSeriesSeedUs), 1.0) *
        (static_cast<double>(*luma) - to.series_luma);
  }
  // Même rendu que Re:Zero (la référence : 0,30 en moyenne, mesuré sur 155 secondes de son épisode) : les
  // seuils suivent la série EN PROPORTION, si bien qu'une scène à 1,45 fois la luminosité moyenne de Slime se
  // règle comme une scène à 1,45 fois celle de Re:Zero.
  const double ratio = series_ratio(to.series_luma);
  namespace rules = render::adaptive_rules;
  const double bs_x = std::clamp(
      (to.series_luma - rules::kBrightSeriesLo) / (rules::kBrightSeriesHi - rules::kBrightSeriesLo), 0.0,
      1.0);
  double bs = bs_x * bs_x * (3.0 - 2.0 * bs_x);
  const double lo = g.adaptive_lo / 100.0 * ratio;
  const double hi = std::max(std::min(g.adaptive_hi / 100.0 * ratio, rules::kBrightCap), lo + 0.01);
  const double x = std::clamp((to.adaptive_luma - lo) / (hi - lo), 0.0, 1.0);
  double t = x * x * (3.0 - 2.0 * x);
  // La vibrance dépend de la couleur, pas de la luminosité : un ciel ou un aplat bleu saturé (« bleu » moyen
  // élevé) n'en veut presque pas, un désert jaune ou des murs ternes en veulent. Seuils mesurés sur les
  // scènes de Tatoub (plein jour bleu : vibrance 0 ; désert, donjon, teints : 0,80).
  const double xv =
      std::clamp((to.adaptive_blue - rules::kBlueLo) / (rules::kBlueHi - rules::kBlueLo), 0.0, 1.0);
  // Un bleu saturé mais SOMBRE (écran d'interface bleu nuit) n'est pas un ciel : le bleu ne compte qu'avec
  // une scène à moitié claire. Et un flash très saturé de n'importe quelle teinte (feu, éclair : chroma
  // moyenne au-dessus de 0,45) n'a pas besoin de vibrance non plus (audit sur trois séries).
  const double gate = std::clamp(t / rules::kBlueGate, 0.0, 1.0);
  const double xc =
      std::clamp((to.adaptive_chroma - rules::kChromaLo) / (rules::kChromaHi - rules::kChromaLo), 0.0, 1.0);
  double tv = std::max(xv * xv * (3.0 - 2.0 * xv) * gate, xc * xc * (3.0 - 2.0 * xc));
  double glow = std::clamp(to.adaptive_glow_seen, 0.0, 1.0) * (1.0 - t);
  if (g.adaptive_force >= 0.0) {  // calibration : un bout de la courbe, sans la couleur ni la série
    t = render::grade_limits::forced_bright(g.adaptive_force) ? 1.0 : 0.0;
    glow = render::grade_limits::forced_glow(g.adaptive_force) ? 1.0 : 0.0;
    tv = 0.0;
    bs = 0.0;
  }
  to.adaptive_forced = g.adaptive_force >= 0.0;
  to.adaptive_cut = false;
  const bool same = std::abs(t - to.adaptive_t) < 0.02 && std::abs(tv - to.adaptive_tv) < 0.02 &&
                    !(t == 0.0 && to.adaptive_t != 0.0) && !(t == 1.0 && to.adaptive_t != 1.0) &&
                    !(tv == 0.0 && to.adaptive_tv != 0.0) && !(tv == 1.0 && to.adaptive_tv != 1.0) &&
                    std::abs(bs - to.adaptive_bs) < 0.02 && std::abs(glow - to.adaptive_glow) < 0.02 &&
                    !(glow == 0.0 && to.adaptive_glow != 0.0) && !(glow == 1.0 && to.adaptive_glow != 1.0);
  if (same) return;
  to.adaptive_t = t;
  to.adaptive_tv = tv;
  to.adaptive_bs = bs;
  to.adaptive_glow = glow;
  // Le plan vient de la pré-analyse, en avance : les images déjà prêtes devant lui sont du plan d'avant, le
  // grade change après elles.
  set_grade(to, to.running, true);
}

void refresh_black(Chain& to) {
  to.black_dirty = false;
  to.black_at_us = host_now_us();
  const PlaySettings& s = to.running;
  // Le grade s'applique sur toutes les routes : HDR adaptatif et mode SDR (en SDR comme en PQ).
  const bool wanted = black_wanted(s, to.display) && s.grade;
  const std::optional<render::BlackThresholds> before = std::exchange(to.black, std::nullopt);
  if (!wanted) {
    set_grade(to, s);
    to.ui.set_black_status(
        black_wanted(s, to.display)
            ? i18n::tr("True black: no effect here (grading off, or HDR output without a path).")
            : i18n::tr("True black: off."));
    return;
  }
  // Recale les seuils sur la rampe de 32 marches SDR mesurée à la sortie, grade compris.
  const auto sdr = black_ramp_levels();
  const std::vector<float> levels(sdr.begin(), sdr.end());
  set_grade(to, s);  // sans seuils : la rampe se mesure sans le vrai noir
  auto nits = to.screen.grade_ramp(levels);
  if (!nits) {
    log_channel("app")->warn("vrai noir : {}", nits.error().detail);
    to.ui.set_black_status("Vrai noir : mesure impossible (voir le journal).");
    return;
  }
  to.black = render::black_thresholds(*nits, s.oled_black_steps);
  set_grade(to, s);
  if (!to.black) {
    to.ui.set_black_status(i18n::tr("True black: the ramp does not rise, nothing changed."));
    return;
  }
  if (before && before->lo == to.black->lo && before->hi == to.black->hi) return;  // rien de neuf à dire
  log_channel("app")->info(
      "vrai noir : noir de la source {:.4f} nit, marches 1 et 2 {:.4f} et {:.4f} nit ; 0 sous {:.4f}, pied "
      "jusqu'à {:.4f} nit",
      (*nits)[0], (*nits)[1], (*nits)[2], to.black->lo, to.black->hi);
  to.ui.set_black_status(
      i18n::tr_format("True black: 0 below {:.4f} nit, toe up to {:.4f} nit (source black was {:.4f} nit).",
                      to.black->lo, to.black->hi, (*nits)[0]));
}

}  // namespace ajp::app::play
