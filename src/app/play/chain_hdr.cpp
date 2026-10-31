// Le HDR de la chaîne : RTX HDR ouvert par le SDK Maxine, le grade de chaque sortie et le vrai
// noir OLED (chain.h). Les actions de l'utilisateur sont dans chain.cpp.
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
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "core/clock.h"
#include "core/log.h"
#include "render/looks.h"
#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Le grade HDR de la chaîne : route TrueHDR si RTX HDR est ouvert, native si le HDR adaptatif est choisi
// (le SDR converti en PQ par libplacebo, aspect TrueHDR émulé avec les réglages de RTX HDR), aucun
// en SDR. Pic : celui de RTX HDR, ou celui de l'écran (1000 nits s'il ne le dit pas).
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

// Couleur de la sortie de RTX HDR : PQ BT.2020 au pic donné à TrueHDR.
ColorInfo pq_color(int peak_nits) {
  return ColorInfo{.matrix = ColorMatrix::Rgb,
                   .range = ColorRange::Full,
                   .transfer = Transfer::Pq,
                   .primaries = Primaries::Bt2020,
                   .hdr = HdrMetadata{.max_luminance = static_cast<float>(peak_nits)}};
}

vendor_hdr::TrueHdrParams true_hdr_params(const PlaySettings& s, int auto_peak) {
  const int sdk_max = static_cast<int>(render::grade_limits::kSdkContrastMax);
  return vendor_hdr::TrueHdrParams{.contrast = std::min(s.true_hdr_contrast, sdk_max),
                                   .saturation = std::min(s.true_hdr_saturation, sdk_max),
                                   .middle_gray = s.true_hdr_gray,
                                   .peak_nits = s.true_hdr_peak > 0 ? s.true_hdr_peak : auto_peak,
                                   .debanding = s.true_hdr_debanding};
}

// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR, que rend la fonction.
std::expected<int, Refusal> open_rtx_hdr(const std::filesystem::path& sdk_source, const PlayerPaths& paths,
                                         const PlaySettings& settings, int auto_peak, PlaceboScreen& screen,
                                         gpu::KernelOps& kernels, int width, int height,
                                         const ColorInfo& source) {
  const auto log = log_channel("app");
  vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  vendor_hdr::SdkStatus sdk = store.status();
  if (!sdk.installed) {
    log->info("RTX HDR : import du SDK Maxine depuis {} (environ 135 Mo, une seule fois)",
              sdk_source.string());
    auto imported = store.import(sdk_source);
    if (!imported) return std::unexpected(std::move(imported.error()));
    sdk = *imported;
  }
  if (!sdk.installed) return refuse(RefusalCode::HdrUnavailable, i18n::tr_format("RTX HDR: {}", sdk.reason));
  auto params = vendor_hdr::validate(true_hdr_params(settings, auto_peak));
  if (!params) return std::unexpected(std::move(params.error()));
  if (auto ok = kernels.open_true_hdr(
          *sdk.installed,
          vendor_hdr::TrueHdrConfig{.width = width, .height = height, .source = source, .params = *params});
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  screen.set_source_color(pq_color(params->peak_nits));
  log->info("RTX HDR : Maxine {} · pic {} nits · contraste {} · saturation {} · gris moyen {} · débandage {}",
            sdk.installed->version().text(), params->peak_nits, params->contrast, params->saturation,
            params->middle_gray, params->debanding ? "oui" : "non");
  return params->peak_nits;
}

render::GradeChoice grade_of(const PlaySettings& s, bool true_hdr_open, int peak_nits, int auto_peak,
                             const std::optional<DisplayFacts>& display,
                             const std::optional<render::BlackThresholds>& black, bool source_hdr) {
  const int peak = true_hdr_open && peak_nits > 0 ? peak_nits
                   : s.true_hdr_peak > 0          ? s.true_hdr_peak
                   : auto_peak > 0                ? auto_peak
                                                  : 1000;
  render::GradeSettings g = s.grade_values;
  g.contrast = s.true_hdr_contrast;
  g.saturation = s.true_hdr_saturation;
  g.middle_gray = s.true_hdr_gray;
  // Dalle OLED : détectée, ou affirmée par un vrai noir forcé. Son limiteur suit la moyenne
  // d'image que l'écran annonce, quand il l'annonce.
  g.oled = oled_panel(s, display);
  if (display && display->frame_average_nits) {
    if (const auto abl = render::abl_range(*display->frame_average_nits)) {
      g.abl_lo = abl->lo;
      g.abl_hi = abl->hi;
    }
  }
  g.true_black = black_wanted(s, display) && black.has_value();
  if (g.true_black) {
    g.black_lo = black->lo;
    g.black_hi = black->hi;
  }
  const core::HdrMode route = true_hdr_open  ? core::HdrMode::TrueHdr
                              : s.hdr_native ? core::HdrMode::Native
                                             : core::HdrMode::Off;
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
    if (pq) {
      // Les valeurs du grade sur la sortie PQ, dites une fois par changement : de quoi comprendre une image
      // trop claire.
      static std::string last_said;
      const std::string said = std::format(
          "grade PQ : mode {} · pic {:.0f} nits · blanc papier {:.0f} nits · blanc de la source {:.0f} nits "
          "· "                            // i18n-ok : journal
          "adaptatif {} · émulation {}",  // i18n-ok : journal
          static_cast<int>(decision->mode), decision->params.peak, decision->params.paper,
          decision->source_white_nits, decision->params.ad_on > 0.5, decision->params.emulate > 0.5);
      if (said != last_said) {
        last_said = said;
        log_channel("app")->info("{}", said);
      }
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
    loaded[i] = std::move(*lut);
    luts[i] = &*loaded[i];
  }
  if (auto set = screen.set_looks(luts); !set) {
    log_channel("app")->warn("look : {}", set.error().detail);
    failed.push_back(set.error().detail);
  }
  return failed;
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits) {
  std::string text = screen.output_description();
  if (peak_nits > 0) {
    text += i18n::tr_format(" · RTX HDR, peak {} nits", peak_nits);
  } else if (s.hdr_native && screen.hdr_output()) {
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
void set_grade(Chain& to, const PlaySettings& wanted) {
  const PlaySettings s = with_adaptive(wanted, to.adaptive_t, to.adaptive_tv, to.adaptive_bs);
  to.screen.set_grade(
      grade_of(s, to.true_hdr_open, to.peak_nits, to.auto_peak, to.display, to.black, to.source_hdr));
}

// Vrai noir OLED : recale les seuils sur la rampe de 32 marches mesurée à la sortie (route native :
// les marches SDR ; RTX HDR : celles que gpu a passées par TrueHDR), grade compris, puis pose le
// grade avec eux.
PlaySettings with_adaptive(const PlaySettings& s, double t, double tv, double bs) {
  const render::GradeSettings& g = s.grade_values;
  if (!g.adaptive || !(t > 0.0 || tv > 0.0 || bs > 0.0)) return s;
  bs = std::clamp(bs, 0.0, 1.0);
  t = std::clamp(t, 0.0, 1.0);
  const double vibrance_t_smooth = std::clamp(tv, 0.0, 1.0);
  PlaySettings out = s;
  render::GradeSettings& o = out.grade_values;
  o.shadows = std::lerp(g.shadows, g.adaptive_shadows, t);
  o.vibrance = std::lerp(g.vibrance, g.adaptive_vibrance, vibrance_t_smooth);
  o.vibrance_rtx = std::lerp(g.vibrance_rtx, g.adaptive_vibrance, vibrance_t_smooth);
  out.true_hdr_gray =
      static_cast<int>(std::lround(std::lerp(static_cast<double>(s.true_hdr_gray), g.adaptive_gray, t)));
  // Série claire par nature (luminosité moyenne de l'épisode haute) : plus de contraste et de couleur pour
  // que l'image reste percutante, que ses scènes ordinaires ne soient pas toutes traitées comme « claires ».
  o.punch_base += 0.12 * bs;
  o.chroma_base += 0.06 * bs;
  out.true_hdr_contrast = std::min(out.true_hdr_contrast + static_cast<int>(std::lround(15.0 * bs)), 200);
  return out;
}

// Apprentissage : une ligne par seconde de lecture dans adaptive-log.tsv (statistiques de la scène, réglages
// courants de l'utilisateur et position t du grade adaptatif), pour ajuster des courbes sur ses réglages
// faits à la main.
void adaptive_record(Chain& to, const gpu::KernelOps::SceneStats& stats, std::int64_t now_us) {
  if (to.adaptive_log.empty() || to.session.view().paused || now_us - to.adaptive_log_us < 1'000'000) return;
  to.adaptive_log_us = now_us;
  std::error_code error;
  const bool fresh = !std::filesystem::exists(to.adaptive_log, error);
  std::ofstream out(to.adaptive_log, std::ios::app);
  if (!out) return;
  if (fresh)
    out << "unix_s\tfichier\tposition_s\tmoyenne\tp05\tp50\tp95\tp995\tpart_noir\tpart_"
           "clair\tnoirs\tcouleurs\teclats\t"
           "aplats\tetendues\tlimiteur\tgris_moyen\tcontraste\tsaturation\tadaptatif\tt\ttv\tserie\tluma_"
           "serie\n";
  const PlaySettings& s = to.running;
  const render::GradeSettings& g = s.grade_values;
  out << std::time(nullptr) << '\t' << to.file.filename().string() << '\t'
      << static_cast<double>(std::max<std::int64_t>(to.session.view().position_us, 0)) / 1e6 << '\t'
      << stats.mean << '\t' << stats.p05 << '\t' << stats.p50 << '\t' << stats.p95 << '\t' << stats.p995
      << '\t' << stats.dark << '\t' << stats.bright << '\t' << stats.blue << '\t' << stats.chroma << '\t'
      << g.shadows << '\t' << g.vibrance_rtx << '\t' << g.highlights_rtx << '\t' << g.area_rtx << '\t'
      << g.expand << '\t' << g.guard << '\t' << s.true_hdr_gray << '\t' << s.true_hdr_contrast << '\t'
      << s.true_hdr_saturation << '\t' << (g.adaptive ? 1 : 0) << '\t' << to.adaptive_t << '\t'
      << to.adaptive_tv << '\t' << to.adaptive_bs << '\t' << to.series_luma << '\n';
}

void adaptive_tick(Chain& to, std::int64_t now_us) {
  if (to.running.adaptive_record && to.kernels != nullptr && to.true_hdr_open) {
    if (const auto stats = to.kernels->scene_stats()) adaptive_record(to, *stats, now_us);
  }
  const render::GradeSettings& g = to.running.grade_values;
  if (!g.adaptive || to.kernels == nullptr || !to.true_hdr_open) {
    if (to.adaptive_t != 0.0 || to.adaptive_tv != 0.0 || to.adaptive_bs != 0.0) {
      to.adaptive_bs = 0.0;  // coupé en cours de route : retour aux curseurs
      to.adaptive_t = 0.0;
      to.adaptive_tv = 0.0;
      to.adaptive_blue = 0.0;
      to.adaptive_chroma = 0.0;
      to.adaptive_luma = -1.0;
      set_grade(to, to.running);
      to.live.set_true_hdr(true_hdr_params(to.running, to.auto_peak));
    }
    return;
  }
  if (now_us - to.adaptive_at_us < 100'000) return;
  const std::int64_t dt = to.adaptive_at_us == 0 ? 0 : now_us - to.adaptive_at_us;
  to.adaptive_at_us = now_us;
  const auto stats = to.kernels->scene_stats();
  if (!stats) return;
  const std::optional<float> luma = stats->mean;
  constexpr double kTauUs = 1'200'000.0;
  constexpr double kCut = 0.20;  // un saut de luminance moyenne : coupe de scène, la mesure repart
  const double follow = 1.0 - std::exp(-static_cast<double>(dt) / kTauUs);
  if (to.adaptive_luma < 0.0 || std::abs(static_cast<double>(*luma) - to.adaptive_luma) > kCut) {
    to.adaptive_luma = static_cast<double>(*luma);
    to.adaptive_blue = static_cast<double>(stats->blue);
    to.adaptive_chroma = static_cast<double>(stats->chroma);
  } else {
    to.adaptive_luma += follow * (static_cast<double>(*luma) - to.adaptive_luma);
    to.adaptive_blue += follow * (static_cast<double>(stats->blue) - to.adaptive_blue);
    to.adaptive_chroma += follow * (static_cast<double>(stats->chroma) - to.adaptive_chroma);
  }
  // La luminosité de la SÉRIE : la moyenne de l'épisode, lissée sur une à deux minutes (moyenne cumulée au
  // début, puis exponentielle), amorcée à celle de Re:Zero (0,30). Une série claire par nature (Slime : plus
  // de 0,5) ne doit pas voir toutes ses scènes ordinaires traitées comme « claires » : les seuils glissent
  // avec elle, et la série claire reçoit en plus du contraste et de la couleur (with_adaptive). Une scène
  // plus claire que sa série reste « claire ».
  constexpr double kSeriesRef = 0.30;
  constexpr double kSeriesSeedUs = 20'000'000.0;
  constexpr double kSeriesTauUs = 90'000'000.0;
  to.series_us += static_cast<double>(dt);
  if (to.series_luma < 0.0) to.series_luma = kSeriesRef;
  to.series_luma +=
      std::min(static_cast<double>(dt) / (std::min(to.series_us, kSeriesTauUs) + kSeriesSeedUs), 1.0) *
      (static_cast<double>(*luma) - to.series_luma);
  // Même rendu que Re:Zero (la référence : 0,30 en moyenne, mesuré sur 155 secondes de son épisode) : les
  // seuils suivent la série EN PROPORTION, si bien qu'une scène à 1,45 fois la luminosité moyenne de Slime se
  // règle comme une scène à 1,45 fois celle de Re:Zero.
  const double ratio = std::clamp(to.series_luma / kSeriesRef, 0.7, 1.8);
  const double bs_x = std::clamp((to.series_luma - 0.36) / 0.14, 0.0, 1.0);
  const double bs = bs_x * bs_x * (3.0 - 2.0 * bs_x);
  const double lo = g.adaptive_lo / 100.0 * ratio;
  const double hi = std::max(std::min(g.adaptive_hi / 100.0 * ratio, 0.95), lo + 0.01);
  const double x = std::clamp((to.adaptive_luma - lo) / (hi - lo), 0.0, 1.0);
  const double t = x * x * (3.0 - 2.0 * x);
  // La vibrance dépend de la couleur, pas de la luminosité : un ciel ou un aplat bleu saturé (« bleu » moyen
  // élevé) n'en veut presque pas, un désert jaune ou des murs ternes en veulent. Seuils mesurés sur les
  // scènes de Tatoub (plein jour bleu : vibrance 0 ; désert, donjon, teints : 0,80).
  constexpr double kBlueLo = 0.05;
  constexpr double kBlueHi = 0.13;
  const double xv = std::clamp((to.adaptive_blue - kBlueLo) / (kBlueHi - kBlueLo), 0.0, 1.0);
  // Un bleu saturé mais SOMBRE (écran d'interface bleu nuit) n'est pas un ciel : le bleu ne compte qu'avec
  // une scène à moitié claire. Et un flash très saturé de n'importe quelle teinte (feu, éclair : chroma
  // moyenne au-dessus de 0,45) n'a pas besoin de vibrance non plus (audit sur trois séries).
  const double gate = std::clamp(t / 0.5, 0.0, 1.0);
  const double xc = std::clamp((to.adaptive_chroma - 0.40) / 0.20, 0.0, 1.0);
  const double tv = std::max(xv * xv * (3.0 - 2.0 * xv) * gate, xc * xc * (3.0 - 2.0 * xc));
  const bool same = std::abs(t - to.adaptive_t) < 0.02 && std::abs(tv - to.adaptive_tv) < 0.02 &&
                    !(t == 0.0 && to.adaptive_t != 0.0) && !(t == 1.0 && to.adaptive_t != 1.0) &&
                    !(tv == 0.0 && to.adaptive_tv != 0.0) && !(tv == 1.0 && to.adaptive_tv != 1.0) &&
                    std::abs(bs - to.adaptive_bs) < 0.02;
  if (same) return;
  const vendor_hdr::TrueHdrParams before =
      true_hdr_params(with_adaptive(to.running, to.adaptive_t, to.adaptive_tv, to.adaptive_bs), to.auto_peak);
  to.adaptive_t = t;
  to.adaptive_tv = tv;
  to.adaptive_bs = bs;
  const vendor_hdr::TrueHdrParams after = true_hdr_params(with_adaptive(to.running, t, tv, bs), to.auto_peak);
  if (after.middle_gray != before.middle_gray) {
    to.live.set_true_hdr(after);
    if (to.session.view().paused) to.hdr_refresh_us = now_us + 120'000;
  }
  set_grade(to, to.running);
}

void refresh_black(Chain& to) {
  to.black_dirty = false;
  to.black_at_us = host_now_us();
  const PlaySettings& s = to.running;
  // Le grade s'applique sur toutes les routes : RTX HDR, HDR adaptatif et mode SDR (en SDR comme en PQ).
  const bool routed = true;
  const bool wanted = black_wanted(s, to.display) && s.grade && routed;
  to.live.set_black_wanted(wanted && to.true_hdr_open);
  to.black.reset();
  if (!wanted) {
    set_grade(to, s);
    to.ui.set_black_status(
        black_wanted(s, to.display)
            ? i18n::tr("True black: no effect here (grading off, or HDR output without a path).")
            : i18n::tr("True black: off."));
    return;
  }
  std::vector<float> levels;
  if (to.true_hdr_open) {
    if (to.truehdr_ramp.empty()) {
      set_grade(to, s);
      to.ui.set_black_status(i18n::tr("True black: measuring the TrueHDR ramp…"));
      return;
    }
    levels = to.truehdr_ramp;
  } else {
    const auto sdr = black_ramp_levels();
    levels.assign(sdr.begin(), sdr.end());
  }
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
  log_channel("app")->info(
      "vrai noir : noir de la source {:.4f} nit, marches 1 et 2 {:.4f} et {:.4f} nit ; 0 sous {:.4f}, pied "
      "jusqu'à {:.4f} nit",
      (*nits)[0], (*nits)[1], (*nits)[2], to.black->lo, to.black->hi);
  to.ui.set_black_status(
      i18n::tr_format("True black: 0 below {:.4f} nit, toe up to {:.4f} nit (source black was {:.4f} nit).",
                      to.black->lo, to.black->hi, (*nits)[0]));
}

}  // namespace ajp::app::play
