// Le HDR de la chaîne : RTX HDR ouvert par le SDK Maxine, le grade de chaque sortie et le vrai
// noir OLED (chain.h). Les actions de l'utilisateur sont dans chain.cpp.
#include <algorithm>
#include <format>
#include <optional>
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
    if (decision->mode != render::GradeMode::Neutral) {
      render::GradeParams params = decision->params;
      // Une LUT sans nom est sautée (intensité 0) ; une seule LUT d'HDR met toute la chaîne en codes PQ.
      params.look_mix = s.look.empty() ? 0.0 : s.look_mix;
      params.look_mix2 = s.look2.empty() ? 0.0 : s.look2_mix;
      params.look_mix3 = s.look3.empty() ? 0.0 : s.look3_mix;
      const bool any_pq = (!s.look.empty() && render::look_is_pq(s.look, "")) ||
                          (!s.look2.empty() && render::look_is_pq(s.look2, "")) ||
                          (!s.look3.empty() && render::look_is_pq(s.look3, ""));
      params.look_pq = pq && any_pq ? 1.0 : 0.0;
      (pq ? choice.pq : choice.sdr) = params;
    }
  }
  return choice;
}

std::vector<std::string> apply_looks(PlaceboScreen& screen, const PlaySettings& s,
                                     const std::filesystem::path& luts_dir) {
  const std::array<const std::string*, render::kLookSlots> keys{&s.look, &s.look2, &s.look3};
  std::array<std::optional<render::CubeLut>, render::kLookSlots> loaded;
  std::array<const render::CubeLut*, render::kLookSlots> luts{};
  std::vector<std::string> failed;
  for (std::size_t i = 0; i < render::kLookSlots; ++i) {
    if (keys[i]->empty()) continue;
    auto lut = load_look(*keys[i], luts_dir);
    if (!lut) {
      log_channel("app")->warn("look : {}", lut.error().detail);
      failed.push_back(*keys[i]);
      continue;
    }
    log_channel("app")->info("look {} : {} ({} points)", i + 1, *keys[i], lut->size);
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
void set_grade(Chain& to, const PlaySettings& s) {
  to.screen.set_grade(
      grade_of(s, to.true_hdr_open, to.peak_nits, to.auto_peak, to.display, to.black, to.source_hdr));
}

// Vrai noir OLED : recale les seuils sur la rampe de 32 marches mesurée à la sortie (route native :
// les marches SDR ; RTX HDR : celles que gpu a passées par TrueHDR), grade compris, puis pose le
// grade avec eux.
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
