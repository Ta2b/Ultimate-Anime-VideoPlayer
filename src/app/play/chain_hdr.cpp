// Le HDR de la chaîne : RTX HDR ouvert par le SDK Maxine, le grade de chaque sortie et le vrai
// noir OLED (chain.h). Les actions de l'utilisateur sont dans chain.cpp.
#include <algorithm>
#include <format>
#include <optional>
#include <utility>

#include "app/play/chain.h"
#include "core/clock.h"
#include "core/log.h"
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
  return vendor_hdr::TrueHdrParams{.contrast = s.true_hdr_contrast,
                                   .saturation = s.true_hdr_saturation,
                                   .middle_gray = s.true_hdr_gray,
                                   .peak_nits = s.true_hdr_peak > 0 ? s.true_hdr_peak : auto_peak,
                                   .debanding = s.true_hdr_debanding};
}

// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR, que rend la fonction.
std::expected<int, Refusal> open_rtx_hdr(const std::filesystem::path& sdk_source, const PlayerPaths& paths,
                                         const PlaySettings& settings, int auto_peak, PlaceboScreen& screen,
                                         gpu::KernelOps& kernels, int width, int height) {
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
  if (!sdk.installed) return refuse(RefusalCode::HdrUnavailable, std::format("RTX HDR : {}", sdk.reason));
  auto params = vendor_hdr::validate(true_hdr_params(settings, auto_peak));
  if (!params) return std::unexpected(std::move(params.error()));
  if (auto ok = kernels.open_true_hdr(
          *sdk.installed,
          vendor_hdr::TrueHdrConfig{
              .width = width, .height = height, .source = ColorInfo{}, .params = *params});
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
                             const std::optional<render::BlackThresholds>& black) {
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
                                                                 .source_hdr = false,
                                                                 .output_pq = pq,
                                                                 .enabled = s.grade,
                                                                 .peak_nits = static_cast<double>(peak)});
    if (!decision) {
      log_channel("app")->warn("grade : {}", decision.error().detail);
      return {};
    }
    if (decision->mode != render::GradeMode::Neutral) (pq ? choice.pq : choice.sdr) = decision->params;
  }
  return choice;
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits) {
  std::string text = screen.output_description();
  if (peak_nits > 0) {
    text += std::format(" · RTX HDR, pic {} nits", peak_nits);
  } else if (s.hdr_native && screen.hdr_output()) {
    text += " · HDR adaptatif";
  }
  if (s.grade && (!screen.hdr_output() || peak_nits > 0 || s.hdr_native))
    text += std::format(" · grade {}", render::preset_label(s.grade_preset));
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
  to.screen.set_grade(grade_of(s, to.true_hdr_open, to.peak_nits, to.auto_peak, to.display, to.black));
}

// Vrai noir OLED : recale les seuils sur la rampe de 32 marches mesurée à la sortie (route native :
// les marches SDR ; RTX HDR : celles que gpu a passées par TrueHDR), grade compris, puis pose le
// grade avec eux.
void refresh_black(Chain& to) {
  to.black_dirty = false;
  to.black_at_us = host_now_us();
  const PlaySettings& s = to.running;
  // En SDR, le grade s'applique sur toutes les routes ; en PQ, sur RTX HDR et la route native.
  const bool routed = to.true_hdr_open || s.hdr_native || !to.screen.hdr_output();
  const bool wanted = black_wanted(s, to.display) && s.grade && routed;
  to.live.set_black_wanted(wanted && to.true_hdr_open);
  to.black.reset();
  if (!wanted) {
    set_grade(to, s);
    to.ui.set_black_status(black_wanted(s, to.display)
                               ? "Vrai noir : sans effet ici (grade coupé, ou sortie HDR sans route)."
                               : "Vrai noir : coupé.");
    return;
  }
  std::vector<float> levels;
  if (to.true_hdr_open) {
    if (to.truehdr_ramp.empty()) {
      set_grade(to, s);
      to.ui.set_black_status("Vrai noir : rampe TrueHDR en cours de mesure…");
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
    to.ui.set_black_status("Vrai noir : la rampe ne monte pas, rien n'est changé.");
    return;
  }
  log_channel("app")->info(
      "vrai noir : noir de la source {:.4f} nit, marches 1 et 2 {:.4f} et {:.4f} nit ; 0 sous {:.4f}, pied "
      "jusqu'à {:.4f} nit",
      (*nits)[0], (*nits)[1], (*nits)[2], to.black->lo, to.black->hi);
  to.ui.set_black_status(
      std::format("Vrai noir : 0 sous {:.4f} nit, pied jusqu'à {:.4f} nit (noir de la source "
                  "à {:.4f} nit avant).",
                  to.black->lo, to.black->hi, (*nits)[0]));
}

}  // namespace ajp::app::play
