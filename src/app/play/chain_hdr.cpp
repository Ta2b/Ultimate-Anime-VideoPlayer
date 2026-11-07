// Le HDR de la chaîne : le grade de chaque sortie et le vrai noir OLED (chain.h). Les actions de
// l'utilisateur sont dans chain.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "app/play/chain.h"
#include "app/play/chain_shots.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"
#include "render/looks.h"

namespace ajp::app::play {
namespace {

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

// Le grade HDR de la chaîne : route native en HDR (le SDR converti en PQ par libplacebo, puis le grade
// anime guidé), le grade SDR sinon. Pic : celui de l'écran (1000 nits s'il ne le dit pas).
render::GradeChoice grade_of(const PlaySettings& s, int auto_peak, const std::optional<DisplayFacts>& display,
                             const std::optional<render::BlackThresholds>& black, bool source_hdr) {
  const int peak = s.true_hdr_peak > 0 ? s.true_hdr_peak : auto_peak > 0 ? auto_peak : 1000;
  render::GradeSettings g = s.grade_values;
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
      params.look_clean = 1.0;  // audit F1 : grain et paliers non étirés par la pile
      (pq ? choice.pq : choice.sdr) = params;
    }
  }
  // La méthode « HDR anime » vers un écran SDR : son rendu PQ sans la pile de LUT d'HDR (sdr_tonemap la
  // coupe), ramené au SDR plan par plan ; l'écran ne la prend qu'avec ce corps de grade.
  if (!source_hdr && choice.pq) {
    render::GradeParams anime = *choice.pq;
    render::sdr_tonemap(anime);
    choice.sdr_anime = anime;
  }
  return choice;
}

LoadedLooks load_looks(const PlaySettings& s, const std::filesystem::path& luts_dir) {
  const std::vector<LookLayer> keys = look_slots(s);
  LoadedLooks out;
  for (std::size_t i = 0; i < render::kLookSlots; ++i) {
    if (i >= keys.size() || keys[i].key.empty()) continue;
    auto lut = load_look(keys[i].key, luts_dir);
    if (!lut) {
      log_channel("app")->warn("look : {}", lut.error().detail);
      out.failed.push_back(keys[i].key);
      continue;
    }
    log_channel("app")->info("look {} : {} ({} points)", i + 1, keys[i].key, lut->size);
    out.luts.at(i).emplace(std::move(*lut));
  }
  return out;
}

std::vector<std::string> show_looks(PlaceboScreen& screen, LoadedLooks& looks) {
  std::array<const render::CubeLut*, render::kLookSlots> luts{};
  for (std::size_t i = 0; i < render::kLookSlots; ++i)
    if (const std::optional<render::CubeLut>& lut = looks.luts.at(i); lut.has_value()) luts.at(i) = &*lut;
  std::vector<std::string> failed = std::move(looks.failed);
  if (auto set = screen.set_looks(luts); !set) {
    log_channel("app")->warn("look : {}", set.error().detail);
    failed.push_back(set.error().detail);
  }
  return failed;
}

std::vector<std::string> apply_looks(PlaceboScreen& screen, const PlaySettings& s,
                                     const std::filesystem::path& luts_dir) {
  LoadedLooks looks = load_looks(s, luts_dir);
  return show_looks(screen, looks);
}

void start_looks(Chain& to, const PlaySettings& s) {
  auto job = std::make_shared<LookLoad::Job>();
  // Un refus (thread refusé, exception du corps) finit la lecture sans résultat : rien ne l'attend plus.
  ThreadReport report([job](const Refusal& refused) {
    log_channel("app")->warn("looks : {}", refused.detail);
    job->done = true;
  });
  auto thread =
      start_thread("looks", report, [job, s, dir = to.luts_dir](const std::stop_token&, const ThreadReport&) {
        LoadedLooks looks = load_looks(s, dir);
        const std::scoped_lock lock(job->mutex);
        job->result.emplace(std::move(looks));
        job->done = true;
      });
  if (!thread) {
    log_channel("app")->warn("looks : {}", thread.error().detail);
    return;
  }
  to.look_loads.push_back(LookLoad{.job = std::move(job), .thread = std::move(*thread)});
}

void tend_looks(Chain& to, std::int64_t now_us) {
  if (to.look_loads.empty()) return;
  if (!to.look_loads.back().job->done) {
    // Les lectures d'avant finies ne servent plus : leur fil se joint tout de suite.
    for (std::size_t i = to.look_loads.size() - 1; i-- > 0;)
      if (to.look_loads[i].job->done)
        to.look_loads.erase(to.look_loads.begin() + static_cast<std::ptrdiff_t>(i));
    return;
  }
  std::optional<LoadedLooks> looks;
  {
    LookLoad::Job& job = *to.look_loads.back().job;
    const std::scoped_lock lock(job.mutex);
    looks = std::move(job.result);
  }
  to.look_loads.clear();  // la dernière est finie ; les autres se joignent (leur lecture se termine)
  if (!looks) return;
  for (const std::string& failed : show_looks(to.screen, *looks))
    to.ui.controls().notify(i18n::ui_tr_format("Look unreadable: {}", failed), now_us);
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, bool source_hdr) {
  std::string text = screen.output_description();
  if (s.hdr_native && screen.hdr_output()) {
    text += i18n::tr(" · HDR");
  }
  if (s.grade && source_hdr && !s.grade_values.grade_native_hdr)
    text += i18n::tr(" · grade off: HDR source");
  else if (s.grade)
    text += i18n::tr(" · grade");
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
  s.grade_values.anime_exposure = to.anime_e;
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

namespace {

// La lumière de la scène pour le grade en trois passes : envoyée quand elle change vraiment (3 % de
// luminance moyenne, 2 points de part claire), au plus tous les dixièmes de seconde, sauf à une coupe.
// Sous --anime-shots (méga plan A2), un plan de la pré-analyse fixe aussi l'exposition de la méthode « HDR
// anime », et l'une et l'autre glissent aux coupes (anime_shots.h) : envoyées à chaque tour du glissé.
void light_tick(Chain& to, const std::optional<ShotScene>& scene, std::int64_t now_us, bool cut,
                bool planned) {
  if (!scene || !to.running.grade) return;
  render::SceneLight light = scene->light;
  if (to.anime_shots && planned) {
    to.anime_glide.aim({.e = anime_exposure(light.apl_nits, to.anime_series_apl), .light = light}, now_us,
                       cut);
    const AnimeShotLook shown = to.anime_glide.at(now_us);
    const bool moved =
        std::abs(shown.e - to.anime_e) > 0.005 || !to.scene_light || shown.light != *to.scene_light;
    if (!moved) return;
    to.anime_e = shown.e;
    to.scene_light = shown.light;
    to.scene_light_at_us = now_us;
    set_grade(to, to.running, true);
    return;
  }
  to.anime_e = 0.0;  // sans plan analysé : le relèvement fixe
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

void grade_tick(Chain& to, std::int64_t now_us) {
  tend_looks(to, now_us);  // un look changé : posé une fois lu (1.11)
  // La scène : le plan de la pré-analyse (chain_shots.h) s'il est connu, sinon la scène vue à l'ingestion.
  feed_plan_shots(to);
  const std::optional<ShotScene> shot = next_shot(to, now_us);
  const bool new_shot = shot && shot->serial != to.shot_serial;
  to.shot_serial = shot ? shot->serial : 0;
  light_tick(to, shot ? shot : next_live(to), now_us, new_shot, shot.has_value());
}

void refresh_black(Chain& to) {
  to.black_dirty = false;
  to.black_at_us = host_now_us();
  const PlaySettings& s = to.running;
  // Le grade s'applique sur toutes les routes : HDR et SDR (en SDR comme en PQ).
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
