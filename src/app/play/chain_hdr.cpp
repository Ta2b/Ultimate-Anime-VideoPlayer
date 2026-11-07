// Le HDR de la chaîne : le grade de chaque sortie et le vrai noir OLED (chain.h). Les actions de
// l'utilisateur sont dans chain.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
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

#include <SDL3/SDL_video.h>

#include "app/play/chain.h"
#include "app/play/chain_shots.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"
#include "render/cube_lut.h"

namespace ajp::app::play {
namespace {

inline constexpr double kFlashPeakNits = 250.0;  // plafond des flashs dangereux (P3.51)

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
      bool any_pq = false;
      for (std::size_t i = 0; i < slots.size(); ++i) {
        params.look_mixes[i] = slots[i].key.empty() ? 0.0 : slots[i].mix;
        any_pq = any_pq || (!slots[i].key.empty() && render::look_is_pq(slots[i].key, ""));
      }
      // Une pile qui contient une LUT d'HDR est étalonnée pour des codes PQ : sur une sortie SDR elle
      // donnerait des couleurs fausses, elle est coupée en entier.
      if (!pq && any_pq) {
        for (std::size_t i = 0; i < slots.size(); ++i) params.look_mixes[i] = 0.0;
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

struct LookLoader::State {
  std::mutex mutex;
  std::condition_variable_any wake;
  std::optional<std::pair<PlaySettings, std::filesystem::path>> pending;  // sous mutex
  std::optional<LoadedLooks> done;                                        // sous mutex
  std::uint64_t asked = 0;                                                // sous mutex : demandes reçues
  std::uint64_t finished = 0;  // sous mutex : la demande dont done est le résultat
  bool busy = false;           // sous mutex : une lecture en cours
};

LookLoader::~LookLoader() {
  if (!thread_.joinable()) return;
  thread_.request_stop();
  bool busy = false;
  {
    const std::scoped_lock lock(state_->mutex);
    busy = state_->busy;
  }
  // Une lecture en cours finit seule (le fil garde l'état) : la fermeture de la chaîne ne l'attend pas.
  if (busy) thread_.detach();
}

void LookLoader::request(const PlaySettings& s, const std::filesystem::path& luts_dir) {
  if (!state_) state_ = std::make_shared<State>();
  {
    const std::scoped_lock lock(state_->mutex);
    state_->pending.emplace(s, luts_dir);
    ++state_->asked;
  }
  state_->wake.notify_one();
  if (thread_.joinable()) return;
  auto started = start_thread(
      "looks",
      ThreadReport([](const Refusal& refused) { log_channel("app")->warn("looks : {}", refused.detail); }),
      [state = state_](const std::stop_token& stop, const ThreadReport&) {
        std::unique_lock lock(state->mutex);
        while (state->wake.wait(lock, stop, [&] { return state->pending.has_value(); })) {
          auto [settings, dir] = std::move(*state->pending);
          state->pending.reset();
          const std::uint64_t asked = state->asked;
          state->busy = true;
          lock.unlock();
          LoadedLooks looks = load_looks(settings, dir);
          lock.lock();
          state->busy = false;
          state->done.emplace(std::move(looks));
          state->finished = asked;
        }
      });
  if (started) {
    thread_ = std::move(*started);
    return;
  }
  // Sans fil : la lecture se fait ici, comme avant 1.11.
  log_channel("app")->warn("looks : {}", started.error().detail);
  const std::scoped_lock lock(state_->mutex);
  auto [settings, dir] = std::move(*state_->pending);
  state_->pending.reset();
  state_->done.emplace(load_looks(settings, dir));
  state_->finished = state_->asked;
}

std::optional<LoadedLooks> LookLoader::take() {
  if (!state_) return std::nullopt;
  const std::scoped_lock lock(state_->mutex);
  if (!state_->done || state_->finished != state_->asked) return std::nullopt;  // une plus récente vient
  std::optional<LoadedLooks> looks = std::move(state_->done);
  state_->done.reset();
  return looks;
}

void start_looks(Chain& to, const PlaySettings& s) {
  to.looks.request(s, to.luts_dir);
}

void tend_looks(Chain& to, std::int64_t now_us) {
  std::optional<LoadedLooks> looks = to.looks.take();
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
  s.grade_values.series_apl_nits = to.anime_series_apl;
  render::GradeChoice choice = grade_of(s, to.auto_peak, to.display, to.black, to.source_hdr);
  // Flashs dangereux (P3.51) : le grade HDR plafonne vers 250 nits, le plein champ d'un écran OLED.
  if (to.flash_limited && choice.pq) choice.pq->peak = std::min(choice.pq->peak, kFlashPeakNits);
  to.live.set_deband_source(wanted.deband);  // la case Débandage, sur tous les moteurs (P3.56)
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
  // P4.2 : fenêtre réduite ou cachée, l'interpolation s'arrête (rien n'est vu) ; l'audio et SPAN continuent,
  // la reprise est immédiate.
  if (SDL_Window* window = to.screen.window()) {
    const bool hidden = (SDL_GetWindowFlags(window) & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_OCCLUDED)) != 0;
    if (hidden != to.live.hidden())
      log_channel("app")->info("fenêtre {} : interpolation {}", hidden ? "cachée" : "revenue",
                               hidden ? "suspendue" : "reprise");
    to.live.set_hidden(hidden);
  }
  tend_looks(to, now_us);  // un look changé : posé une fois lu (1.11)
  // La scène : le plan de la pré-analyse (chain_shots.h) s'il est connu, sinon la scène vue à l'ingestion.
  feed_plan_shots(to);
  const std::optional<ShotScene> shot = next_shot(to, now_us);
  const bool new_shot = shot && shot->serial != to.shot_serial;
  to.shot_serial = shot ? shot->serial : 0;
  if (const bool flashing = to.running.grade && next_flashing(to); flashing != to.flash_limited) {
    to.flash_limited = flashing;
    if (flashing)
      log_channel("app")->info("flashs rapides : pic du grade plafonné à {:.0f} nits", kFlashPeakNits);
    if (!flashing) log_channel("app")->info("flashs finis : pic du grade rendu");
    set_grade(to, to.running, true);
  }
  light_tick(to, shot ? shot : next_live(to), now_us, new_shot, shot.has_value());
}

void follow_display(Chain& to, std::optional<double> peak_nits) {
  const int peak = auto_peak_nits(peak_nits);
  std::string connector = to.display ? to.display->connector : std::string();
  // Le premier écran vu est celui de l'ouverture : il ne compte pas comme un changement.
  const bool moved = !to.display_seen.empty() && connector != to.display_seen;
  to.display_seen = connector;
  if (!moved && peak == to.auto_peak) return;
  log_channel("app")->info("écran suivi : {}, pic {} nits", connector.empty() ? "?" : connector, peak);
  to.auto_peak = peak;
  apply_oled_care(to, to.running);
  to.black_dirty = moved || to.black_dirty;
  set_grade(to, to.running);
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
