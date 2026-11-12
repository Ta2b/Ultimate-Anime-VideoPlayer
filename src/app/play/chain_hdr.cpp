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
#include "core/thread_role.h"
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
// anime guidé), le même rendu ramené au SDR sinon. Pic : celui de l'écran (1000 nits s'il ne le dit pas).
void cap_flash_peak(render::GradeChoice& choice) {
  if (!choice.pq || choice.pq->peak <= kFlashPeakNits) return;
  choice.pq->uncapped_peak = choice.pq->peak;  // l'annonce HDR garde le pic (B41-03)
  choice.pq->peak = kFlashPeakNits;
}

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
  auto decision = render::decide_grade(g, render::GradeContext{.route = route,
                                                               .source_hdr = source_hdr,
                                                               .output_pq = true,
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
    params.look_pq = any_pq ? 1.0 : 0.0;
    params.look_clean = 1.0;  // audit F1 : grain et paliers non étirés par la pile
    params.surface_white = std::clamp(s.surface_white, kSurfaceWhite.lo, kSurfaceWhite.hi) / 100.0;
    choice.pq = params;
  }
  // La méthode « HDR anime » vers un écran SDR : son rendu PQ sans la pile de LUT d'HDR (sdr_tonemap la
  // coupe : étalonnée pour des codes PQ, elle donnerait des couleurs fausses), ramené au SDR plan par plan.
  if (choice.pq) {
    render::GradeParams anime = *choice.pq;
    render::sdr_tonemap(anime);
    choice.sdr_anime = anime;
  }
  // En HDR, la luminosité ne suit ni le plan ni la scène : les surfaces gardent le relèvement fixe et les
  // éclats la marge entière, la dalle limite elle-même les scènes claires (docs/hdr-anime.md). Amplifiées par
  // la pile de LUT, l'exposition par plan et la marge selon la scène faisaient sauter un même décor d'un plan
  // à l'autre (retour d'essai). La sortie SDR garde sa courbe par plan.
  if (choice.pq) render::steady_hdr(*choice.pq);
  return choice;
}

LoadedLooks load_looks(const PlaySettings& s, const std::filesystem::path& luts_dir) {
  const std::vector<LookLayer> keys = look_slots(s);
  LoadedLooks out;
  for (std::size_t i = 0; i < render::kLookSlots; ++i) {
    if (i >= keys.size() || keys[i].key.empty()) continue;
    // La pile répète ses fichiers (le contraste y est sept fois) : chacun n'est lu qu'une fois, et partagé.
    for (std::size_t j = 0; j < i && !out.luts.at(i); ++j)
      if (keys[j].key == keys[i].key) out.luts.at(i) = out.luts.at(j);
    if (out.luts.at(i)) continue;
    if (std::ranges::find(out.failed, keys[i].key) != out.failed.end()) continue;  // déjà refusé, déjà dit
    auto lut = load_look(keys[i].key, luts_dir);
    if (!lut) {
      log_channel("app")->warn("look : {}", lut.error().detail);
      out.failed.push_back(keys[i].key);
      continue;
    }
    log_channel("app")->info("look : {} ({} points)", keys[i].key, lut->size);
    out.luts.at(i) = std::make_shared<const render::CubeLut>(std::move(*lut));
  }
  // La pile est un tout : sans l'un de ses fichiers, les autres donneraient une autre image. Aucun look.
  if (!out.failed.empty()) {
    for (std::shared_ptr<const render::CubeLut>& lut : out.luts) lut.reset();
    log_channel("app")->warn("look : pile incomplète ({} fichier(s) manquant(s)), image sans look",
                             out.failed.size());
  }
  return out;
}

std::vector<std::string> show_looks(PlaceboScreen& screen, LoadedLooks& looks) {
  std::array<const render::CubeLut*, render::kLookSlots> luts{};
  for (std::size_t i = 0; i < render::kLookSlots; ++i) luts.at(i) = looks.luts.at(i).get();
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
};

LookLoader::~LookLoader() {
  // Le fil est joint, jamais détaché : détaché, il journalisait après close_log et appelait SDL après
  // SDL_Quit, d'où un plantage à la sortie (A30-02). Une lecture en cours est bornée (fichiers locaux de
  // 16 Mo au plus, cube_lut.h) ; une demande pas encore commencée ne l'est plus après l'arrêt.
  if (!thread_.joinable()) return;
  thread_.request_stop();
  thread_.join();
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
        set_thread_role(ThreadRole::Background);  // lecture des .cube, du fond (C20-06)
        std::unique_lock lock(state->mutex);
        while (state->wake.wait(lock, stop, [&] { return state->pending.has_value(); })) {
          if (stop.stop_requested()) break;  // la demande vue avec l'arrêt ne se lit plus
          auto [settings, dir] = std::move(*state->pending);
          state->pending.reset();
          const std::uint64_t asked = state->asked;
          lock.unlock();
          LoadedLooks looks = load_looks(settings, dir);
          lock.lock();
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
  if (s.grade && source_hdr)
    text += i18n::tr(" · grade off: HDR source");
  else if (s.grade)
    text += screen.has_grade() ? i18n::tr(" · grade") : i18n::tr(" · grade unavailable");  // A41-12
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
  s.grade_values.skin_tones = wanted.face_skin ? to.skin_tones : render::SkinTones{};
  to.series_apl_sent = to.anime_series_apl;
  render::GradeChoice choice = grade_of(s, to.auto_peak, to.display, to.black, to.source_hdr);
  // Flashs dangereux (P3.51) : le grade HDR plafonne vers 250 nits, le plein champ d'un écran OLED.
  // Le limiteur juge la lumière que l'écran montre (A42-08) : celle de la source fois le blanc du grade, pris
  // avant le plafond (sinon le plafond relâcherait le limiteur, qui le reposerait aussitôt). Une source déjà
  // en PQ ou HLG est mesurée en nits : aucun gain (B42-07).
  to.live.scenes().set_flash_gain(choice.pq && !to.source_hdr ? flash_gain_of(choice.pq->peak) : 1.0);
  if (to.flash_limited) cap_flash_peak(choice);
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

// Au plus une lumière de scène tous les dixièmes de seconde hors coupe : le grade ne repompe pas à chaque
// image.
constexpr std::int64_t kSceneLightEveryUs = 100'000;

// Une lumière de scène qui change vraiment : 3 % de luminance moyenne, 2 points de part claire.
bool light_moved(const render::SceneLight& now, const render::SceneLight& was) {
  return std::abs(now.apl_nits - was.apl_nits) > std::max(0.03 * was.apl_nits, 0.25) ||
         std::abs(now.lit - was.lit) > 0.02;
}

// La peau des visages du plan (face_skin) : elle change à la coupe, sans glissé (une teinte ne se mélange
// pas), en sortie HDR comme SDR. Vrai si elle a changé.
bool take_skins(Chain& to, const ShotScene& scene, bool planned) {
  const render::SkinTones skins = planned && to.running.face_skin ? scene.skins : render::SkinTones{};
  if (skins == to.skin_tones) return false;
  to.skin_tones = skins;
  return true;
}

// La lumière de la scène pour le grade en trois passes : envoyée quand elle change vraiment (light_moved), au
// plus tous les dixièmes de seconde, sauf à une coupe. Sous --anime-shots (méga plan A2), un plan de la
// pré-analyse fixe aussi l'exposition de la méthode « HDR anime » ; l'une et l'autre changent à la coupe et
// glissent hors coupe (anime_shots.h), avec le même seuil et la même cadence (B42-08 : un fondu
// reconstruisait le grade à chaque image). B38-14 : en sortie HDR, la lumière de la scène et l'exposition par
// plan sont effacées (render::steady_hdr) : rien à calculer ni à renvoyer, sauf la série des noirs profonds
// quand elle change.
void light_tick(Chain& to, const std::optional<ShotScene>& scene, std::int64_t now_us, bool cut,
                bool planned) {
  if (!scene || !to.running.grade) return;
  if (take_skins(to, *scene, planned)) set_grade(to, to.running, true);
  if (to.screen.hdr_output()) {
    to.scene_light.reset();  // revenue en SDR, la sortie reçoit aussitôt la lumière de sa scène
    to.anime_e = 0.0;
    if (std::abs(to.anime_series_apl - to.series_apl_sent) > 0.01 * std::max(to.series_apl_sent, 1.0))
      set_grade(to, to.running, true);
    return;
  }
  render::SceneLight light = scene->light;
  if (to.anime_shots && planned) {
    to.anime_glide.aim({.e = anime_exposure(light.apl_nits, to.anime_series_apl), .light = light}, now_us,
                       cut);
    const AnimeShotLook shown = to.anime_glide.at(now_us);
    // Le glissé fini : sa valeur exacte part, même sous le seuil.
    const bool settled = !to.anime_glide.gliding(now_us) &&
                         (shown.e != to.anime_e || !to.scene_light || shown.light != *to.scene_light);
    const bool moved = std::abs(shown.e - to.anime_e) > 0.005 || !to.scene_light ||
                       light_moved(shown.light, *to.scene_light) || settled;
    if (!moved || (!cut && to.scene_light && now_us - to.scene_light_at_us < kSceneLightEveryUs)) return;
    to.anime_e = shown.e;
    to.scene_light = shown.light;
    to.scene_light_at_us = now_us;
    set_grade(to, to.running, true);
    return;
  }
  to.anime_e = 0.0;  // sans plan analysé : le relèvement fixe
  if (to.scene_light) {
    const bool moved = light_moved(light, *to.scene_light);
    if (!moved || (!cut && now_us - to.scene_light_at_us < kSceneLightEveryUs)) return;
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
  // Le plan de la pré-analyse qui arrive sur une scène vue à l'ingestion n'est pas une coupe (A42-04) : le
  // grade glisse de la mesure en direct aux valeurs du plan, il ne saute pas au milieu du plan.
  const bool replaces_live = shot && shot->planned && (to.shot_serial & kLiveSerialBit) != 0;
  // B42-01 : un même plan réel coupé en deux au point de reprise d'une analyse n'est pas une coupe.
  const bool joined = shot && continues_shot(*shot, to.shot_end_us);
  const bool new_shot = shot && shot->serial != to.shot_serial && !replaces_live && !joined;
  to.shot_serial = shot ? shot->serial : 0;
  to.shot_end_us = shot && shot->planned ? shot->end_us : -1;
  // Le plafond ne joue que sur la sortie PQ (A42-19) : en SDR la sortie ne dépasse pas le blanc de l'écran,
  // et le journal annonçait un plafond qui ne changeait rien à l'image.
  if (const bool flashing =
          to.running.flash_guard && to.running.grade && to.screen.hdr_output() && next_flashing(to);
      flashing != to.flash_limited) {
    to.flash_limited = flashing;
    if (flashing)
      log_channel("app")->info("flashs rapides : pic du grade plafonné à {:.0f} nits", kFlashPeakNits);
    if (!flashing) log_channel("app")->info("flashs finis : pic du grade rendu");
    set_grade(to, to.running, true);
  }
  // Seul un plan de la pré-analyse a des valeurs fixes : une scène en direct, dont la moyenne bouge à chaque
  // image, garde le chemin ordinaire (pas de 3 %, un dixième de seconde), sans exposition par plan.
  light_tick(to, shot ? shot : next_live(to), now_us, new_shot, shot && shot->planned);
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
  // Le vrai noir s'applique sur les deux sorties du grade (A41-07) : en HDR10, et en SDR où ses seuils
  // sont mesurés sur l'image que la dernière passe ramène au SDR (render::sdr_tonemap les garde).
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
    to.ui.set_black_status(i18n::tr("True black: cannot measure (see the log)."));
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
