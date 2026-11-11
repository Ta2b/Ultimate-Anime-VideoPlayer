#include "app/play/display_follow.h"

#include <cmath>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#include "app/play/auto_config.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "core/clock.h"

namespace ajp::app::play {
namespace {

// Le pic et le HDR du système se relisent sur événement (ContractWatch::take_peak_change) ; ce sondage n'est
// que le filet, pour un système qui ne dit rien (A11-02 : sous Windows, chaque relevé énumère les écrans).
constexpr std::int64_t kHdrPollUs = 10'000'000;
// Un relevé de l'écran plus long que cela est dit au journal (il ne retient plus la présentation, mais il
// dit ce que coûtent QueryDisplayConfig, DXGI ou NVAPI sur la machine).
constexpr std::int64_t kSlowReadingUs = 5'000;

bool fullscreen_of(SDL_Window* window) {
  return (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
}

}  // namespace

DisplayFollow::DisplayFollow(Player& p, Chain& chain, Session& session, DisplayState& state,
                             double source_fps, Hooks hooks, std::shared_ptr<spdlog::logger> log,
                             void (*wake)(void*), void* context)
    : p_(p),
      chain_(chain),
      session_(session),
      state_(state),
      hooks_(std::move(hooks)),
      log_(std::move(log)),
      known_display_(SDL_GetDisplayForWindow(p.screen.window())),
      session_hz_seen_(state.known_hz),
      opened_{display_height(p.screen.window()), state.known_hz, source_fps},
      contract_watch_(p.screen.window(), p.display, log_),  // 8.2 : relu sur HDR_STATE_CHANGED et KWin
      vrr_watch_(state.vrr_armed),
      hdr_polled_at_(host_now_us()),
      probe_(p.screen.window(), system_display_readers(), wake, context) {}

void DisplayFollow::poll() {
  PlaceboScreen& out = p_.screen;
  if (const auto contract = contract_watch_.tick()) out.set_contract(*contract);  // 8.2
  // Le HDR et le pic suivent le système et l'écran (Windows, KDE) : relus sur événement, sinon à 10 s.
  const bool changed = contract_watch_.take_peak_change();
  const bool peak = !p_.options.sdr && (changed || host_now_us() - hdr_polled_at_ >= kHdrPollUs);
  if (peak) hdr_polled_at_ = host_now_us();
  const bool vrr = p_.display && !p_.display->connector.empty();
  probe_.ask(display_sdl(out.window()), peak, vrr, vrr ? p_.display->connector : std::string());
  // L'écran change (éteint, autre sortie, nouveau mode) : sa fréquence (le dernier relevé), son VRR et son
  // état sont relus.
  const SDL_DisplayID display_id = SDL_GetDisplayForWindow(out.window());
  const bool display_changed = display_id != known_display_ || (display_id != 0 && state_.known_hz > 0.0 &&
                                                                state_.known_hz != session_hz_seen_);
  known_display_ = display_id;
  session_hz_seen_ = state_.known_hz;
  session_.set_display(state_.known_period_us, state_.known_hz);
  hooks_.apply_auto();
  follow_window(display_changed);
}

void DisplayFollow::take() {
  const std::optional<DisplayReading> reading = probe_.take();
  if (!reading) return;
  if (reading->took_us > kSlowReadingUs)
    log_->debug("relevé de l'écran : {:.1f} ms, hors du thread de présentation",
                static_cast<double>(reading->took_us) / 1000.0);
  if (reading->hz > 1.0) {
    state_.known_hz = reading->hz;
    state_.known_period_us = std::llround(1e6 / reading->hz);
  }
  if (reading->peak_read) follow_peak(reading->peak_nits);
  if (reading->vrr_read) follow_vrr(reading->vrr_armed);
}

void DisplayFollow::follow_peak(std::optional<double> peak_now) {
  if (system_hdr(peak_now) != chain_.running.hdr_native) {
    PlaySettings switched = chain_.running;
    switched.hdr_native = !switched.hdr_native;
    chain_.auto_peak = auto_peak_nits(peak_now);
    log_->info("HDR du système {} : sortie {}", switched.hdr_native ? "allumé" : "coupé",
               switched.hdr_native ? "HDR10" : "SDR gradée");
    apply(Input{ApplySettings{.settings = switched}}, chain_, host_now_us());
  } else {
    follow_display(chain_, peak_now);  // P3.58
  }
  // B08-04 : la page Sortie suit le HDR du système et le pic de l'écran ; décocher « pic automatique » ne
  // fige plus le pic de l'ouverture.
  p_.ui.set_display_output(p_.screen.hdr_output(), chain_.auto_peak);
}

void DisplayFollow::follow_vrr(std::optional<bool> reading) {
  // KDE arme et désarme le VRR selon la fenêtre active : relu à chaque relevé.
  if (!p_.display || p_.display->connector.empty()) return;
  const std::optional<bool> now = vrr_watch_.observe(p_.screen.vrr_armed(), reading);
  if (!now) return;
  const bool armed = !p_.options.no_vrr && p_.display->vrr.value_or(false) && *now;
  log_->info("VRR : {}", *now ? "armé" : "non armé");
  p_.ui.set_display_vrr(p_.display->vrr.value_or(false), *now, fullscreen_of(p_.screen.window()));
  if (armed != state_.vrr_armed) {
    state_.vrr_armed = armed;
    hooks_.apply_floor();
    hooks_.apply_auto();
  }
}

void DisplayFollow::follow_window(bool display_changed) {
  // KDE arme le VRR en plein écran seulement : on le relit quand la fenêtre change d'état.
  PlaceboScreen& out = p_.screen;
  const bool fullscreen = fullscreen_of(out.window());
  if (fullscreen == vrr_fullscreen_ && !display_changed) return;
  if (display_changed) log_->info("écran : changement détecté ({} Hz), relecture du VRR", state_.known_hz);
  if (display_changed && !p_.options.bench_out && !p_.options.file.empty() &&
      display_moves_chain(chain_.running.preset,
                          opened_.now(display_height(out.window()), state_.known_hz))) {
    chain_.reopen = chain_.running;  // A29-03 : la chaîne, pas le lecteur ; pistes et sous-titres gardés
    chain_.rederive = true;          // B29-01 : refaite pour le nouvel écran, pas recopiée
  }
  vrr_fullscreen_ = fullscreen;
  session_.set_composited(!fullscreen);
  const std::optional<DisplayFacts> display = window_display(out.window());
  if (!display) return;
  if (display_changed) {
    // Le nouvel écran : son contrat (P3.40, A11-01), puis son pic, ses soins OLED et son vrai noir tout de
    // suite, en SDR aussi (A08-07).
    contract_watch_.set_display(p_.display = display);
    follow_display(chain_, content_peak_nits(out.window()));
  }
  p_.ui.set_display_vrr(display->vrr.value_or(false), display->vrr_armed.value_or(false), fullscreen);
  const bool armed = !p_.options.no_vrr && display->vrr.value_or(false) && display->vrr_armed.value_or(false);
  if (armed != state_.vrr_armed) {
    state_.vrr_armed = armed;
    hooks_.apply_floor();
    hooks_.apply_auto();  // les demi-paliers suivent le VRR
  }
  log_->info("VRR : {} · {}", display->vrr.value_or(false) ? "capable" : "non capable",
             display->vrr_armed.value_or(false) ? "armé" : "non armé");
}

}  // namespace ajp::app::play
