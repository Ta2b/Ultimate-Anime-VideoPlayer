#include "app/play/chain.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>
#include <variant>

#include "core/clock.h"
#include "core/log.h"
#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

namespace {

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

}  // namespace

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

int open_optical_flow(gpu::KernelOps& kernels, int width, int height, Controls& controls) {
  auto grid = kernels.open_optical_flow(width, height);
  if (grid) return *grid;
  log_channel("app")->warn("flux optique matériel indisponible ({}) : RIFE à la place", grid.error().detail);
  controls.notify("Flux optique indisponible : RIFE", host_now_us());
  return 0;
}

namespace {

std::string volume_text(const Controls& controls) {
  if (controls.muted()) return "Son coupé";
  return std::format("Volume {} %", std::lround(controls.volume() * 100.0F));
}

std::string label_of(const std::vector<TrackEntry>& entries, int track) {
  const auto found = std::ranges::find(entries, track, &TrackEntry::index);
  return found != entries.end() ? found->label : std::format("piste {}", track);
}

// Le grade HDR de la chaîne : route TrueHDR si RTX HDR est ouvert, native si le HDR natif est choisi
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

std::optional<render::GradeParams> grade_of(const PlaySettings& s, bool true_hdr_open, int peak_nits,
                                            int auto_peak, const std::optional<DisplayFacts>& display,
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
  auto decision = render::decide_grade(g, render::GradeContext{.route = route,
                                                               .source_hdr = false,
                                                               .output_pq = true,
                                                               .enabled = s.grade,
                                                               .peak_nits = static_cast<double>(peak)});
  if (!decision) {
    log_channel("app")->warn("grade : {}", decision.error().detail);
    return std::nullopt;
  }
  if (decision->mode == render::GradeMode::Neutral) return std::nullopt;
  return decision->params;
}

std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits) {
  std::string text = screen.output_description();
  if (peak_nits > 0) {
    text += std::format(" · RTX HDR, pic {} nits", peak_nits);
  } else if (s.hdr_native && screen.hdr_output()) {
    text += " · HDR natif";
  }
  if (s.grade && (peak_nits > 0 || s.hdr_native) && screen.hdr_output())
    text += std::format(" · grade {}", render::preset_label(s.grade_preset));
  return text;
}

double fps_of(const media::Rational& rate) {
  return rate.den > 0 ? static_cast<double>(rate.num) / static_cast<double>(rate.den) : 0.0;
}

ProfilesView view_of(const ProfileState& state) {
  ProfilesView view{.active = state.active, .start = state.data.file.start_profile};
  for (std::size_t i = 0; i < view.rows.size(); ++i) {
    if (const auto& profile = state.data.file.profiles[i])
      view.rows[i] = ProfileRow{.name = profile->name, .summary = profile_summary(*profile, state.source)};
  }
  return view;
}

// La chaîne en cours devient celle du profil actif, dans user.toml (rien n'y est écrit si elle
// redevient celle de la base).
void remember_chain(ProfileState& state, const PlaySettings& s) {
  const std::array<core::UserEdit, 2> edits{
      core::edit::ProfileRules{.slot = state.active, .rules = rules_of(s, state.default_model)},
      core::edit::ProfileHdr{.slot = state.active, .mode = hdr_of(s)}};
  auto edited = edit_profiles(state.user_path, state.data, edits);
  if (!edited) {
    log_channel("app")->warn("user.toml : {}", edited.error().detail);
    return;
  }
  state.data = std::move(*edited);
}

namespace {

bool chain_differs(const PlaySettings& a, const PlaySettings& b) {
  return a.upscale != b.upscale || a.model != b.model || a.work_height != b.work_height ||
         a.downscale != b.downscale || a.rife_factor != b.rife_factor || a.true_hdr != b.true_hdr;
}

}  // namespace

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
  const bool routed = to.true_hdr_open || s.hdr_native;
  const bool wanted = black_wanted(s, to.display) && s.grade && routed && to.screen.hdr_output();
  to.live.set_black_wanted(wanted && to.true_hdr_open);
  to.black.reset();
  if (!wanted) {
    set_grade(to, s);
    to.ui.set_black_status(black_wanted(s, to.display)
                               ? "Vrai noir : sans effet ici (grade coupé ou sortie SDR)."
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

namespace {

// Piste audio : le pont change de piste à l'époque du seek qui suit, à la position courante.
void switch_audio(Chain& to, int track, std::int64_t now_us) {
  if (to.pipeline.tracks().audio_selected == track) return;
  const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
  auto switched = to.pipeline.select_audio(track, to.session.epoch() + 1, [&] {
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  });
  if (!switched) {
    log_channel("app")->warn("piste audio {} : {}", track, switched.error().detail);
    to.ui.controls().notify("Piste audio refusée", now_us);
    return;
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(std::format("Audio : {}", label_of(to.pipeline.tracks().audio, track)), now_us);
}

// Sous-titres : subs prend la nouvelle piste, un seek à la position courante relit ses paquets.
void switch_subtitles(Chain& to, std::optional<int> track, std::int64_t now_us) {
  if (to.pipeline.tracks().subtitle_selected == track) return;
  auto switched = to.pipeline.select_subtitles(track);
  if (!switched) {
    log_channel("app")->warn("sous-titres : {}", switched.error().detail);
    to.ui.controls().notify("Piste de sous-titres refusée", now_us);
    return;
  }
  to.session.set_subtitles(track.has_value());
  if (track) {
    const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(
      track ? std::format("Sous-titres : {}", label_of(to.pipeline.tracks().subtitles, *track))
            : std::string("Sous-titres désactivés"),
      now_us);
}

// Réglages en direct du menu des paramètres : RIFE, réduction, RTX HDR, son, statistiques.
void apply_live(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  PlaySettings& r = to.running;
  Controls& controls = to.ui.controls();
  if (wanted.rife_factor != r.rife_factor) {
    log_channel("app")->info("RIFE en direct : ×{}", wanted.rife_factor);
    to.live.set_rife_factor(static_cast<std::uint32_t>(std::max(wanted.rife_factor, 0)));
    controls.notify(wanted.rife_factor >= 2 ? std::format("Interpolation ×{}", wanted.rife_factor)
                                            : std::string("Interpolation désactivée"),
                    now_us);
  }
  if (wanted.downscale != r.downscale) {
    log_channel("app")->info("réduction en direct : {}", label(wanted.downscale));
    to.live.set_reduction(reduction_of(wanted.downscale));
    controls.notify(std::format("Réduction : {}", label(wanted.downscale)), now_us);
  }
  const vendor_hdr::TrueHdrParams before = true_hdr_params(r, to.auto_peak);
  const vendor_hdr::TrueHdrParams after = true_hdr_params(wanted, to.auto_peak);
  if (to.true_hdr_open && after != before) {
    log_channel("app")->info(
        "RTX HDR en direct : contraste {} · saturation {} · gris moyen {} · pic {} nits · débandage {}",
        after.contrast, after.saturation, after.middle_gray, after.peak_nits,
        after.debanding ? "oui" : "non");
    to.live.set_true_hdr(after);
    if (after.peak_nits != before.peak_nits) {
      to.screen.set_source_color(pq_color(after.peak_nits));
      to.peak_nits = after.peak_nits;
    }
  }
  if (wanted.grade != r.grade || wanted.grade_preset != r.grade_preset ||
      wanted.grade_values != r.grade_values || wanted.hdr_native != r.hdr_native || after != before ||
      wanted.oled_black != r.oled_black || wanted.oled_black_steps != r.oled_black_steps) {
    set_grade(to, wanted);
    to.black_dirty = true;
    to.session.set_output(output_text(to.screen, wanted, to.peak_nits));
    if (wanted.grade_preset != r.grade_preset || wanted.grade != r.grade) {
      controls.notify(wanted.grade ? std::format("Grade : {}", render::preset_label(wanted.grade_preset))
                                   : std::string("Grade désactivé"),
                      now_us);
    }
  }
  if (wanted.audio_delay_ms != r.audio_delay_ms)
    to.session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{wanted.audio_delay_ms} * 1000});
  if (wanted.stats != r.stats) to.session.set_stats(wanted.stats);
  if (wanted.smooth_motion != r.smooth_motion) {
    controls.notify(wanted.smooth_motion ? std::string("Smooth Motion : au prochain lancement")
                                         : std::string("Smooth Motion coupé au prochain lancement"),
                    now_us);
  }
  if (wanted.oled_care != r.oled_care || wanted.steady_cadence != r.steady_cadence ||
      wanted.oled_black != r.oled_black)
    apply_oled_care(to, wanted);
  if (wanted.subtitle_scale != r.subtitle_scale) {
    to.session.set_subtitle_scale(wanted.subtitle_scale);
    controls.notify(std::format("Sous-titres : {} %", std::lround(wanted.subtitle_scale * 100.0F)), now_us);
  }
  r = wanted;
  r.volume = controls.volume();
  to.ui.set_running(r);
  to.dirty = true;
}

// Profil slot : sa chaîne pour cette source, en direct si elle le permet, sinon par réouverture.
void switch_profile(Chain& to, int slot, std::int64_t now_us) {
  const auto& profile = to.profiles.data.file.profiles[static_cast<std::size_t>(std::clamp(slot, 1, 9) - 1)];
  if (!profile) {
    to.ui.controls().notify(std::format("Profil {} : vide", slot), now_us);
    return;
  }
  std::vector<std::string> notes;
  const PlaySettings wanted = apply_profile(*profile, to.profiles.source, to.running, notes);
  for (const std::string& line : notes) log_channel("app")->warn("profil {} : {}", slot, line);
  log_channel("app")->info("profil {} « {} »", slot, profile->name);
  to.profiles.active = slot;
  to.ui.set_profiles(view_of(to.profiles));
  to.ui.controls().notify(std::format("Profil {} · {}", slot, profile->name), now_us);
  const Change change = change_between(to.running, wanted, to.rife_loaded);
  if (change.cost == ChangeCost::Reopen) {
    to.reopen = wanted;
  } else if (change.cost == ChangeCost::Live) {
    apply_live(to, wanted, now_us);
  }
}

}  // namespace

// Une action du clavier ou de l'interface. La barre paraît aussi après un seek au clavier.
void apply(const Input& input, Chain& to, std::int64_t now_us) {
  Session& session = to.session;
  PlaceboScreen& screen = to.screen;
  Controls& controls = to.ui.controls();
  const auto volume_changed = [&] {
    to.running.volume = controls.volume();
    to.ui.set_running(to.running);
    to.dirty = true;
    controls.notify(volume_text(controls), now_us);
  };
  std::visit(
      Overloaded{
          [&](const Command& command) {
            if (const auto* select = std::get_if<cmd::SelectProfile>(&command)) {  // Ctrl+1 à 9, menu
              switch_profile(to, select->slot, now_us);
              return;
            }
            if (std::holds_alternative<cmd::TogglePause>(command)) {
              log_channel("app")->info("action : {}", session.paused() ? "lecture" : "pause");
            } else if (const auto* seek = std::get_if<cmd::Seek>(&command)) {
              log_channel("app")->info("action : seek {}{}", format_position(seek->target_us),
                                       seek->exact ? "" : " (point clé)");
            }
            session.command(command);
            if (std::holds_alternative<cmd::SeekRelative>(command)) controls.activity(now_us);
            if (std::holds_alternative<cmd::SetVolume>(command)) {  // curseur de la barre
              to.running.volume = controls.volume();
              to.ui.set_running(to.running);
              to.dirty = true;
            }
            if (std::holds_alternative<cmd::ToggleOsd>(command)) {
              to.running.stats.all = session.stats_settings().all;
              to.ui.set_running(to.running);
              to.dirty = true;
              controls.notify(to.running.stats.all ? "Statistiques affichées" : "Statistiques masquées",
                              now_us);
            }
          },
          [&](const ToggleFullscreen&) { session.toggle_fullscreen(); },
          [&](const LeaveFullscreen&) {
            if (to.ui.settings_open()) {
              to.ui.close_settings();
            } else if (screen.fullscreen()) {
              session.toggle_fullscreen();
            }
          },
          [&](const ToggleMute&) {
            session.command(Command{controls.toggle_mute()});
            volume_changed();
          },
          [&](const VolumeStep& step) {
            session.command(Command{controls.step_volume(step.delta)});
            volume_changed();
          },
          [&](const SelectAudio& select) { switch_audio(to, select.track, now_us); },
          [&](const SelectSubtitles& select) { switch_subtitles(to, select.track, now_us); },
          [&](const CycleAudio&) {
            if (auto next = next_audio(to.pipeline.tracks())) switch_audio(to, *next, now_us);
          },
          [&](const CycleSubtitles&) { switch_subtitles(to, next_subtitle(to.pipeline.tracks()), now_us); },
          [&](const ToggleSettings&) { to.ui.toggle_settings(); },
          [&](const ApplySettings& apply) {
            // Changer la chaîne au menu modifie le profil actif.
            if (chain_differs(to.running, apply.settings)) {
              remember_chain(to.profiles, apply.settings);
              to.ui.set_profiles(view_of(to.profiles));
            }
            if (apply.reopen) {
              log_channel("app")->info("paramètres : réouverture de la chaîne");
              to.reopen = apply.settings;
            } else {
              apply_live(to, apply.settings, now_us);
            }
          },
          [&](const EditProfile& edit) {
            const std::array<core::UserEdit, 1> edits{edit.edit};
            auto edited = edit_profiles(to.profiles.user_path, to.profiles.data, edits);
            if (!edited) {
              log_channel("app")->warn("user.toml : {}", edited.error().detail);
              to.ui.controls().notify("Profil : modification refusée", now_us);
              return;
            }
            to.profiles.data = std::move(*edited);
            to.ui.set_profiles(view_of(to.profiles));
            // Revenir à la base change la chaîne du profil actif : elle s'applique.
            if (std::holds_alternative<core::edit::ResetProfile>(edit.edit))
              switch_profile(to, to.profiles.active, now_us);
          }},
      input);
}

void write_record(std::ofstream& stats, const PresentRecord& record) {
  stats << record.host_us << '\t' << record.pts_us << '\t' << (record.fresh ? 1 : 0) << '\t'
        << record.render_us << '\t' << record.interval_us << '\t' << record.dropped << '\t' << record.why
        << '\n';
}

void log_summary(const Session& session) {
  const FrameStats& stats = session.stats();
  log_channel("app")->info(
      "fin : {} images neuves, {} présentations, {} sautées, {} jetées au seek, {} sans sous-titres en "
      "attente ; "
      "temps par image médian {} (σ {}), p95 {} ; composition médiane {}",
      stats.new_frames, stats.presents, stats.dropped, stats.stale, stats.subtitle_gaps,
      format_ms(stats.frame_interval.median()),
      format_ms(static_cast<std::int64_t>(stats.frame_interval.stddev())),
      format_ms(stats.frame_interval.percentile(95)), format_ms(stats.render.median()));
}

}  // namespace ajp::app::play
