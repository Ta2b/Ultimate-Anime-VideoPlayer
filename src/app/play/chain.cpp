#include "app/play/chain.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>
#include <variant>

#include "app/play/chapters.h"
#include "app/play/episodes.h"
#include "app/play/i18n.h"
#include "audio/devices.h"
#include "core/clock.h"
#include "core/log.h"
#include "prepare/utf8.h"
#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kHdrRefreshDelayUs =
    120'000;  // après le dernier changement d'un réglage de RTX HDR, en pause

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

std::string volume_text(const Controls& controls) {
  if (controls.muted()) return i18n::tr("Muted");
  return i18n::tr_format("Volume {} %", std::lround(controls.volume() * 100.0F));
}

std::string label_of(const std::vector<TrackEntry>& entries, int track) {
  const auto found = std::ranges::find(entries, track, &TrackEntry::index);
  return found != entries.end() ? found->label : i18n::tr_format("track {}", track);
}

}  // namespace

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

bool chain_differs(const PlaySettings& a, const PlaySettings& b) {
  return a.upscale != b.upscale || a.model != b.model || a.work_height != b.work_height ||
         a.downscale != b.downscale || factor_of(a) != factor_of(b) || a.true_hdr != b.true_hdr ||
         a.hdr_native != b.hdr_native;
}

PlaySettings with_intended_hdr(const Chain& to, PlaySettings settings) {
  if (to.hdr_wanted) {
    settings.true_hdr = to.hdr_wanted->true_hdr;
    settings.hdr_native = to.hdr_wanted->hdr_native;
  }
  return settings;
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
    to.ui.controls().notify(i18n::tr("Audio track refused"), now_us);
    return;
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(i18n::tr_format("Audio: {}", label_of(to.pipeline.tracks().audio, track)), now_us);
}

// Sous-titres : subs prend la nouvelle piste, un seek à la position courante relit ses paquets.
void switch_subtitles(Chain& to, std::optional<int> track, std::int64_t now_us) {
  if (to.pipeline.tracks().subtitle_selected == track) return;
  auto switched = to.pipeline.select_subtitles(track);
  if (!switched) {
    log_channel("app")->warn("sous-titres : {}", switched.error().detail);
    to.ui.controls().notify(i18n::tr("Subtitle track refused"), now_us);
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
      track ? i18n::tr_format("Subtitles: {}", label_of(to.pipeline.tracks().subtitles, *track))
            : std::string(i18n::tr("Subtitles off")),
      now_us);
}

// Réglages en direct du menu des paramètres : RIFE, réduction, RTX HDR, son, statistiques.
void apply_live(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  PlaySettings& r = to.running;
  Controls& controls = to.ui.controls();
  i18n::set_language(i18n::language_from_code(wanted.language));
  if (factor_of(wanted) != factor_of(r)) {
    log_channel("app")->info("interpolation en direct : {}", factor_text(factor_of(wanted)));
    to.live.set_rife_factor(factor_of(wanted));
    controls.notify(wanted.rife_factor >= 2
                        ? i18n::tr_format("Interpolation {}", factor_text(factor_of(wanted)))
                        : std::string(i18n::tr("Interpolation off")),
                    now_us);
  }
  if (wanted.rife_auto != r.rife_auto) {
    if (wanted.rife_auto) controls.notify(i18n::tr("Interpolation: Auto"), now_us);
  }
  if (wanted.auto_free != r.auto_free) {
    log_channel("app")->info("mode d'Auto en direct : {}",
                             wanted.auto_free ? "rapide (expérimental)" : "stable");
    controls.notify(wanted.auto_free ? i18n::tr("Auto: Fast (experimental)") : i18n::tr("Auto: Stable"),
                    now_us);
  }
  if (to.header_text && (factor_of(wanted) != factor_of(r) || wanted.downscale != r.downscale ||
                         wanted.hdr_native != r.hdr_native)) {
    to.session.set_header({to.header_text(wanted)});
  }
  if (wanted.keep_poses != r.keep_poses) {
    to.live.set_keep_poses(wanted.keep_poses);
    controls.notify(wanted.keep_poses ? std::string(i18n::tr("Keep poses: on"))
                                      : std::string(i18n::tr("Keep poses: off")),
                    now_us);
  }
  if (wanted.downscale != r.downscale) {
    log_channel("app")->info("réduction en direct : {}", label(wanted.downscale));
    to.live.set_reduction(reduction_of(wanted.downscale));
    controls.notify(i18n::tr_format("Downscale: {}", label(wanted.downscale)), now_us);
  }
  const vendor_hdr::TrueHdrParams before = true_hdr_params(r, to.auto_peak);
  const vendor_hdr::TrueHdrParams after = true_hdr_params(wanted, to.auto_peak);
  if (to.true_hdr_open && after != before) {
    log_channel("app")->info(
        "HDR en direct : contraste {} · saturation {} · gris moyen {} · pic {} nits · débandage {}",
        after.contrast, after.saturation, after.middle_gray, after.peak_nits,
        after.debanding ? "oui" : "non");
    to.live.set_true_hdr(after);
    // RTX HDR ne traite que les images neuves : en pause, la position tenue est relue après le dernier
    // changement (le curseur qu'on fait glisser n'en lance qu'une).
    if (to.session.view().paused) to.hdr_refresh_us = now_us + kHdrRefreshDelayUs;
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
      controls.notify(wanted.grade ? i18n::tr_format("Grade: {}", grade_label(wanted.grade_preset))
                                   : std::string(i18n::tr("Grade off")),
                      now_us);
    }
  }
  if (wanted.final_scaler != r.final_scaler || wanted.deband != r.deband) {
    to.screen.set_finish(wanted.final_scaler, wanted.deband);
    controls.notify(i18n::tr_format("Scaler: {}, deband {}", label(wanted.final_scaler),
                                    wanted.deband ? i18n::tr("on") : i18n::tr("off")),
                    now_us);
  }
  if (wanted.audio_delay_ms != r.audio_delay_ms)
    to.session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{wanted.audio_delay_ms} * 1000});
  {
    // L'écoute (casque, enceintes, home cinéma, fidèle) se change en direct ; l'appareil, lui, rouvre la
    // vidéo.
    const auto device = audio::resolve_device(audio::playback_devices(), wanted.audio_device);
    const std::string name = device ? device->name : std::string();
    const EffectiveListening now = effective_listening(wanted, name);
    const EffectiveListening was = effective_listening(r, name);
    if (now != was) {
      cmd::SetListening listening{.listen = static_cast<std::uint8_t>(now.listen),
                                  .faithful = now.faithful,
                                  .tuned = true,
                                  .tuning = audio::to_wire(now.tuning),
                                  .anime_set = true,
                                  .anime = audio::to_wire(now.anime)};
      if (now.hrir_path != was.hrir_path) {
        // Le fichier se lit ici, hors du thread audio ; illisible : on le dit et le modèle paramétrique
        // reste.
        listening.hrir_set = true;
        if (auto loaded = load_listening_hrir(now.hrir_path)) {
          listening.hrir = std::move(*loaded);
        } else {
          log_channel("app")->warn("{}", loaded.error().detail);
          controls.notify(i18n::tr("Impulse response unreadable: using the default"), now_us);
        }
      }
      to.session.command(std::move(listening));
    }
  }
  if (wanted.display_sync != r.display_sync) to.session.set_display_sync(wanted.display_sync);
  if (wanted.video_latency_ms != r.video_latency_ms)
    to.session.set_video_latency(std::int64_t{wanted.video_latency_ms} * 1000);
  if (wanted.stats != r.stats) to.session.set_stats(wanted.stats);
  if (wanted.smooth_motion != r.smooth_motion) {
    controls.notify(wanted.smooth_motion ? std::string(i18n::tr("Smooth Motion: on at next launch"))
                                         : std::string(i18n::tr("Smooth Motion: off at next launch")),
                    now_us);
  }
  if (wanted.oled_care != r.oled_care || wanted.steady_cadence != r.steady_cadence ||
      wanted.oled_black != r.oled_black)
    apply_oled_care(to, wanted);
  if (wanted.subtitle_scale != r.subtitle_scale) {
    to.session.set_subtitle_scale(wanted.subtitle_scale);
    controls.notify(i18n::tr_format("Subtitle size: {} %", std::lround(wanted.subtitle_scale * 100.0F)),
                    now_us);
  }
  // Grade, finition et RTX HDR agissent sur l'image tenue : sans cette demande, en pause (ou entre deux
  // images neuves) rien ne la recomposait et le réglage n'apparaissait qu'au mouvement suivant.
  if (wanted.grade != r.grade || wanted.grade_preset != r.grade_preset ||
      wanted.grade_values != r.grade_values || wanted.final_scaler != r.final_scaler ||
      wanted.deband != r.deband || wanted.hdr_native != r.hdr_native || after != before ||
      wanted.oled_black != r.oled_black || wanted.oled_black_steps != r.oled_black_steps)
    to.session.request_present();
  r = wanted;
  r.volume = controls.volume();
  to.ui.set_running(r);
  to.dirty = true;
}

// Profil slot : sa chaîne pour cette source, en direct si elle le permet, sinon par réouverture.
void switch_profile(Chain& to, int slot, std::int64_t now_us) {
  const auto& profile = to.profiles.data.file.profiles[static_cast<std::size_t>(std::clamp(slot, 1, 9) - 1)];
  if (!profile) {
    to.ui.controls().notify(i18n::tr_format("Profile {}: empty", slot), now_us);
    return;
  }
  std::vector<std::string> notes;
  const PlaySettings wanted = apply_profile(*profile, to.profiles.source, to.running, notes);
  for (const std::string& line : notes) log_channel("app")->warn("profil {} : {}", slot, line);
  log_channel("app")->info("profil {} « {} »", slot, profile->name);
  to.profiles.active = slot;
  to.ui.set_profiles(view_of(to.profiles));
  to.ui.controls().notify(i18n::tr_format("Profile {} · {}", slot, profile->name), now_us);
  const Change change = change_between(to.running, wanted, to.rife_loaded);
  if (change.cost == ChangeCost::Reopen) {
    to.reopen = wanted;
  } else if (change.cost == ChangeCost::Live) {
    apply_live(to, wanted, now_us);
  }
}

// Raccourcis Ctrl+G, Ctrl+I, Ctrl+0 : la chaîne voulue, en direct si elle le permet.
void switch_chain(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  const Change change = change_between(to.running, wanted, to.rife_loaded);
  if (change.cost == ChangeCost::Reopen) {
    to.reopen = wanted;
  } else if (change.cost == ChangeCost::Live) {
    apply_live(to, wanted, now_us);
  }
}

// Onglet Moteurs : la chaîne se ferme pour changer de moteur ou rouvrir l'écran des moteurs, puis
// se rouvre avec les mêmes réglages au même endroit.
void ask_engines(Chain& to, EngineChange request) {
  log_channel("app")->info("moteurs : {}", std::holds_alternative<UseEngine>(request)
                                               ? "changement du moteur des réseaux"
                                               : "écran des moteurs rouvert");
  if (std::holds_alternative<OpenEngineScreen>(request)) to.ui.close_settings();
  to.engines = request;
  to.reopen = to.running;
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
            if (std::holds_alternative<cmd::ToggleGrade>(command)) {
              PlaySettings wanted = to.running;
              wanted.grade = !wanted.grade;
              switch_chain(to, wanted, now_us);
              return;
            }
            if (std::holds_alternative<cmd::ToggleInterpolation>(command)) {
              PlaySettings wanted = to.running;
              wanted.rife_factor = wanted.rife_factor >= 2 ? 0 : kDefaultRifeFactor;
              if (wanted.rife_factor == kDefaultRifeFactor) wanted.rife_factor_den = 1;
              switch_chain(to, wanted, now_us);
              return;
            }
            if (std::holds_alternative<cmd::ProcessingOff>(command)) {  // la source telle quelle
              PlaySettings wanted = to.running;
              wanted.grade = false;
              wanted.rife_factor = 0;
              switch_chain(to, wanted, now_us);
              return;
            }
            if (std::holds_alternative<cmd::StepFrame>(command)) controls.activity(now_us);
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
              controls.notify(
                  to.running.stats.all ? i18n::tr("Statistics shown") : i18n::tr("Statistics hidden"),
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
          [&](const ToggleFaithful&) {
            PlaySettings wanted = to.running;
            wanted.audio_faithful = !wanted.audio_faithful;
            apply_live(to, wanted, now_us);
            to.running.audio_faithful = wanted.audio_faithful;
            to.dirty = true;
            controls.notify(
                wanted.audio_faithful ? i18n::tr("Sound: original") : i18n::tr("Sound: processed"), now_us);
          },
          [&](const SelectAudio& select) { switch_audio(to, select.track, now_us); },
          [&](const SelectSubtitles& select) { switch_subtitles(to, select.track, now_us); },
          [&](const CycleAudio&) {
            if (auto next = next_audio(to.pipeline.tracks())) switch_audio(to, *next, now_us);
          },
          [&](const CycleSubtitles&) { switch_subtitles(to, next_subtitle(to.pipeline.tracks()), now_us); },
          [&](const StepEpisode& step) {
            const auto neighbour = sibling_episode(to.file, step.direction);
            if (!neighbour) {
              controls.notify(step.direction > 0 ? i18n::tr("No next episode in this folder")
                                                 : i18n::tr("No previous episode in this folder"),
                              now_us);
              return;
            }
            log_channel("app")->info("épisode {} : {}", step.direction > 0 ? "suivant" : "précédent",
                                     prepare::utf8_path(*neighbour));
            to.next_file = *neighbour;
            to.reopen = to.running;
          },
          [&](const SkipCredits&) {
            const ControlsView view = session.view();
            std::int64_t target = 0;
            if (const auto credits = credits_skip_target(view.chapters, view.position_us, view.duration_us)) {
              target = *credits;
              controls.notify(i18n::tr("Credits skipped"), now_us);
            } else {
              target = view.position_us + kSkipFallbackUs;
              controls.notify(i18n::tr("Skipped 85 s"), now_us);
            }
            if (view.duration_us > 0) target = std::min(target, view.duration_us);
            session.command(Command{cmd::Seek{.target_us = target}});
            controls.activity(now_us);
          },
          [&](const StepChapter& step) {
            const ControlsView view = session.view();
            const auto chapter = chapter_step(view.chapters, view.position_us, step.direction);
            if (!chapter) return;
            session.command(Command{cmd::Seek{.target_us = view.chapters[*chapter].start_us}});
            controls.activity(now_us);
            controls.notify(chapter_label(view.chapters, *chapter), now_us);
          },
          [&](const ToggleSettings&) { to.ui.toggle_settings(); },
          [&](const ApplySettings& apply) {
            // Changer la chaîne au menu modifie le profil actif. Un repli HDR en cours n'est pas
            // retenu : sauf si l'utilisateur a lui-même changé le mode HDR, le profil garde le mode
            // demandé (with_intended_hdr).
            if (apply.settings.true_hdr != to.running.true_hdr ||
                apply.settings.hdr_native != to.running.hdr_native)
              to.hdr_wanted.reset();
            const PlaySettings intended = with_intended_hdr(to, apply.settings);
            if (chain_differs(with_intended_hdr(to, to.running), intended)) {
              remember_chain(to.profiles, intended);
              to.ui.set_profiles(view_of(to.profiles));
            }
            if (apply.reopen) {
              log_channel("app")->info("paramètres : réouverture de la chaîne");
              to.reopen = intended;
            } else {
              apply_live(to, apply.settings, now_us);
            }
          },
          [&](const UseEngine& use) { ask_engines(to, use); },
          [&](const OpenEngineScreen& open) { ask_engines(to, open); },
          [&](const StartExport& start) {
            log_channel("app")->info("export demandé : la lecture se ferme pendant l'export");
            to.ui.close_settings();
            to.export_job = start.job;
            to.reopen = to.running;
          },
          [&](const EditProfile& edit) {
            const std::array<core::UserEdit, 1> edits{edit.edit};
            auto edited = edit_profiles(to.profiles.user_path, to.profiles.data, edits);
            if (!edited) {
              log_channel("app")->warn("user.toml : {}", edited.error().detail);
              to.ui.controls().notify(i18n::tr("Profile change refused"), now_us);
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
        << '\t' << record.missing << '\t' << record.lag_us << '\t' << record.uninterpolated << '\n';
}

void log_summary(const Session& session) {
  const FrameStats& stats = session.stats();
  log_channel("app")->info(
      "fin : {} images neuves, {} présentations, {} sautées, {} manquées, {} sans interpolation à "
      "dessein (coupes, dessins tenus), {} jetées au seek, {} sans sous-titres en attente ; "
      "temps par image médian {} (σ {}), p95 {} ; retard p95 {} ; composition médiane {}",
      stats.new_frames, stats.presents, stats.dropped, stats.missing, stats.uninterpolated, stats.stale,
      stats.subtitle_gaps, format_ms(stats.frame_interval.median()),
      format_ms(static_cast<std::int64_t>(stats.frame_interval.stddev())),
      format_ms(stats.frame_interval.percentile(95)), format_ms(stats.lag.percentile(95)),
      format_ms(stats.render.median()));
}

}  // namespace ajp::app::play
