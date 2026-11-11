#include "app/play/chain.h"

#include <algorithm>
#include <array>
#include <format>
#include <tuple>
#include <utility>
#include <variant>

#include "app/play/chapters.h"
#include "app/play/episodes.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "app/play/present_stats.h"
#include "app/play/visible_error.h"
#include "audio/devices.h"
#include "core/clock.h"
#include "core/log.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

i18n::Translated volume_text(const Controls& controls) {
  if (controls.muted()) return i18n::ui_tr("Muted");
  return i18n::ui_tr_format("Volume {} %", std::lround(controls.volume() * 100.0F));
}

// Un fichier dans le journal : son nom seul, le rapport public ne dit ni le dossier ni l'utilisateur
// (A39-03).
std::string file_for_log(const std::filesystem::path& file) {
  return file.empty() ? std::string("l'accueil") : prepare::utf8_path(file.filename());
}

std::string label_of(const std::vector<TrackEntry>& entries, int track) {
  const auto found = std::ranges::find(entries, track, &TrackEntry::index);
  return found != entries.end() ? found->label : i18n::tr_format("track {}", track);
}

}  // namespace

double fps_of(const media::Rational& rate) {
  return rate.den > 0 ? static_cast<double>(rate.num) / static_cast<double>(rate.den) : 0.0;
}

PlaySettings with_kept_choices(const KeptChoices& kept, PlaySettings settings) {
  if (kept.grade) settings.grade = *kept.grade;
  if (kept.rife) {
    settings.rife_factor = kept.rife->first;
    settings.rife_factor_den = kept.rife->second;
  }
  return settings;
}

PlaySettings with_kept_choices(const Chain& to, PlaySettings settings) {
  return with_kept_choices(to.kept, std::move(settings));
}

namespace {

// Ce qui recompose le grade : une seule liste pour la pose du grade et pour la recomposition en pause.
bool grade_changed(const PlaySettings& a, const PlaySettings& b) {
  return a.grade != b.grade || a.grade_values != b.grade_values || a.hdr_native != b.hdr_native ||
         a.true_hdr_peak != b.true_hdr_peak || a.oled_black != b.oled_black ||
         a.oled_black_steps != b.oled_black_steps || a.look_contrast != b.look_contrast ||
         a.look_highlight != b.look_highlight || a.surface_white != b.surface_white;
}

}  // namespace

std::pair<int, int> toggled_interpolation(int factor, int den,
                                          const std::optional<std::pair<int, int>>& kept) {
  if (factor >= 2) return {0, den};
  if (kept && kept->first >= 2) return *kept;
  return {kDefaultRifeFactor, 1};
}

namespace {

// Un raccourci change la chaîne pour la séance : play.toml garde ce qu'il avait avant.
void keep_for_file(Chain& to, bool grade, bool rife) {
  if (grade && !to.kept.grade) to.kept.grade = to.running.grade;
  if (rife && !to.kept.rife) to.kept.rife = std::pair{to.running.rife_factor, to.running.rife_factor_den};
}

// Piste audio : le pont change de piste à l'époque du seek qui suit, à la position courante.
void switch_audio(Chain& to, int track, std::int64_t now_us) {
  if (to.pipeline.tracks().audio_selected == track) return;
  const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
  auto switched = to.pipeline.select_audio(track, to.session.epoch() + 1, [&] {
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  });
  if (!switched) {
    log_channel("app")->warn("piste audio {} : {}", track, switched.error().detail);
    to.ui.controls().notify(i18n::ui_tr("Audio track refused"), now_us);
    return;
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(i18n::ui_tr_format("Audio: {}", label_of(to.pipeline.tracks().audio, track)),
                          now_us);
}

// Sous-titres : subs prend la nouvelle piste, un seek à la position courante relit ses paquets. Un fichier
// externe est lu en entier : pas de seek, l'image ne saute pas (P3.80).
void switch_subtitles(Chain& to, std::optional<int> track, std::int64_t now_us) {
  if (to.pipeline.tracks().subtitle_selected == track) return;
  auto switched = to.pipeline.select_subtitles(track);
  if (!switched) {
    log_channel("app")->warn("sous-titres : {}", switched.error().detail);
    to.ui.controls().notify(i18n::ui_tr("Subtitle track refused"), now_us);
    return;
  }
  to.session.set_subtitles(track.has_value());
  if (track && *track < kExternalSubtitleBase) {
    const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
  to.ui.controls().notify(
      track ? i18n::ui_tr_format("Subtitles: {}", label_of(to.pipeline.tracks().subtitles, *track))
            : i18n::ui_tr(("Subtitles off")),
      now_us);
}

// Réglages en direct du menu des paramètres : RIFE, réduction, HDR, son, statistiques.
void apply_live(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  PlaySettings& r = to.running;
  Controls& controls = to.ui.controls();
  i18n::set_language(i18n::language_from_code(wanted.language));
  if (factor_of(wanted) != factor_of(r)) {
    log_channel("app")->info("interpolation en direct : {}", factor_text(factor_of(wanted)));
    to.live.set_rife_factor(factor_of(wanted));
    controls.notify(wanted.rife_factor >= 2
                        ? i18n::ui_tr_format("Interpolation {}", factor_text(factor_of(wanted)))
                        : i18n::ui_tr(("Interpolation off")),
                    now_us);
  }
  if (wanted.rife_auto != r.rife_auto) {
    if (wanted.rife_auto) controls.notify(i18n::ui_tr("Interpolation: Auto"), now_us);
  }
  if (wanted.auto_free != r.auto_free) {
    log_channel("app")->info("mode d'Auto en direct : {}", wanted.auto_free ? "rapide" : "stable");
    controls.notify(wanted.auto_free ? i18n::ui_tr("Auto: Instant") : i18n::ui_tr("Auto: Stable"), now_us);
  }
  if (to.header_text && (factor_of(wanted) != factor_of(r) || wanted.downscale != r.downscale ||
                         wanted.hdr_native != r.hdr_native)) {
    to.session.set_header({to.header_text(wanted)});
  }
  if (wanted.keep_poses != r.keep_poses) {
    to.live.set_keep_poses(wanted.keep_poses);
    controls.notify(wanted.keep_poses ? i18n::ui_tr(("Keep poses: on")) : i18n::ui_tr(("Keep poses: off")),
                    now_us);
  }
  if (wanted.downscale != r.downscale) {
    log_channel("app")->info("réduction en direct : {}", label(wanted.downscale));
    to.live.set_reduction(reduction_of(wanted.downscale));
    controls.notify(i18n::ui_tr_format("Downscale: {}", label(wanted.downscale)), now_us);
  }
  if (!same_look_files(wanted, r)) {
    start_looks(to, wanted);  // lues en fond, posées au tour suivant (tend_looks)
  }
  const bool grade_moved = grade_changed(wanted, r);
  if (grade_moved) {
    // SDR ou HDR en direct : la swapchain change d'espace de couleur à la présentation suivante.
    if (wanted.hdr_native != r.hdr_native) to.screen.set_sdr_output(!wanted.hdr_native);
    set_grade(to, wanted);
    to.black_dirty = true;
    to.session.set_output(output_text(to.screen, wanted, to.source_hdr));
    if (wanted.grade != r.grade)
      controls.notify(wanted.grade ? i18n::ui_tr("Grade on") : i18n::ui_tr("Grade off"), now_us);
  }
  // B08-14 : la finition se pose une fois, même quand la netteté et le filtre changent ensemble.
  if (wanted.sharpness != r.sharpness || wanted.final_scaler != r.final_scaler || wanted.deband != r.deband)
    to.screen.set_finish(wanted.final_scaler, wanted.deband, wanted.sharpness);
  if (wanted.sharpness != r.sharpness)
    controls.notify(i18n::ui_tr_format("Sharpness: {} %", wanted.sharpness), now_us);
  if (wanted.final_scaler != r.final_scaler || wanted.deband != r.deband) {
    to.live.set_deband_source(wanted.deband);
    controls.notify(i18n::ui_tr_format("Scaler: {}, deband {}", label(wanted.final_scaler),
                                       wanted.deband ? i18n::tr("on") : i18n::tr("off")),
                    now_us);
  }
  if (wanted.audio_delay_ms != r.audio_delay_ms) {
    to.session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{wanted.audio_delay_ms} * 1000});
    controls.notify(i18n::ui_tr_format("Audio delay: {} ms", wanted.audio_delay_ms), now_us);
  }
  if (wanted.display_sync != r.display_sync) to.session.set_display_sync(wanted.display_sync);
  if (wanted.video_latency_ms != r.video_latency_ms)
    to.session.set_video_latency(std::int64_t{wanted.video_latency_ms} * 1000);
  if (wanted.stats != r.stats) to.session.set_stats(wanted.stats);
  if (wanted.smooth_motion != r.smooth_motion) {
    controls.notify(wanted.smooth_motion ? i18n::ui_tr(("Smooth Motion: on at next launch"))
                                         : i18n::ui_tr(("Smooth Motion: off at next launch")),
                    now_us);
  }
  if (wanted.oled_care != r.oled_care || wanted.steady_cadence != r.steady_cadence ||
      wanted.oled_black != r.oled_black)
    apply_oled_care(to, wanted);
  if (wanted.subtitle_delay_ms != r.subtitle_delay_ms) {
    to.session.set_subtitle_delay(std::int64_t{wanted.subtitle_delay_ms} * 1000);
    controls.notify(i18n::ui_tr_format("Subtitle delay: {} ms", wanted.subtitle_delay_ms), now_us);
  }
  if (wanted.subtitle_scale != r.subtitle_scale) {
    to.session.set_subtitle_scale(wanted.subtitle_scale);
    controls.notify(i18n::ui_tr_format("Subtitle size: {} %", std::lround(wanted.subtitle_scale * 100.0F)),
                    now_us);
  }
  // Grade et finition agissent sur l'image tenue : sans cette demande, en pause (ou entre deux
  // images neuves) rien ne la recomposait et le réglage n'apparaissait qu'au mouvement suivant.
  if (grade_moved || wanted.final_scaler != r.final_scaler || wanted.deband != r.deband ||
      wanted.sharpness != r.sharpness)
    to.session.request_present();
  r = wanted;
  r.volume = controls.volume();
  to.ui.set_running(r);
  to.dirty = true;
}

// Raccourcis Ctrl+G, Ctrl+I, Ctrl+0 : la chaîne voulue, en direct si elle le permet.
void switch_chain(Chain& to, const PlaySettings& wanted, std::int64_t now_us) {
  const Change change = change_between(to.running, wanted, to.opened.rife_loaded, &to.opened);
  if (change.cost == ChangeCost::Reopen) {
    to.reopen = wanted;
  } else if (change.cost == ChangeCost::Live) {
    apply_live(to, wanted, now_us);
  }
}

// « Appliquer » du panneau, ou des réglages directs qui ne tiennent pas dans la chaîne ouverte : le panneau
// ne livre en direct que ce qui y tient, ce garde-fou rouvre si un autre chemin en sortait (A08-03).
bool must_reopen(const Chain& to, const ApplySettings& apply) {
  return apply.reopen || !fits(to.opened, apply.settings);
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

void take_drop(Chain& to, const Dropped& drop, std::int64_t now_us) {
  const std::string name = prepare::utf8_path(drop.path.filename());
  if (drop.kind == DropKind::Video) {  // une autre vidéo : la chaîne se ferme pour elle
    log_channel("app")->info("fichier glissé : {}", name);
    to.next_file = drop.path;
    to.reopen = to.running;
    return;
  }
  if (drop.kind == DropKind::Subtitles) {
    auto added = to.pipeline.add_subtitle_file(drop.path);
    if (added) {
      if (to.dropped_subtitles != nullptr) to.dropped_subtitles->push_back(drop.path);
      switch_subtitles(to, *added, now_us);
      return;
    }
    // B08-11 : des sous-titres illisibles ne sont pas « ni une vidéo ni des sous-titres » ; la raison va au
    // journal.
    log_channel("app")->warn("sous-titres glissés refusés : {} ({})", name, added.error().detail);
    to.ui.controls().notify(
        i18n::ui_tr_format("Subtitles refused ({}): {}", visible_error(added.error()), name), now_us);
    return;
  }
  log_channel("app")->info("fichier glissé refusé : {}", name);
  to.ui.controls().notify(i18n::ui_tr_format("Not a video or subtitles: {}", name), now_us);
}

void carry_session(Chain& to, StageMemory* stages, std::vector<std::filesystem::path>& dropped,
                   std::optional<std::optional<int>> wanted) {
  to.stages = stages;
  to.dropped_subtitles = &dropped;
  std::erase_if(*to.dropped_subtitles, [&](const std::filesystem::path& file) {
    auto added = to.pipeline.add_subtitle_file(file);
    if (!added) {
      log_channel("app")->warn("sous-titres glissés, à la réouverture : {}", added.error().detail);
      return true;
    }
    if (wanted && *wanted == *added && to.pipeline.select_subtitles(*added)) to.session.set_subtitles(true);
    return false;
  });
  to.ui.set_tracks(to.pipeline.tracks());
  to.session.set_tracks(to.pipeline.describe());
}

namespace {

void start_calibration(Chain& to) {
  log_channel("app")->info("banc demandé : la lecture se ferme pendant la calibration");
  to.ui.close_settings();
  to.calibrate = true;
  to.reopen = to.running;
}

// « Réessayer » (décision 10) : les étages refusés sont oubliés, la chaîne se rouvre telle qu'elle est
// demandée.
void retry_stages(Chain& to) {
  log_channel("app")->info("réessayer : les étages refusés sont oubliés, la chaîne se rouvre");
  if (to.stages != nullptr) to.stages->forget();
  to.ui.close_settings();
  to.reopen = to.running;
}

}  // namespace

// Une action du clavier ou de l'interface. La barre paraît aussi après un seek au clavier.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
            if (std::holds_alternative<cmd::ToggleGrade>(command)) {
              PlaySettings wanted = to.running;
              wanted.grade = !wanted.grade;
              keep_for_file(to, true, false);
              switch_chain(to, wanted, now_us);
              return;
            }
            if (std::holds_alternative<cmd::ToggleInterpolation>(command)) {
              PlaySettings wanted = to.running;
              keep_for_file(to, false, true);
              std::tie(wanted.rife_factor, wanted.rife_factor_den) =
                  toggled_interpolation(wanted.rife_factor, wanted.rife_factor_den, to.kept.rife);
              switch_chain(to, wanted, now_us);
              return;
            }
            // Ni grade ni interpolation pour la séance. B08-05 : SPAN, le débandage et la netteté restent
            // (les couper rouvrirait la chaîne) ; l'aide et le README le disent.
            if (std::holds_alternative<cmd::ProcessingOff>(command)) {
              PlaySettings wanted = to.running;
              wanted.grade = false;
              wanted.rife_factor = 0;
              keep_for_file(to, true, true);
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
                  to.running.stats.all ? i18n::ui_tr("Statistics shown") : i18n::ui_tr("Statistics hidden"),
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
          [&](const SubtitleDelayStep& step) {
            PlaySettings wanted = to.running;
            if (step.audio) {
              wanted.audio_delay_ms = std::clamp(wanted.audio_delay_ms + step.delta_ms,
                                                 limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
            } else {
              wanted.subtitle_delay_ms = std::clamp(wanted.subtitle_delay_ms + step.delta_ms,
                                                    limits::kSubtitleDelayMs.lo, limits::kSubtitleDelayMs.hi);
            }
            apply_live(to, wanted, now_us);
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
          [&](const StepEpisode& step) {
            const auto neighbour = sibling_episode(to.file, step.direction);
            if (!neighbour) {
              controls.notify(step.direction > 0 ? i18n::ui_tr("No next episode in this folder")
                                                 : i18n::ui_tr("No previous episode in this folder"),
                              now_us);
              return;
            }
            log_channel("app")->info("épisode {} : {}", step.direction > 0 ? "suivant" : "précédent",
                                     file_for_log(*neighbour));
            to.next_file = *neighbour;
            to.reopen = to.running;
          },
          [&](const PlayFile& play) {
            log_channel("app")->info("lecture de {}", file_for_log(play.file));
            to.next_file = play.file;
            to.next_start_us = play.start_us;
            to.next_paused = play.paused;
            to.reopen = to.running;
          },
          [&](const SkipCredits&) {
            const ControlsView view = session.view();
            // P3.82 : seulement dans un chapitre de générique ; plus de saut de 85 s à l'aveugle.
            const auto credits = credits_skip_target(view.chapters, view.position_us, view.duration_us);
            if (!credits) {
              controls.notify(i18n::ui_tr("No credits chapter here"), now_us);
              return;
            }
            std::int64_t target = *credits;
            controls.notify(i18n::ui_tr("Credits skipped"), now_us);
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
            // Le menu qui change un de ces réglages le choisit pour de bon ; sinon le raccourci reste de
            // séance.
            if (apply.settings.grade != to.running.grade) to.kept.grade.reset();
            if (apply.settings.rife_factor != to.running.rife_factor ||
                apply.settings.rife_factor_den != to.running.rife_factor_den)
              to.kept.rife.reset();
            if (must_reopen(to, apply)) {
              // Le raccourci de séance tient, play.toml n'en garde rien (Player::kept, A08-04).
              log_channel("app")->info("paramètres : réouverture de la chaîne");
              to.reopen = apply.settings;
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
            // P3.90 : les pistes et le délai audio de la lecture passent dans le fichier.
            const TrackMenu& tracks = to.pipeline.tracks();
            to.export_job->copies = {.audio = tracks.audio_selected.value_or(-1),
                                     .subtitle = tracks.subtitle_selected.value_or(-1) < kExternalSubtitleBase
                                                     ? tracks.subtitle_selected.value_or(-1)
                                                     : -1,
                                     .subtitles_off = !tracks.subtitle_selected,
                                     .audio_delay_us = std::int64_t{to.running.audio_delay_ms} * 1000};
            to.reopen = to.running;
          },
          [&](const StartCalibration&) { start_calibration(to); },
          [&](const RetryStages&) { retry_stages(to); }},
      input);
}

void write_record(std::ofstream& stats, const PresentRecord& record) {
  stats << stats_line(record);
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
  if (const std::string range = take_gpu_range_text(); !range.empty()) log_channel("app")->info("{}", range);
}

}  // namespace ajp::app::play
