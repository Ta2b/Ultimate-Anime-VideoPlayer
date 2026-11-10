#include "app/play/settings_system.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <expected>
#include <format>

#include <SDL3/SDL_misc.h>
#include <SDL3/SDL_video.h>
#include <imgui.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_bench_run.h"
#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "app/play/paths.h"
#include "app/play/problem_report.h"
#include "app/play/span_auto.h"
#include "app/play/updater.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

void default_player_page(const PlayerPlace& place, SystemNote& last, bool already_default) {
  section(i18n::tr("Default player"));
  panel::wrapped(i18n::tr("Videos (mkv, mp4, webm, avi, mov…) open in AJP from the file manager."));
  if (already_default) note(i18n::tr("AJP is already the default video player."));
#ifdef __APPLE__
  static_cast<void>(last);
  note(register_player(place).value_or(std::string()));
#else
  if (last.pending && last.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = last.pending->get();
    if (!done) log_channel("app")->warn("inscription : {} {}", name(done.error().code), done.error().detail);
    last =
        done ? SystemNote{.text = *done} : SystemNote{.text = visible_error(done.error()), .refused = true};
  }
  const bool busy = last.pending != nullptr;
  const auto start = [&](std::expected<std::string, Refusal> (*action)(const PlayerPlace&)) {
    last.pending = start_task<std::string>("registration", [action, place] { return action(place); });
    last.text.clear();
    last.refused = false;
  };
  ImGui::BeginDisabled(busy);
  if (ImGui::Button(i18n::tr("Make this the default player"))) start(&choose_as_default);
  panel::same_line_if_fits(i18n::tr("Add to \"Open with\""));
  if (ImGui::Button(i18n::tr("Add to \"Open with\""))) start(&register_player);
  panel::same_line_if_fits(i18n::tr("Remove from \"Open with\""));
  if (ImGui::Button(i18n::tr("Remove from \"Open with\""))) start(&unregister_player);
  ImGui::EndDisabled();
  if (busy) {
    note(i18n::tr("Working…"));
  } else if (last.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(last.text.c_str());
    ImGui::PopStyleColor();
  } else if (!last.text.empty()) {
    note(last.text);
  }
#endif
}

void language_section(PlaySettings& d) {
  section(i18n::tr("Language"));
  const char* shown = d.language == "fr" ? "Français" : "English";  // i18n-ok : noms des langues
  if (ImGui::BeginCombo(i18n::tr("Interface language"), shown)) {
    if (ImGui::Selectable("English", d.language == "en")) d.language = "en";   // i18n-ok
    if (ImGui::Selectable("Français", d.language == "fr")) d.language = "fr";  // i18n-ok
    ImGui::EndCombo();
  }
  i18n::set_language(i18n::language_from_code(d.language));
}

void gpu_section(PlaySettings& d, const std::vector<std::string>& gpus) {
  if (gpus.size() < 2) return;
  section(i18n::tr("Graphics card"));
  const char* shown = d.gpu.empty() ? i18n::tr("Automatic (dedicated card first)") : d.gpu.c_str();
  if (ImGui::BeginCombo(i18n::tr("Card for playback"), shown)) {
    if (ImGui::Selectable(i18n::tr("Automatic (dedicated card first)"), d.gpu.empty())) d.gpu.clear();
    for (const std::string& name : gpus)
      if (ImGui::Selectable(name.c_str(), d.gpu == name)) d.gpu = name;
    ImGui::EndCombo();
  }
  note(i18n::tr("Applies at next launch."));
}

namespace {

// A23-04 : update.toml relu à l'ouverture de la page, après une vérification, puis toutes les deux secondes
// (le fil des mises à jour peut l'écrire), et non plus à chaque image.
const UpdateState& page_state(UpdateNote& note, const std::filesystem::path& file) {
  const auto now = std::chrono::steady_clock::now();
  if (!note.state || now - note.read_at > std::chrono::seconds(2)) {
    note.state = load_update_state(file);
    note.read_at = now;
  }
  return *note.state;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void updates_page(PlaySettings& d, const PlayerPlace& place, UpdateNote& result) {
  const std::filesystem::path state_file = paths::update_state(place.app_data);
  if (result.pending && result.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = result.pending->get();
    if (!done) log_channel("app")->warn("maj : {} {}", name(done.error().code), done.error().detail);
    result.pending.reset();
    result.text = done ? *done : visible_error(done.error());
    result.refused = !done;
    result.state.reset();  // la vérification vient de réécrire update.toml
  }
  const UpdateState& state = page_state(result, state_file);

  // Une simple notification (updater.h) : une version existe, et où la prendre ; rien ne s'installe.
  section(i18n::tr("Updates"));
  struct Mode {
    UpdateMode key;
    const char* label;
    const char* detail;
  };
  constexpr std::array<Mode, 2> kModes{
      {{UpdateMode::Notify, "Notify me",
        "A notification with the link to the new version; you install it yourself."},
       {UpdateMode::Never, "Never", "No checks."}}};
  for (const Mode& mode : kModes) {
    if (ImGui::RadioButton(i18n::tr(mode.label), d.update_mode == mode.key)) d.update_mode = mode.key;
    panel::tip(i18n::tr(mode.detail));
  }
  if (std::string_view(update_public_key()).empty())
    note(i18n::tr("This package has no signing key: updates cannot be verified."));

  const char* frequency = d.update_frequency == UpdateFrequency::Launch   ? i18n::tr("At every launch")
                          : d.update_frequency == UpdateFrequency::Weekly ? i18n::tr("Weekly")
                                                                          : i18n::tr("Daily");
  if (ImGui::BeginCombo(i18n::tr("Frequency"), frequency)) {
    if (ImGui::Selectable(i18n::tr("At every launch"), d.update_frequency == UpdateFrequency::Launch))
      d.update_frequency = UpdateFrequency::Launch;
    if (ImGui::Selectable(i18n::tr("Daily"), d.update_frequency == UpdateFrequency::Daily))
      d.update_frequency = UpdateFrequency::Daily;
    if (ImGui::Selectable(i18n::tr("Weekly"), d.update_frequency == UpdateFrequency::Weekly))
      d.update_frequency = UpdateFrequency::Weekly;
    ImGui::EndCombo();
  }
  note(i18n::tr_format("Installed version: {}", installed_version()));
  if (!state.latest_version.empty()) note(i18n::tr_format("Available: {}", state.latest_version));
  if (state.last_check_unix > 0) {
    const std::time_t at = static_cast<std::time_t>(state.last_check_unix);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &at);
#else
    localtime_r(&at, &local);
#endif
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), "%Y-%m-%d %H:%M", &local);
    note(i18n::tr_format("Last check: {}", text.data()));
  }

  const bool busy = result.pending != nullptr;
  ImGui::BeginDisabled(busy);
  if (ImGui::Button(i18n::tr("Check now"))) {
    result.pending = start_task<std::string>(
        "settings", [state_file, stop = result.stop] -> std::expected<std::string, Refusal> {
          return check_updates_now(state_file, stop->get_token());
        });
    result.text.clear();
    result.refused = false;
  }
  ImGui::EndDisabled();
  // La page de la release (ses notes, du manifeste signé, en https), sinon celle des releases : l'utilisateur
  // y prend l'archive ou l'installeur.
  if (!state.latest_version.empty()) {
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("Download page"))) {
      const std::string url =
          state.notes_url.starts_with("https://") ? state.notes_url : releases_page(release_repo());
      SDL_OpenURL(url.c_str());
    }
  }
  if (busy) {
    note(i18n::tr("Working…"));
  } else if (result.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(result.text.c_str());
    ImGui::PopStyleColor();
  } else if (!result.text.empty()) {
    note(result.text);
  }
  if (!state.last_error.empty() && result.text.empty())
    note(i18n::tr_format("The last check failed: {}", state.last_error));
}

bool start_span_prepare(const SpanPrepare& prepare, PrepareNote& state) {
  if (!prepare.offered || !prepare.run || state.pending) return false;
  auto progress = std::make_shared<SpanDeriveProgress>();
  auto stop = std::make_shared<std::stop_source>();
  state.progress = progress;
  state.stop = stop;
  state.pending = start_task<std::string>(
      "settings", [run = prepare.run, progress, stop] -> std::expected<std::string, Refusal> {
        try {
          return run(*progress, stop->get_token());
        } catch (const std::exception& error) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, error.what()});
        }
      });
  state.text.clear();
  state.refused = false;
  state.completed = false;
  return true;
}

void span_prepare_section(const SpanPrepare& prepare, PrepareNote& state) {
  section(i18n::tr("SPAN Performance mode (FP8)"));
  panel::wrapped(
      i18n::tr("With SPAN's Performance precision, the player prepares this mode by itself for the open "
               "video (about a minute, FP16 meanwhile). Redo it by hand here."));
  poll_prepare(state);
  const bool busy = state.pending != nullptr;
  note(prepare.note);
  if (prepare.offered) {
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(i18n::tr("Prepare again for this video"))) start_span_prepare(prepare, state);
    ImGui::EndDisabled();
    if (busy) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Cancel##span_prepare")) && state.stop) state.stop->request_stop();
    }
  }
  if (busy && state.progress) {
    const auto [fraction, stage] = state.progress->get();
    ImGui::ProgressBar(fraction, ImVec2(-1.0F, 0.0F), stage.c_str());
  } else if (state.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(state.text.c_str());
    ImGui::PopStyleColor();
  } else if (!state.text.empty()) {
    note(state.text);
  }
}

struct BenchDone {
  BenchResult result;
  std::string sentence;  // « Votre PC tient : … » (bench_steps.h : result_sentence)
};

struct BenchState {
  std::shared_ptr<std::future<std::expected<BenchDone, Refusal>>> pending;
  std::shared_ptr<BenchProgress> progress = std::make_shared<BenchProgress>();
  std::shared_ptr<SpanDeriveProgress> span_progress = std::make_shared<SpanDeriveProgress>();
  std::shared_ptr<std::stop_source> stop = std::make_shared<std::stop_source>();
  BenchUndo undo;  // valable après une optimisation réussie : « Annuler ces réglages »
  std::optional<BenchResult> result;
};

BenchNote::BenchNote() = default;
BenchNote::~BenchNote() {
  stop();
}
void BenchNote::stop() {
  if (state && state->stop) state->stop->request_stop();
}

bool bench_running(const BenchNote& bench) {
  return bench.state && bench.state->pending != nullptr;
}

namespace {

[[maybe_unused]] ChainContext chain_of(const BenchContext& context) {  // hors macOS
  return ChainContext{.display_height = context.display_height,
                      .display_hz = context.display_hz,
                      .source_height = context.source_height,
                      .source_fps = context.source_fps};
}

// L'étape 4 : le mode Performance de SPAN. Rien à faire quand la carte ou le moteur ne le permettent pas (une
// phrase le dit), quand il est déjà prêt, ou quand aucune vidéo n'est ouverte ; sinon le lecteur le prépare
// sur la vidéo, ce qui prend une minute environ.
[[maybe_unused]] std::expected<SpanFp8Verdict, Refusal> span_step(const BenchContext& context, Tier level,
                                                                  BenchProgress& shown,
                                                                  SpanDeriveProgress& span_progress,
                                                                  const std::stop_token& stop) {
  using State = SpanFp8Verdict::State;
  if (!context.fp8_reason.empty()) return SpanFp8Verdict{State::Unavailable, context.fp8_reason};
  if (context.span.ready) return SpanFp8Verdict{State::Ready, {}};
  if (level < Tier::Light || !context.span.offered || !context.span.run || context.span.refused)
    return SpanFp8Verdict{};
  shown.set(BenchStep::Span, i18n::tr("SPAN Performance mode (FP8) preparation"), 0.0F);
  auto done = context.span.run(span_progress, stop);
  if (done) return SpanFp8Verdict{State::Ready, {}};
  if (done.error().code == RefusalCode::Cancelled) return std::unexpected(std::move(done.error()));
  log_channel("app")->warn("SPAN Performance : {} {}", name(done.error().code), done.error().detail);
  return SpanFp8Verdict{State::Unavailable, visible_error(done.error())};
}

}  // namespace

BenchEvent bench_page(const PlayerPlace& place, const std::filesystem::path& video,
                      const BenchContext& context, BenchNote& bench, PlaySettings& draft) {
#ifdef __APPLE__
  static_cast<void>(place);
  static_cast<void>(video);
  static_cast<void>(context);
  static_cast<void>(bench);
  static_cast<void>(draft);
  return BenchEvent::None;
#else
  BenchEvent event = BenchEvent::None;
  if (!bench.state) bench.state = std::make_shared<BenchState>();
  BenchState& state = *bench.state;
  section(i18n::tr("Calibrate this machine"));
  panel::wrapped(i18n::tr(
      "The player tries a few seconds of video with its recommended settings, lightest to heaviest, "
      "keeps "
      "what your PC holds without stutter and applies it. About a minute (several the first time)."));
  // Le résultat arrive : les réglages recommandés sont posés dans le brouillon, une fois.
  if (state.pending && state.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    std::expected<BenchDone, Refusal> done = state.pending->get();
    state.pending.reset();
    if (done) {
      bench.text = done->sentence;
      bench.refused = false;
      state.result = done->result;
      PlaySettings next = draft;
      next.preset = Preset::Auto;
      draft = apply_preset(next,
                           Machine{.tier = done->result.level,
                                   .display_height = context.display_height,
                                   .display_hz = context.display_hz,
                                   .explored_work_height = done->result.work_height,
                                   .explored_rife_model = done->result.rife_model},
                           core::SourceInfo{.width = context.source_height * 16 / 9,
                                            .height = context.source_height,
                                            .fps = context.source_fps});
      event = BenchEvent::Applied;
    } else if (done.error().code == RefusalCode::Cancelled) {
      bench.text = i18n::tr("Optimization cancelled: nothing changed.");
      bench.refused = false;
      state.undo = BenchUndo{};
    } else {
      log_channel("app")->warn("optimisation : {} {}", name(done.error().code), done.error().detail);
      bench.text = visible_error(done.error());
      bench.refused = true;
      state.undo = BenchUndo{};
    }
  }
  const bool measuring = state.pending != nullptr;
  ImGui::BeginDisabled(measuring || place.executable.empty());
  if (ImGui::Button(i18n::tr("Calibrate this machine"))) {
    state.undo = capture_undo(paths::bench(context.data_dir), draft);
    state.progress = std::make_shared<BenchProgress>();
    state.span_progress = std::make_shared<SpanDeriveProgress>();
    state.stop = std::make_shared<std::stop_source>();
    state.pending = start_task<BenchDone>(
        "settings",
        [place, video, context, progress = state.progress, span_progress = state.span_progress,
         stop = state.stop]() -> std::expected<BenchDone, Refusal> {
          try {
            std::error_code error;
            const auto work = std::filesystem::temp_directory_path(error) / "ajp-bench";
            std::filesystem::create_directories(work, error);
            const BenchPlan plan{.executable = place.executable,
                                 .video = video,
                                 .work_dir = work,
                                 .data = context.data_dir,
                                 .shown = progress.get()};
            auto evaluation = evaluate_machine(plan, context.data_dir, stop->get_token());
            std::filesystem::remove_all(work, error);
            if (!evaluation) return std::unexpected(std::move(evaluation.error()));
            auto verdict =
                span_step(context, evaluation->result.level, *progress, *span_progress, stop->get_token());
            if (!verdict) return std::unexpected(std::move(verdict.error()));
            return BenchDone{.result = evaluation->result,
                             .sentence = result_sentence(evaluation->result, chain_of(context), *verdict)};
          } catch (const std::exception& error) {  // un thread qui lève ne doit pas emporter le lecteur
            return std::unexpected(Refusal{RefusalCode::ThreadFailed, error.what()});
          }
        });
    bench.text.clear();
    bench.refused = false;
    state.result.reset();
    event = BenchEvent::Started;
  }
  ImGui::EndDisabled();
  if (state.pending) {
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("Cancel##bench"))) state.stop->request_stop();
    const BenchProgress::View view = state.progress->get();
    float fraction = view.fraction;
    if (view.step == BenchStep::Span)
      fraction = overall_fraction(BenchStep::Span, state.span_progress->get().first);
    panel::wrapped(step_line(view.step, view.what).c_str());
    ImGui::ProgressBar(fraction, ImVec2(-1.0F, 0.0F));
    if (!view.note.empty()) note(view.note);
  } else if (bench.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(bench.text.c_str());
    ImGui::PopStyleColor();
  } else if (!bench.text.empty()) {
    panel::wrapped(bench.text.c_str());
    if (state.undo.valid && state.result) {
      if (ImGui::Button(i18n::tr("Undo these settings"))) {
        if (const auto restored = restore_bench_file(state.undo, paths::bench(context.data_dir)); !restored) {
          bench.text = visible_error(restored.error());
          bench.refused = true;
        } else {
          draft = state.undo.settings;
          state.undo = BenchUndo{};
          state.result.reset();
          bench.text = i18n::tr("Previous settings restored.");
          event = BenchEvent::Undone;
        }
      }
    }
  }
  return event;
#endif
}

void report_problem_section(const PlayerPlace& place, const std::string& gpu_name) {
  constexpr const char* kIssueUrl =
      "https://github.com/Ta2b/Ultimate-Anime-VideoPlayer/issues/new?template=bug.yml";
  // Ce que le dernier clic a donné, montré sous le bouton.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) : le seul thread de l'interface
  static std::string last;
  section(i18n::tr("Report a problem"));
  if (ImGui::Button(i18n::tr("Report a problem"))) {
    if (!gpu_name.empty()) note_report_fact("GPU", gpu_name);
    if (const char* driver = SDL_GetCurrentVideoDriver(); driver != nullptr)
      note_report_fact("SDL video", driver);
    ImGui::SetClipboardText(current_report(core::to_utf8(place.home)).c_str());
    last = SDL_OpenURL(kIssueUrl) ? i18n::tr("Report copied. Paste it into the page that opened.")
                                  : i18n::tr_format("Report copied. Open {} and paste it there.", kIssueUrl);
  }
  if (!last.empty()) panel::wrapped(last.c_str());
  panel::tip(i18n::tr(
      "Copies the version, the system, the graphics card, the preset and the end of the log (and the one "
      "before it after a crash), with only the file names of your videos, then opens the report page. "
      "Paste the report there."));
}

}  // namespace ajp::app::play
