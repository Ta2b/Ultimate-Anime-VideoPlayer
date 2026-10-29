#include "app/play/settings_system.h"

#include <array>
#include <ctime>
#include <expected>
#include <format>

#include <imgui.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_bench_run.h"
#include "app/play/auto_config.h"
#include "app/play/panel_style.h"
#include "app/play/span_auto.h"
#include "app/play/updater.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

void default_player_page(const PlayerPlace& place, SystemNote& last, bool& dismissed, bool already_default) {
  section("Lecteur par défaut");
  ImGui::TextWrapped(
      "Les vidéos (mkv, mp4, webm, avi, mov…) s'ouvrent dans AJP depuis le gestionnaire de fichiers.");
  if (already_default) note("AJP est déjà le lecteur par défaut des vidéos.");
  ImGui::Checkbox("Ne plus me le demander", &dismissed);
  panel::tip("Une fois coché, cette page disparaît de Général et ne revient qu'avec « Réglages avancés ».");
#ifdef __APPLE__
  static_cast<void>(last);
  note(register_player(place).value_or(std::string()));
#else
  if (last.pending && last.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = last.pending->get();
    last = done ? SystemNote{.text = *done} : SystemNote{.text = done.error().detail, .refused = true};
  }
  const bool busy = last.pending != nullptr;
  const auto start = [&](std::expected<std::string, Refusal> (*action)(const PlayerPlace&)) {
    last.pending = std::make_shared<std::future<std::expected<std::string, Refusal>>>(
        std::async(std::launch::async, action, place));
    last.text.clear();
    last.refused = false;
  };
  ImGui::BeginDisabled(busy);
  if (ImGui::Button("Faire de ce lecteur le lecteur par défaut")) start(&choose_as_default);
  ImGui::SameLine();
  if (ImGui::Button("Ajouter à « Ouvrir avec »")) start(&register_player);
  ImGui::SameLine();
  if (ImGui::Button("Retirer##inscription")) start(&unregister_player);
  ImGui::EndDisabled();
  if (busy) {
    note("En cours…");
  } else if (last.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", last.text.c_str());
    ImGui::PopStyleColor();
  } else if (!last.text.empty()) {
    note(last.text);
  }
#endif
}

bool default_player_prompt(const PlayerPlace& place, SystemNote& last, bool& dismissed, float scale) {
#ifdef __APPLE__
  static_cast<void>(place);
  static_cast<void>(last);
  static_cast<void>(dismissed);
  static_cast<void>(scale);
  return true;  // macOS : l'inscription se fait à l'installation
#else
  if (last.pending && last.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = last.pending->get();
    last = done ? SystemNote{.text = *done} : SystemNote{.text = done.error().detail, .refused = true};
  }
  const ImGuiIO& io = ImGui::GetIO();
  // Une petite fenêtre qu'on peut déplacer, au milieu de la fenêtre du lecteur au départ, indépendante du
  // panneau.
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Appearing,
                          {0.5F, 0.5F});
  ImGui::SetNextWindowSize({std::min(420.0F * scale, io.DisplaySize.x - 40.0F * scale), 0.0F},
                           ImGuiCond_Always);
  static bool brought_forward = false;  // au-dessus du panneau des réglages la première fois seulement
  if (!brought_forward) {
    ImGui::SetNextWindowFocus();
    brought_forward = true;
  }
  bool closed = false;
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize;
  if (ImGui::Begin("Lecteur par défaut", nullptr, kFlags)) {
    ImGui::TextWrapped(
        "Ouvrir vos vidéos (mkv, mp4, webm…) directement dans AJP depuis le gestionnaire de fichiers ?");
    ImGui::Spacing();
    const bool busy = last.pending != nullptr;
    const bool done = !last.text.empty() && !last.refused;
    ImGui::BeginDisabled(busy || done);
    if (ImGui::Button("Définir AJP comme lecteur par défaut")) {
      last.pending = std::make_shared<std::future<std::expected<std::string, Refusal>>>(
          std::async(std::launch::async, &choose_as_default, place));
      last.text.clear();
      last.refused = false;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(done ? "Fermer" : "Plus tard")) closed = true;
    ImGui::Checkbox("Ne plus me le demander", &dismissed);
    if (busy) {
      note("En cours…");
    } else if (last.refused) {
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped("%s", last.text.c_str());
      ImGui::PopStyleColor();
    } else if (!last.text.empty()) {
      note(last.text);
    }
    note("Réglage modifiable plus tard : Général > Lecteur par défaut.");
  }
  ImGui::End();
  return closed;
#endif
}

void updates_page(PlaySettings& d, const PlayerPlace& place, UpdateNote& result) {
  const UpdatePaths paths{
      .data = place.data, .install = place.executable.parent_path(), .executable = place.executable};
  const std::filesystem::path state_file = place.data / "update.toml";
  if (result.pending && result.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = result.pending->get();
    result = done ? UpdateNote{.text = *done} : UpdateNote{.text = done.error().detail, .refused = true};
  }
  const UpdateState state = load_update_state(state_file);
  const std::string managed = update_managed_reason(paths);

  section("Mises à jour");
  struct Mode {
    const char* key;
    const char* label;
    const char* detail;
  };
  constexpr std::array<Mode, 3> kModes{
      {{"notify", "Me prévenir", "Une notification ; vous lancez la mise à jour vous-même."},
       {"auto", "Automatique", "Téléchargée, vérifiée, installée au prochain lancement."},
       {"never", "Jamais", "Aucune vérification."}}};
  ImGui::BeginDisabled(!managed.empty());
  for (const Mode& mode : kModes) {
    if (ImGui::RadioButton(mode.label, d.update_mode == mode.key)) d.update_mode = mode.key;
    panel::tip(mode.detail);
  }
  ImGui::EndDisabled();
  if (!managed.empty()) note(managed);
  if (std::string_view(update_public_key()).empty())
    note("Ce paquet n'a pas de clé de signature : les mises à jour ne peuvent pas être vérifiées.");

  section("Détail");
  ImGui::BeginDisabled(!managed.empty());
  if (ImGui::BeginCombo("Canal", d.update_channel == "preview" ? "Préversions" : "Stable")) {
    if (ImGui::Selectable("Stable", d.update_channel == "stable")) d.update_channel = "stable";
    if (ImGui::Selectable("Préversions", d.update_channel == "preview")) d.update_channel = "preview";
    ImGui::EndCombo();
  }
  const char* frequency = d.update_frequency == "launch"   ? "À chaque lancement"
                          : d.update_frequency == "weekly" ? "Chaque semaine"
                                                           : "Chaque jour";
  if (ImGui::BeginCombo("Fréquence", frequency)) {
    if (ImGui::Selectable("À chaque lancement", d.update_frequency == "launch"))
      d.update_frequency = "launch";
    if (ImGui::Selectable("Chaque jour", d.update_frequency == "daily")) d.update_frequency = "daily";
    if (ImGui::Selectable("Chaque semaine", d.update_frequency == "weekly")) d.update_frequency = "weekly";
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  note(std::format("Version installée : {}", installed_version()));
  if (!state.latest_version.empty())
    note(std::format("Disponible : {}{}", state.latest_version,
                     state.staged_version.empty() ? "" : " (préparée)"));
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
    note(std::format("Dernière vérification : {}", text.data()));
  }

  const bool busy = result.pending != nullptr;
  ImGui::BeginDisabled(busy);
  if (ImGui::Button("Vérifier maintenant")) {
    result.pending = std::make_shared<std::future<std::expected<std::string, Refusal>>>(std::async(
        std::launch::async,
        [paths, state_file, mode = d.update_mode,
         channel = d.update_channel] -> std::expected<std::string, Refusal> {
          UpdateState current = load_update_state(state_file);
          auto message = run_update_check(paths, mode == "never" ? "notify" : mode, channel, current, {});
          if (const auto kept = save_update_state(state_file, current); !kept && message)
            return std::unexpected(kept.error());
          return message;
        }));
    result.text.clear();
    result.refused = false;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(state.previous_version.empty());
  if (ImGui::Button("Revenir à la version précédente")) {
    result.pending = std::make_shared<std::future<std::expected<std::string, Refusal>>>(
        std::async(std::launch::async, [paths, state_file] -> std::expected<std::string, Refusal> {
          UpdateState current = load_update_state(state_file);
          auto message = roll_back_update(paths, current);
          if (!message) return message;
          if (const auto kept = save_update_state(state_file, current); !kept)
            return std::unexpected(kept.error());
          return *message + " : relancez le lecteur";
        }));
    result.text.clear();
    result.refused = false;
  }
  ImGui::EndDisabled();
  ImGui::EndDisabled();
  if (busy) {
    note("En cours…");
  } else if (result.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", result.text.c_str());
    ImGui::PopStyleColor();
  } else if (!result.text.empty()) {
    note(result.text);
  }
  if (!state.last_error.empty() && result.text.empty()) note("Dernière erreur : " + state.last_error);
}

bool start_span_prepare(const SpanPrepare& prepare, PrepareNote& state) {
  if (!prepare.offered || !prepare.run || state.pending) return false;
  auto progress = std::make_shared<SpanDeriveProgress>();
  auto stop = std::make_shared<std::stop_source>();
  state.progress = progress;
  state.stop = stop;
  state.pending = std::make_shared<std::future<std::expected<std::string, Refusal>>>(std::async(
      std::launch::async, [run = prepare.run, progress, stop] -> std::expected<std::string, Refusal> {
        try {
          return run(*progress, stop->get_token());
        } catch (const std::exception& error) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, error.what()});
        }
      }));
  state.text.clear();
  state.refused = false;
  state.completed = false;
  return true;
}

void span_prepare_section(const SpanPrepare& prepare, PrepareNote& state) {
  section("Mode Performance de SPAN (FP8)");
  ImGui::TextWrapped(
      "Quand le cran « Performance » de la précision de SPAN est choisi, le lecteur prépare tout seul ce "
      "mode pour "
      "la vidéo ouverte (une minute environ, la lecture reste en FP16 pendant ce temps). Cette page sert à "
      "le refaire à la main.");
  poll_prepare(state);
  const bool busy = state.pending != nullptr;
  note(prepare.note);
  if (prepare.offered) {
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Refaire la préparation sur cette vidéo")) start_span_prepare(prepare, state);
    ImGui::EndDisabled();
    if (busy) {
      ImGui::SameLine();
      if (ImGui::Button("Annuler##span_prepare") && state.stop) state.stop->request_stop();
    }
  }
  if (busy && state.progress) {
    const auto [fraction, stage] = state.progress->get();
    ImGui::ProgressBar(fraction, ImVec2(-1.0F, 0.0F), stage.c_str());
  } else if (state.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", state.text.c_str());
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

ChainContext chain_of(const BenchContext& context) {
  return ChainContext{.display_height = context.display_height,
                      .display_hz = context.display_hz,
                      .source_height = context.source_height,
                      .source_fps = context.source_fps};
}

// L'étape 4 : le mode Performance de SPAN. Rien à faire quand la carte ou le moteur ne le permettent pas (une
// phrase le dit), quand il est déjà prêt, ou quand aucune vidéo n'est ouverte ; sinon le lecteur le prépare
// sur la vidéo, ce qui prend une minute environ.
std::expected<SpanFp8Verdict, Refusal> span_step(const BenchContext& context, Tier level,
                                                 BenchProgress& shown, SpanDeriveProgress& span_progress,
                                                 const std::stop_token& stop) {
  using State = SpanFp8Verdict::State;
  if (!context.fp8_reason.empty()) return SpanFp8Verdict{State::Unavailable, context.fp8_reason};
  if (context.span.ready) return SpanFp8Verdict{State::Ready, {}};
  if (level < Tier::Light || !context.span.offered || !context.span.run || context.span.refused)
    return SpanFp8Verdict{};
  shown.set(BenchStep::Span, "préparation du mode Performance de SPAN (FP8)", 0.0F);
  auto done = context.span.run(span_progress, stop);
  if (done) return SpanFp8Verdict{State::Ready, {}};
  if (done.error().code == RefusalCode::Cancelled) return std::unexpected(std::move(done.error()));
  return SpanFp8Verdict{State::Unavailable, done.error().detail};
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
  section("Optimiser pour ma machine");
  ImGui::TextWrapped(
      "Le lecteur essaie quelques secondes de vidéo avec ses réglages recommandés, du plus léger au plus "
      "lourd, "
      "garde ceux que votre PC tient sans saccade, et les applique. Cela dure environ une minute (plusieurs "
      "minutes la toute première fois, le temps de préparer les moteurs).");
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
      bench.text = "Optimisation annulée : rien n'a changé.";
      bench.refused = false;
      state.undo = BenchUndo{};
    } else {
      bench.text = done.error().detail;
      bench.refused = true;
      state.undo = BenchUndo{};
    }
  }
  const bool measuring = state.pending != nullptr;
  ImGui::BeginDisabled(measuring || place.executable.empty());
  if (ImGui::Button("Optimiser pour ma machine")) {
    state.undo = capture_undo(context.data_dir / "bench.toml", draft);
    state.progress = std::make_shared<BenchProgress>();
    state.span_progress = std::make_shared<SpanDeriveProgress>();
    state.stop = std::make_shared<std::stop_source>();
    state.pending = std::make_shared<std::future<std::expected<BenchDone, Refusal>>>(std::async(
        std::launch::async,
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
        }));
    bench.text.clear();
    bench.refused = false;
    state.result.reset();
    event = BenchEvent::Started;
  }
  ImGui::EndDisabled();
  if (state.pending) {
    ImGui::SameLine();
    if (ImGui::Button("Annuler##bench")) state.stop->request_stop();
    const BenchProgress::View view = state.progress->get();
    float fraction = view.fraction;
    if (view.step == BenchStep::Span)
      fraction = overall_fraction(BenchStep::Span, state.span_progress->get().first);
    ImGui::TextWrapped("%s", step_line(view.step, view.what).c_str());
    ImGui::ProgressBar(fraction, ImVec2(-1.0F, 0.0F));
    if (!view.note.empty()) note(view.note);
  } else if (bench.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", bench.text.c_str());
    ImGui::PopStyleColor();
  } else if (!bench.text.empty()) {
    ImGui::TextWrapped("%s", bench.text.c_str());
    if (state.undo.valid && state.result) {
      if (ImGui::Button("Annuler ces réglages")) {
        if (const auto restored = restore_bench_file(state.undo, context.data_dir / "bench.toml");
            !restored) {
          bench.text = restored.error().detail;
          bench.refused = true;
        } else {
          draft = state.undo.settings;
          state.undo = BenchUndo{};
          state.result.reset();
          bench.text = "Réglages d'avant rétablis.";
          event = BenchEvent::Undone;
        }
      }
    }
  }
  return event;
#endif
}

}  // namespace ajp::app::play
