#include "app/play/settings_system.h"

#include <array>
#include <ctime>
#include <expected>
#include <format>

#include <imgui.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_bench_run.h"
#include "app/play/panel_style.h"
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
  panel::tip("Le lecteur ne propose plus de devenir le lecteur par défaut au lancement.");
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
    last.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
        std::async(std::launch::async, action, place));
    last.text.clear();
    last.refused = false;
  };
  ImGui::BeginDisabled(busy);
  if (ImGui::Button("Faire de ce lecteur le lecteur par défaut")) start(&choose_as_default);
  panel::same_line_if_fits("Ajouter à « Ouvrir avec »");
  if (ImGui::Button("Ajouter à « Ouvrir avec »")) start(&register_player);
  panel::same_line_if_fits("Retirer de « Ouvrir avec »");
  if (ImGui::Button("Retirer de « Ouvrir avec »")) start(&unregister_player);
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
      last.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
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
    note("Réglage modifiable plus tard : Général > Application.");
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

  section("Canal et fréquence");
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
    result.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(std::async(
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
    result.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
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
  if (state.previous_version.empty()) panel::tip("Grisé : aucune mise à jour n'a encore été installée.");
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
  state.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(std::async(
      std::launch::async, [run = prepare.run, progress, stop] -> std::expected<std::string, Refusal> {
        try {
          return run(*progress, stop->get_token());
        } catch (const std::exception& error) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, error.what()});
        }
      }));
  state.text.clear();
  state.refused = false;
  return true;
}

void span_prepare_section(PlaySettings& d, const SpanPrepare& prepare, PrepareNote& state) {
  section("Mode Performance de SPAN (FP8)");
  ImGui::TextWrapped(
      "Prépare le calcul de SPAN en FP8 par noyaux CUTLASS, que le cran « Performance » de la précision de "
      "SPAN "
      "sert à la place de TensorRT FP16. Le lecteur tire douze images de la vidéo ouverte, calibre sur huit, "
      "vérifie sur quatre mises de côté (PSNR et SSIM contre le SPAN en FP16), puis écrit un paquet dans le "
      "dossier des modèles. Aucune image n'est gardée. Une minute environ, en tâche de fond ; la lecture "
      "reste "
      "sur TensorRT FP16 tant que le résultat n'est pas bon.");
  if (state.pending && state.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = state.pending->get();
    state.pending.reset();
    state.progress.reset();
    state.stop.reset();
    state.text = done ? *done : done.error().detail;
    state.refused = !done;
  }
  const bool busy = state.pending != nullptr;
  if (!prepare.offered) {
    note(prepare.note);
    ImGui::BeginDisabled(true);
    ImGui::Button("Préparer le mode Performance (SPAN FP8)");
    ImGui::EndDisabled();
  } else {
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Préparer le mode Performance (SPAN FP8)")) start_span_prepare(prepare, state);
    ImGui::EndDisabled();
    if (busy) {
      ImGui::SameLine();
      if (ImGui::Button("Annuler##span_prepare") && state.stop) state.stop->request_stop();
    }
    note(prepare.note);
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
  ImGui::BeginDisabled(!prepare.offered);
  ImGui::Checkbox("Préparer en tâche de fond", &d.span_prepare_background);
  ImGui::EndDisabled();
  panel::tip(
      "Après « Évaluer ma machine », sur une vidéo ouverte, quand le niveau gardé est Rapide ou plus, "
      "prépare le "
      "mode Performance sans rien demander. Désactivé par défaut : la préparation prend la mémoire vidéo et "
      "une "
      "partie du GPU pendant la lecture.");
}

bool prepare_after_bench(const PlaySettings& d, const SpanPrepare& prepare, const std::filesystem::path& data,
                         PrepareNote& state) {
  if (!d.span_prepare_background || !prepare.offered || prepare.ready || state.pending) return false;
  const auto kept = load_bench(data / "bench.toml");
  if (!kept || kept->level < Tier::Fast) return false;
  return start_span_prepare(prepare, state);
}

bool bench_page(const PlayerPlace& place, const std::filesystem::path& video, BenchNote& bench) {
#ifdef __APPLE__
  static_cast<void>(place);
  static_cast<void>(video);
  static_cast<void>(bench);
  return false;
#else
  section("Évaluer ma machine");
  ImGui::TextWrapped(
      "Joue 12 secondes d'une séquence de test (la vôtre si une vidéo est ouverte) avec chaque niveau de "
      "chaîne, "
      "sans rien afficher ni émettre de son, et garde le plus haut qui tient "
      "pour votre carte et votre pilote. Un PC chaud ou occupé donne un résultat plus bas : relancez-le au "
      "besoin.");
  bench.finished = false;
  if (bench.pending && bench.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = bench.pending->get();
    bench = done ? BenchNote{.text = *done, .finished = true}
                 : BenchNote{.text = done.error().detail, .refused = true};
  }
  const bool measuring = bench.pending != nullptr;
  bool started = false;
  ImGui::BeginDisabled(measuring || place.executable.empty());
  if (ImGui::Button("Évaluer ma machine")) {
    bench.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
        std::async(std::launch::async, [place, video] -> std::expected<std::string, Refusal> {
          std::error_code error;
          const auto work = std::filesystem::temp_directory_path(error) / "ajp-bench";
          std::filesystem::create_directories(work, error);
          const BenchPlan plan{
              .executable = place.executable, .video = video, .work_dir = work, .data = place.data};
          auto report = evaluate_machine(plan, place.data, std::stop_token{});
          std::filesystem::remove_all(work, error);
          return report;
        }));
    bench.text.clear();
    bench.refused = false;
    started = true;
  }
  ImGui::EndDisabled();
  if (video.empty())
    note("Aucune vidéo à ouvrir : le lecteur joue sa propre séquence de test, sans rien afficher.");
  if (measuring) {
    note("Chargement des moteurs puis benchmark en cours (environ une minute)…");
  } else if (bench.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", bench.text.c_str());
    ImGui::PopStyleColor();
  } else if (!bench.text.empty()) {
    note(bench.text + "À appliquer à la prochaine vidéo ouverte.");
  }
  return started;
#endif
}

}  // namespace ajp::app::play
