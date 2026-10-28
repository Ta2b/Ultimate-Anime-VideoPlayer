#include "app/play/settings_system.h"

#include <array>
#include <ctime>
#include <expected>
#include <format>

#include <imgui.h>

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
  panel::tip(
      "Le bouton « Faire de ce lecteur le lecteur par défaut » quitte Général et reste tout en bas "
      "d'Avancé.");
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

bool bench_page(const PlayerPlace& place, const std::filesystem::path& video, BenchNote& bench) {
#ifdef __APPLE__
  static_cast<void>(place);
  static_cast<void>(video);
  static_cast<void>(bench);
  return false;
#else
  section("Évaluer ma machine");
  ImGui::TextWrapped(
      "Joue 12 secondes de la vidéo avec chaque niveau de chaîne, son coupé, et garde le plus haut qui tient "
      "pour votre carte et votre pilote. Un PC chaud ou occupé donne un résultat plus bas : relancez-le au "
      "besoin.");
  if (bench.pending && bench.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = bench.pending->get();
    bench = done ? BenchNote{.text = *done} : BenchNote{.text = done.error().detail, .refused = true};
  }
  const bool measuring = bench.pending != nullptr;
  bool started = false;
  ImGui::BeginDisabled(measuring || video.empty() || place.executable.empty());
  if (ImGui::Button("Évaluer ma machine")) {
    bench.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
        std::async(std::launch::async, [place, video] -> std::expected<std::string, Refusal> {
          std::error_code error;
          const auto work = std::filesystem::temp_directory_path(error) / "ajp-bench";
          std::filesystem::create_directories(work, error);
          const BenchPlan plan{.executable = place.executable, .video = video, .work_dir = work};
          auto report = evaluate_machine(plan, place.data, std::stop_token{});
          std::filesystem::remove_all(work, error);
          return report;
        }));
    bench.text.clear();
    bench.refused = false;
    started = true;
  }
  ImGui::EndDisabled();
  if (video.empty()) note("Ouvrez d'abord une vidéo : l'évaluation la joue.");
  if (measuring) {
    note("Évaluation en cours (environ une minute)… la lecture est en pause.");
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
