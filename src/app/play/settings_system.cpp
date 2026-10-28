#include "app/play/settings_system.h"

#include <expected>

#include <imgui.h>

#include "app/play/auto_bench_run.h"
#include "app/play/panel_style.h"

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
  panel::tip("Le bouton « Faire de ce lecteur le lecteur par défaut » quitte Général et reste tout en bas d'Avancé.");
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
      "pour votre carte et votre pilote. Un PC chaud ou occupé donne un résultat plus bas : relancez-le au besoin.");
  if (bench.pending && bench.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = bench.pending->get();
    bench = done ? BenchNote{.text = *done} : BenchNote{.text = done.error().detail, .refused = true};
  }
  const bool measuring = bench.pending != nullptr;
  bool started = false;
  ImGui::BeginDisabled(measuring || video.empty() || place.executable.empty());
  if (ImGui::Button("Évaluer ma machine")) {
    bench.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(std::async(
        std::launch::async, [place, video]() -> std::expected<std::string, Refusal> {
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
