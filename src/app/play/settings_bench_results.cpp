#include "app/play/settings_bench_results.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <system_error>
#include <utility>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/machine_calibration.h"
#include "app/play/panel_style.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/thread.h"
#include "media/bench_clip.h"

namespace ajp::app::play {

using panel::note;
using panel::section;

MachineNote::~MachineNote() {
  if (stop) stop->request_stop();  // les essais dépendent du lecteur qui ouvre ce panneau
}

namespace {

// Le bench refait depuis le panneau, sur son thread : les mêmes essais que l'assistant.
void start_bench(const SettingsChoices& choices, const PlaySettings& base, MachineNote& bench) {
  bench.fraction = std::make_shared<std::atomic<float>>(0.0F);
  bench.stop = std::make_shared<std::stop_source>();
  bench.problem.clear();
  const std::filesystem::path data = choices.settings_dir;
  std::error_code error;
  const BenchPlan plan{
      .executable = choices.place.executable,
      .video = choices.bench_clip,
      .work_dir = calibration_work_dir(data),
      .data = data,
      .display_width = choices.display_width > 0 ? choices.display_width : choices.display_height * 16 / 9,
      .display_height = choices.display_height,
      .display_hdr = choices.hdr_output,
      .display_hz = choices.display_hz};
  const int height = choices.display_height > 0 ? choices.display_height : 1080;
  const auto tier = static_cast<Tier>(std::clamp(choices.estimated_tier, 0, kTierCount - 1));
  bench.pending = start_task<MachineCalibration>(
      "calibration",
      [plan, base, height, tier, fraction = bench.fraction,
       stop = bench.stop]() -> std::expected<MachineCalibration, Refusal> {
        try {
          return calibrate_machine(plan, base, height, tier, stop->get_token(),
                                   [fraction](float f) { fraction->store(f); });
        } catch (const std::exception& thrown) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, thrown.what()});
        }
      });
}

// La couleur d'une ligne : vert, la machine tient avec de la marge (la recommandée l'est toujours) ; orange,
// elle tient de justesse ou rate de peu ; rouge, elle ne suit pas.
enum class Rating : std::uint8_t { Good, Fair, Poor };

Rating rate(const CandidateTrial& trial, bool recommended) {
  if (trial.lost_percent < 0.0 || trial.target_fps <= 0.0) return Rating::Poor;
  const double fps = trial.fps / trial.target_fps;
  const double low = trial.low_fps / trial.target_fps;
  if (trial.holds) {
    if (recommended || (fps >= 0.98 && low >= 0.85 && trial.lost_percent <= 0.5)) return Rating::Good;
    return Rating::Fair;
  }
  return trial.lost_percent <= 10.0 && fps >= 0.85 ? Rating::Fair : Rating::Poor;
}

ImVec4 color_of(Rating rating) {
  switch (rating) {
    case Rating::Good:
      return panel::kGood;
    case Rating::Fair:
      return ImVec4{1.0F, 0.62F, 0.25F, 1.0F};
    case Rating::Poor:
      break;
  }
  return ImVec4{1.0F, 0.38F, 0.38F, 1.0F};
}

// La réduction avant SPAN : la hauteur de travail, « no » quand l'image garde sa taille (sans SPAN, ou au
// moins la hauteur de la séquence).
std::string downscale_text(const ChainCandidate& chain) {
  if (!chain.span || chain.height <= 0 || chain.height >= media::BenchClipSpec{}.height) return "no";
  return std::format("{}p", chain.height);
}

// Sous l'en-tête du tableau : les modèles que les essais ont joués, et pourquoi l'Auto ne suit pas la mesure
// quand c'est le cas (version plus ancienne, batterie, autre moteur, autre carte ou écran), ou que le pilote
// a changé depuis.
void measure_notes(const MachineCalibration& r, const CalibrationHost& host) {
  note(i18n::tr_format("SPAN: {} in FP16 · RIFE: {}",
                       model_label(r.models.span.empty() ? std::string(kDefaultModelFile) : r.models.span),
                       rife_label(r.models.rife.empty() ? std::string(kLightRifeModel) : r.models.rife)));
  switch (calibration_use(r, host)) {
    case CalibrationUse::OldVersion:
      note(i18n::tr("Measured by an older version: Auto keeps its estimate. Run the calibration again."));
      break;
    case CalibrationUse::OnBattery:
      note(i18n::tr("Measured on battery: Auto keeps its estimate. Run the calibration again plugged in."));
      break;
    case CalibrationUse::OtherEngine:
      note(i18n::tr("Measured with another engine: Auto keeps its estimate. Run the calibration again."));
      break;
    case CalibrationUse::OtherMachine:
      note(i18n::tr(
          "Measured on another card or display: Auto keeps its estimate. Run the calibration again."));
      break;
    case CalibrationUse::Followed:
      if (calibration_driver_changed(r, host))
        note(i18n::tr(
            "The graphics driver changed since this measure: run the calibration again to refresh it."));
      break;
  }
}

// Le bench refait est fini : son résultat au tableau, et la chaîne que l'Auto en suit, tout de suite
// (B07-13).
void collect_bench(const SettingsChoices& choices, MachineNote& bench) {
  if (!bench.pending || bench.pending->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
  auto done = bench.pending->get();
  bench.pending.reset();
  if (done) {
    bench.result = std::move(*done);
    bench.fresh = followed_chain(*bench.result, choices.calibration_host);
  } else if (done.error().code != RefusalCode::Cancelled) {
    log_channel("app")->warn("banc : {} {}", name(done.error().code), done.error().detail);
    bench.problem = visible_error(done.error());
  }
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool bench_results_section(PlaySettings& d, const SettingsChoices& choices, MachineNote& bench,
                           std::vector<Input>& actions) {
  section(i18n::tr("Calibration results"));
  bool applied = false;
  if (!bench.read) {
    bench.read = true;
    if (auto loaded = load_calibration_at(choices.settings_dir, choices.calibration_host.display_height))
      bench.result = std::move(*loaded);
  }
  collect_bench(choices, bench);
  if (bench.running()) {
    note(
        i18n::tr("Testing settings on a very demanding sequence: 5 minutes, up to 40 with TensorRT (it "
                 "builds its engines)."));
    ImGui::ProgressBar(bench.fraction ? bench.fraction->load() : 0.0F, {-1.0F, 0.0F});
    if (ImGui::Button(i18n::tr("Cancel##calibration"))) bench.stop->request_stop();
    return false;
  }
  if (!bench.problem.empty()) note(i18n::tr_format("The calibration failed: {}", bench.problem));
  if (!bench.result) {
    note(i18n::tr("No calibration yet: the settings are an estimate from the card."));
  } else {
    const MachineCalibration& r = *bench.result;
    note(i18n::tr_format("{} · {} · display {}p", r.gpu.empty() ? i18n::tr("Unknown graphics card") : r.gpu,
                         r.date, r.display_height));
    measure_notes(r, choices.calibration_host);
    constexpr ImGuiTableFlags kFlags =
        ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX;
    if (ImGui::BeginTable("##bench", 6, kFlags)) {
      ImGui::TableSetupColumn(i18n::tr("Downscale"), ImGuiTableColumnFlags_WidthStretch, 1.2F);
      ImGui::TableSetupColumn(i18n::tr("Upscale"), ImGuiTableColumnFlags_WidthStretch, 1.0F);
      ImGui::TableSetupColumn("RIFE", ImGuiTableColumnFlags_WidthStretch, 1.0F);
      ImGui::TableSetupColumn("FPS", ImGuiTableColumnFlags_WidthStretch, 1.4F);
      ImGui::TableSetupColumn(i18n::tr("1% low"), ImGuiTableColumnFlags_WidthStretch, 1.0F);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableHeadersRow();
      int row = 0;
      for (const CandidateTrial& trial : r.tried) {
        ImGui::PushID(row++);
        const bool recommended = trial.chain == r.chosen;
        const ImVec4 color = color_of(rate(trial, recommended));
        ImGui::TableNextRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                               ImGui::GetColorU32(ImVec4{color.x, color.y, color.z, 0.14F}));
        const auto cell = [&](const std::string& text) {
          ImGui::TableNextColumn();
          ImGui::TextColored(color, "%s", text.c_str());
        };
        const bool rendered = trial.lost_percent >= 0.0;
        cell(downscale_text(trial.chain));
        cell(trial.chain.span ? "on" : "off");  // les mots demandés, dans les deux langues
        cell(trial.chain.rife ? "on" : "off");
        cell(rendered ? std::format("{:.1f} / {:.0f}", trial.fps, trial.target_fps) : std::string("-"));
        cell(rendered ? std::format("{:.1f}", trial.low_fps) : std::string("-"));
        if (ImGui::IsItemHovered() && rendered)
          ImGui::SetTooltip("%s",
                            french_spaces(i18n::tr_format("{:.1f} % lost", trial.lost_percent)).c_str());
        ImGui::TableNextColumn();
        if (ImGui::SmallButton(i18n::tr("Apply"))) {
          d = candidate_settings(trial.chain, std::move(d), r.models);
          applied = true;
        }
        if (recommended) panel::badge(i18n::tr("Recommended"), panel::kGood);
        ImGui::PopID();
      }
      ImGui::EndTable();
    }
    note(
        i18n::tr("Green: recommended, with margin. Orange: holds just barely, or misses a little. Red: this "
                 "machine cannot keep up."));
    panel::tip(i18n::tr(
        "Interpolation runs in Auto Instant mode. The working resolution goes up to the 1080p of the "
        "test sequence; SPAN doubles it."));
  }
  if (ImGui::Button(bench.result ? i18n::tr("Run the calibration again") : i18n::tr("Run the calibration"))) {
    if (choices.source_file.empty()) {
      start_bench(choices, d, bench);
    } else {
      // Pendant une lecture (P3.12) : elle se ferme d'abord, sa mémoire vidéo ne fausse plus la mesure.
      actions.emplace_back(StartCalibration{});
      bench.read = false;  // calibration.toml relu au retour
    }
  }
  panel::tip(i18n::tr("Playback closes during the calibration, then resumes at the same spot."));
  return applied;
}

}  // namespace ajp::app::play
