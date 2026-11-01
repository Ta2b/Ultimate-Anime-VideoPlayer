#include "app/play/settings_bench_results.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <system_error>
#include <utility>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"

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
  const BenchPlan plan{.executable = choices.place.executable,
                       .video = data / "bench-clip-v2.mkv",
                       .work_dir = std::filesystem::temp_directory_path(error) / "ajp-calibration",
                       .data = data,
                       .display_width = choices.display_height * 16 / 9,
                       .display_height = choices.display_height,
                       .display_hdr = choices.hdr_output};
  const int height = choices.display_height > 0 ? choices.display_height : 1080;
  bench.pending = std::make_shared<std::future<std::expected<MachineCalibration, Refusal>>>(std::async(
      std::launch::async,
      [plan, base, height, fraction = bench.fraction,
       stop = bench.stop]() -> std::expected<MachineCalibration, Refusal> {
        try {
          return calibrate_machine(plan, base, height, stop->get_token(),
                                   [fraction](float f) { fraction->store(f); });
        } catch (const std::exception& thrown) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, thrown.what()});
        }
      }));
}

std::string outcome_text(const CandidateTrial& trial) {
  if (trial.lost_percent < 0.0) return i18n::tr("nothing rendered");
  return i18n::tr_format("{} · {:.1f} / {:.0f} fps · 1 % low {:.1f} · {:.1f} % lost",
                         trial.holds ? i18n::tr("holds") : i18n::tr("does not hold"), trial.fps,
                         trial.target_fps, trial.low_fps, trial.lost_percent);
}

}  // namespace

bool bench_results_section(PlaySettings& d, const SettingsChoices& choices, MachineNote& bench,
                           std::vector<Input>& actions) {
  section(i18n::tr("Bench results"));
  bool applied = false;
  if (!bench.read) {
    bench.read = true;
    if (auto loaded = load_calibration(calibration_file(choices.settings_dir)))
      bench.result = std::move(*loaded);
  }
  if (bench.pending && bench.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto done = bench.pending->get();
    bench.pending.reset();
    if (done) {
      bench.result = std::move(*done);
    } else if (done.error().code != RefusalCode::Cancelled) {
      bench.problem = done.error().detail;
    }
  }
  if (bench.running()) {
    note(i18n::tr("Testing settings on a very demanding sequence: a few minutes."));
    ImGui::ProgressBar(bench.fraction ? bench.fraction->load() : 0.0F, {-1.0F, 0.0F});
    if (ImGui::Button(i18n::tr("Cancel##calibration"))) bench.stop->request_stop();
    return false;
  }
  if (!bench.problem.empty()) note(i18n::tr_format("The bench failed: {}", bench.problem));
  if (!bench.result) {
    note(i18n::tr("No bench yet: the settings are an estimate from the card."));
  } else {
    const MachineCalibration& r = *bench.result;
    note(i18n::tr_format("{} · {} · display {}p", r.gpu.empty() ? i18n::tr("Unknown graphics card") : r.gpu,
                         r.date, r.display_height));
    constexpr ImGuiTableFlags kFlags =
        ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##bench", 3, kFlags)) {
      ImGui::TableSetupColumn(i18n::tr("Settings"), ImGuiTableColumnFlags_WidthStretch, 3.0F);
      ImGui::TableSetupColumn(i18n::tr("Result"), ImGuiTableColumnFlags_WidthStretch, 3.0F);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableHeadersRow();
      int row = 0;
      for (const CandidateTrial& trial : r.tried) {
        ImGui::PushID(row++);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(describe(trial.chain).c_str());
        if (trial.chain == r.chosen) panel::badge(i18n::tr("Recommended"), panel::kGood);
        ImGui::TableNextColumn();
        ImGui::TextColored(trial.holds ? panel::kGood : panel::kWarn, "%s", outcome_text(trial).c_str());
        ImGui::TableNextColumn();
        if (ImGui::SmallButton(i18n::tr("Apply"))) {
          d = candidate_settings(trial.chain, std::move(d));
          applied = true;
        }
        ImGui::PopID();
      }
      ImGui::EndTable();
    }
    panel::tip(
        i18n::tr("Interpolation is always RIFE 4.22 lite in Auto Fast mode (experimental). The "
                 "working resolution stops at half the display: SPAN doubles it."));
  }
  if (ImGui::Button(bench.result ? i18n::tr("Run the bench again") : i18n::tr("Run the bench"))) {
    start_bench(choices, d, bench);
    actions.emplace_back(Command{cmd::Pause{}});
  }
  panel::tip(i18n::tr("Playback pauses during the bench."));
  return applied;
}

}  // namespace ajp::app::play
