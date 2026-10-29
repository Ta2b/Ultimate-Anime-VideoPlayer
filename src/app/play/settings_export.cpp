#include "app/play/settings_export.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <system_error>

#include <imgui.h>

#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/text_fit.h"
#include "export/container.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

namespace {

// Au-delà, l'arrondi du MKV à la milliseconde se voit dans les écarts entre images.
constexpr int kExactRateFps = 100;

constexpr std::array<int, 6> kHeights{kExportChainHeight, kExportSourceHeight, 720, 1080, 1440, 2160};

std::string height_label(int height, ExportSize chain, ExportSize source) {
  const ExportSize size = export_size(chain, source, height);
  if (height == kExportChainHeight)
    return i18n::tr_format("Playback output · {}x{}", size.width, size.height);
  if (height == kExportSourceHeight) return i18n::tr_format("Source · {}x{}", size.width, size.height);
  return std::format("{}p · {}x{}", height, size.width, size.height);
}

const char* codec_note(exporter::Codec codec) {
  switch (codec) {
    case exporter::Codec::Hevc:
      return i18n::tr("HEVC 10-bit: plays almost everywhere, encoded by the GPU.");
    case exporter::Codec::Av1:
      return i18n::tr(
          "AV1 10-bit: smaller at equal quality; needs a GPU AV1 encoder (RTX 40/50, Radeon RX 7000/9000, "
          "Intel Arc).");
    case exporter::Codec::H264:
      return i18n::tr("H.264 8-bit: for older players; gradients are coarser.");
    case exporter::Codec::Lossless:
      return i18n::tr("FFV1 lossless, 4:2:0 10-bit, encoded by the CPU: very large files, for editing.");
  }
  return "";
}

void copy_path(std::array<char, 1024>& to, const std::string& text) {
  copy_text(to, text);
}

// Ce qui empêche d'exporter vers ce fichier, ou rien.
std::string path_problem(const std::filesystem::path& output, const std::filesystem::path& source,
                         exporter::Codec codec) {
  std::error_code error;
  if (output.empty()) return i18n::tr("Choose the output file.");
  const std::optional<exporter::Container> container = exporter::container_of(output);
  if (!container) return i18n::tr("The output is an MKV, MP4 or MOV: its name ends in .mkv, .mp4 or .mov.");
  if (!exporter::carries(*container, codec)) return i18n::tr("FFV1 does not fit in a MOV: use .mkv or .mp4.");
  if (!std::filesystem::is_directory(output.parent_path(), error))
    return i18n::tr("This folder does not exist.");
  if (std::filesystem::equivalent(output, source, error))
    return i18n::tr("This file is the video being played.");
  return {};
}

}  // namespace

void export_tab(ExportDraft& draft, const PlaySettings& running, const SettingsChoices& choices,
                std::vector<Input>& actions) {
  if (choices.source_file.empty()) {  // l'écran d'accueil : aucune vidéo à exporter
    note(i18n::tr("Open a video: the export uses its settings and is named next to it."));
    return;
  }
  ExportJob& job = draft.job;
  // Une autre vidéo (glisser-déposer, fichier suivant) : la chaîne, le nom proposé et un chemin
  // saisi pour la précédente ne valent plus.
  if (draft.source != choices.source_file) {
    draft.source = choices.source_file;
    draft.chain_set = false;
    draft.path_edited = false;
    draft.named_for.clear();
  }
  if (!draft.chain_set) {
    job.chain = running;
    draft.chain_set = true;
  }
  const ExportSize source{.width = choices.source_width, .height = choices.source_height};
  const ExportSize chain = chain_output(job.chain, source.width, source.height);

  section(i18n::tr("Resolution"));
  if (ImGui::BeginCombo(i18n::tr("Size"), height_label(job.height, chain, source).c_str())) {
    for (const int height : kHeights) {
      if (ImGui::Selectable(height_label(height, chain, source).c_str(), height == job.height))
        job.height = height;
    }
    ImGui::EndCombo();
  }
  const ExportSize size = export_size(chain, source, job.height);
  if (size.height > chain.height) {
    note(
        i18n::tr("Above the playback output, libplacebo scales further (EWA Lanczos): a bigger file, no more "
                 "detail."));
  } else if (size.height < chain.height) {
    note(i18n::tr("Below the playback output, libplacebo scales down with its best filter (EWA Lanczos)."));
  }

  // La chaîne de l'export : les onglets Image et Fluidité, sans les limites de l'écran.
  SettingsChoices chain_choices = choices;
  chain_choices.display_hz = 0.0;
  if (ImGui::CollapsingHeader(i18n::tr("Upscaling"))) {
    ImGui::PushID("export-image");
    image_tab(job.chain, chain_choices);
    ImGui::PopID();
  }
  if (ImGui::CollapsingHeader(i18n::tr("Interpolation"))) {
    ImGui::PushID("export-fluidity");
    fluidity_tab(job.chain, chain_choices, true);
    ImGui::PopID();
  }
  if (ImGui::Button(i18n::tr("Use the playback settings"))) job.chain = running;
  note(
      i18n::tr("Same engines and settings as playback, with no skipped frame. No RTX HDR, adaptive HDR or "
               "grading: the chain's picture as is."));

  section(i18n::tr("Encoding"));
  if (ImGui::BeginCombo(i18n::tr("Codec"), std::string(exporter::label(job.codec)).c_str())) {
    for (int i = 0; i < exporter::kCodecCount; ++i) {
      const auto codec = static_cast<exporter::Codec>(i);
      if (ImGui::Selectable(std::string(exporter::label(codec)).c_str(), codec == job.codec))
        job.codec = codec;
    }
    ImGui::EndCombo();
  }
  note(codec_note(job.codec));
  ImGui::BeginDisabled(job.codec == exporter::Codec::Lossless);
  if (ImGui::BeginCombo(i18n::tr("Quality"), std::string(exporter::label(job.quality)).c_str())) {
    for (int i = 0; i < exporter::kQualityCount; ++i) {
      const auto quality = static_cast<exporter::Quality>(i);
      if (ImGui::Selectable(std::string(exporter::label(quality)).c_str(), quality == job.quality))
        job.quality = quality;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (job.codec != exporter::Codec::Lossless)
    note(i18n::tr(
        "Very high aims for a picture identical to playback; High and Standard make smaller files."));
  ImGui::Checkbox(i18n::tr("Constant frame rate"), &job.constant_rate);
  note(job.constant_rate ? i18n::tr("At scene cuts and held drawings the previous frame repeats, so the file "
                                    "keeps a constant frame rate any editor reads.")
                         : i18n::tr("Frames the chain does not interpolate are left out: variable frame "
                                    "rate, a bit smaller, fine in players."));

  section(i18n::tr("File"));
  const media::Rational rate = export_rate(choices.source_rate, factor_of(job.chain));
  const std::string named = std::format("{}x{} {}/{}", size.width, size.height, rate.num, rate.den);
  if (!draft.path_edited && named != draft.named_for && !choices.source_file.empty()) {
    copy_path(draft.path, prepare::utf8_path(default_export_path(choices.source_file, size, rate)));
    draft.named_for = named;
  }
  // Un champ sur toute la largeur, son bouton à droite : le titre de l'encadré dit ce que c'est.
  const ImGuiStyle& style = ImGui::GetStyle();
  ImGui::SetNextItemWidth(-(ImGui::CalcTextSize(i18n::tr("Suggested name")).x + 2.0F * style.FramePadding.x +
                            style.ItemSpacing.x));
  if (ImGui::InputText("##fichier", draft.path.data(), draft.path.size())) draft.path_edited = true;
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Suggested name"))) {
    draft.path_edited = false;
    draft.named_for.clear();
  }
  const std::filesystem::path output = prepare::path_from_utf8(draft.path.data());
  const std::string problem = path_problem(output, choices.source_file, job.codec);
  const exporter::Container container =
      exporter::container_of(output).value_or(exporter::Container::Matroska);
  std::error_code error;
  if (!problem.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", problem.c_str());
    ImGui::PopStyleColor();
  } else if (std::filesystem::exists(output, error)) {
    note(i18n::tr("This file exists: it is replaced only when the export succeeds."));
  }
  if (container == exporter::Container::Matroska) {
    note(
        i18n::tr("Audio, subtitles, fonts and chapters are copied as is. Playback stops during the export; "
                 "Esc cancels."));
    if (rate.den > 0 && rate.num >= kExactRateFps * rate.den)
      note(
          i18n::tr("At this frame rate MKV rounds frames to the millisecond; .mp4 or .mov keeps exact times "
                   "but drops subtitles and fonts."));
  } else {
    note(
        i18n::tr("MP4 and MOV keep exact frame times, chapters and compatible audio, but no subtitles or "
                 "fonts. Use .mkv to keep everything. Esc cancels."));
  }
  ImGui::BeginDisabled(!problem.empty());
  if (ImGui::Button(i18n::tr("Export"))) {
    ExportJob start = job;
    start.output = output;
    actions.emplace_back(StartExport{.job = std::move(start)});
  }
  ImGui::EndDisabled();
}

}  // namespace ajp::app::play
