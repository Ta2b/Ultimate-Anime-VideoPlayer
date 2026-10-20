#include "app/play/settings_export.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <string>
#include <system_error>

#include <imgui.h>

#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

namespace {

constexpr std::array<int, 6> kHeights{kExportChainHeight, kExportSourceHeight, 720, 1080, 1440, 2160};

std::string height_label(int height, ExportSize chain, ExportSize source) {
  const ExportSize size = export_size(chain, source, height);
  if (height == kExportChainHeight)
    return std::format("Sortie de la chaîne · {}x{}", size.width, size.height);
  if (height == kExportSourceHeight) return std::format("Source · {}x{}", size.width, size.height);
  return std::format("{}p · {}x{}", height, size.width, size.height);
}

const char* codec_note(exporter::Codec codec) {
  switch (codec) {
    case exporter::Codec::Hevc:
      return "HEVC 10 bits : lu presque partout, encodé par le GPU (NVENC, AMF, Quick Sync, VideoToolbox, "
             "VA-API).";
    case exporter::Codec::Av1:
      return "AV1 10 bits : plus compact à qualité égale ; il faut un encodeur AV1 dans le GPU (RTX 40 et "
             "50, "
             "Radeon RX 7000 et 9000, Intel Arc et Core Ultra).";
    case exporter::Codec::H264:
      return "H.264 8 bits : pour les lecteurs anciens ; les dégradés y sont moins fins.";
    case exporter::Codec::Lossless:
      return "FFV1 sans perte, encodé par le processeur : des fichiers très lourds, pour retravailler la "
             "vidéo.";
  }
  return "";
}

void copy_path(std::array<char, 1024>& to, const std::string& text) {
  const std::size_t n = std::min(text.size(), to.size() - 1);
  std::copy_n(text.data(), n, to.data());
  to[n] = '\0';
}

// Ce qui empêche d'exporter vers ce fichier, ou rien.
std::string path_problem(const std::filesystem::path& output, const std::filesystem::path& source) {
  std::error_code error;
  if (output.empty()) return "Choisissez le fichier exporté.";
  if (output.extension() != ".mkv") return "Le fichier exporté est un MKV : son nom finit par .mkv.";
  if (!std::filesystem::is_directory(output.parent_path(), error)) return "Ce dossier n'existe pas.";
  if (std::filesystem::equivalent(output, source, error)) return "Ce fichier est la vidéo lue.";
  return {};
}

}  // namespace

void export_tab(ExportDraft& draft, const PlaySettings& running, const SettingsChoices& choices,
                std::vector<Input>& actions) {
  ExportJob& job = draft.job;
  if (!draft.chain_set) {
    job.chain = running;
    draft.chain_set = true;
  }
  const ExportSize source{.width = choices.source_width, .height = choices.source_height};
  const ExportSize chain = chain_output(job.chain, source.width, source.height);

  section("Résolution");
  if (ImGui::BeginCombo("Taille", height_label(job.height, chain, source).c_str())) {
    for (const int height : kHeights) {
      if (ImGui::Selectable(height_label(height, chain, source).c_str(), height == job.height))
        job.height = height;
    }
    ImGui::EndCombo();
  }
  const ExportSize size = export_size(chain, source, job.height);
  if (size.height > chain.height) {
    note(
        "Au-dessus de la sortie de la chaîne, libplacebo agrandit encore l'image (EWA Lanczos) : un fichier "
        "plus "
        "lourd, pas plus de détail.");
  } else if (size.height < chain.height) {
    note("Sous la sortie de la chaîne, libplacebo la réduit avec son meilleur filtre (EWA Lanczos).");
  }

  // La chaîne de l'export : les onglets Image et Fluidité, sans les limites de l'écran.
  SettingsChoices chain_choices = choices;
  chain_choices.display_hz = 0.0;
  if (ImGui::CollapsingHeader("Agrandissement")) {
    ImGui::PushID("export-image");
    image_tab(job.chain, chain_choices);
    ImGui::PopID();
  }
  if (ImGui::CollapsingHeader("Interpolation")) {
    ImGui::PushID("export-fluidity");
    fluidity_tab(job.chain, chain_choices, true);
    ImGui::PopID();
  }
  if (ImGui::Button("Reprendre les réglages de la lecture")) job.chain = running;
  note(
      "Mêmes moteurs et mêmes réglages qu'à la lecture, sans aucune image sautée : au moins la qualité du "
      "direct. Ni RTX HDR, ni HDR adaptatif, ni grade : l'image de la chaîne telle quelle.");

  section("Encodage");
  if (ImGui::BeginCombo("Codec", std::string(exporter::label(job.codec)).c_str())) {
    for (int i = 0; i < exporter::kCodecCount; ++i) {
      const auto codec = static_cast<exporter::Codec>(i);
      if (ImGui::Selectable(std::string(exporter::label(codec)).c_str(), codec == job.codec))
        job.codec = codec;
    }
    ImGui::EndCombo();
  }
  note(codec_note(job.codec));
  ImGui::BeginDisabled(job.codec == exporter::Codec::Lossless);
  if (ImGui::BeginCombo("Qualité", std::string(exporter::label(job.quality)).c_str())) {
    for (int i = 0; i < exporter::kQualityCount; ++i) {
      const auto quality = static_cast<exporter::Quality>(i);
      if (ImGui::Selectable(std::string(exporter::label(quality)).c_str(), quality == job.quality))
        job.quality = quality;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (job.codec != exporter::Codec::Lossless)
    note(
        "Très haute vise une image identique à l'œil à la lecture ; Haute et Standard donnent des fichiers "
        "plus "
        "légers.");

  section("Fichier");
  const media::Rational rate = export_rate(choices.source_rate, factor_of(job.chain));
  const std::string named = std::format("{}x{} {}/{}", size.width, size.height, rate.num, rate.den);
  if (!draft.path_edited && named != draft.named_for && !choices.source_file.empty()) {
    copy_path(draft.path, prepare::utf8_path(default_export_path(choices.source_file, size, rate)));
    draft.named_for = named;
  }
  if (ImGui::InputText("Fichier", draft.path.data(), draft.path.size())) draft.path_edited = true;
  ImGui::SameLine();
  if (ImGui::SmallButton("Nom proposé")) {
    draft.path_edited = false;
    draft.named_for.clear();
  }
  const std::filesystem::path output = prepare::path_from_utf8(draft.path.data());
  const std::string problem = path_problem(output, choices.source_file);
  std::error_code error;
  if (!problem.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", problem.c_str());
    ImGui::PopStyleColor();
  } else if (std::filesystem::exists(output, error)) {
    note("Ce fichier existe : il sera remplacé.");
  }
  note(
      "L'audio, les sous-titres, les polices et les chapitres de la source sont copiés tels quels. La "
      "lecture "
      "s'arrête pendant l'export puis reprend au même endroit ; Échap annule l'export.");
  ImGui::BeginDisabled(!problem.empty());
  if (ImGui::Button("Exporter")) {
    ExportJob start = job;
    start.output = output;
    actions.emplace_back(StartExport{.job = std::move(start)});
  }
  ImGui::EndDisabled();
}

}  // namespace ajp::app::play
