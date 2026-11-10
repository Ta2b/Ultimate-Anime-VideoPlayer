#include "app/play/first_run.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/engine_install.h"
#include "app/play/engine_switch.h"
#include "app/play/machine_calibration.h"
#include "app/play/panel_style.h"
#include "core/engines.h"
#include "core/text.h"

namespace ajp::app::play {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
constexpr ImVec4 kWarnColor{1.0F, 0.78F, 0.42F, 1.0F};
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

void note(const std::string& text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  panel::wrapped(text.c_str());
  ImGui::PopStyleColor();
}

void section(const char* title) {
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(title);
  ImGui::PopStyleColor();
  ImGui::Separator();
}

ImU32 badge_color(core::Badge badge) {
  switch (badge) {
    case core::Badge::Recommended:
      return IM_COL32(64, 132, 230, 255);
    case core::Badge::Compatibility:
      return IM_COL32(110, 110, 124, 255);
    case core::Badge::NvidiaOnly:
      return IM_COL32(92, 140, 30, 255);
    case core::Badge::AmdOnly:
      return IM_COL32(170, 40, 40, 255);
    case core::Badge::IntelOnly:
      return IM_COL32(20, 110, 180, 255);
    default:
      return IM_COL32(80, 84, 96, 255);
  }
}

// Une étiquette arrondie sur la ligne en cours.
void chip(core::Badge badge) {
  const std::string_view text = badge_label(badge);
  const ImVec2 size = ImGui::CalcTextSize(text.data(), text.data() + text.size());
  const ImGuiStyle& style = ImGui::GetStyle();
  const ImVec2 pad{style.FramePadding.x * 0.8F, 1.0F};
  ImVec2 at = ImGui::GetCursorScreenPos();
  at.y += (ImGui::GetFrameHeight() - size.y - pad.y * 2.0F) * 0.5F;  // centrée sur la case à cocher
  const ImVec2 end{at.x + size.x + pad.x * 2.0F, at.y + size.y + pad.y * 2.0F};
  ImDrawList* list = ImGui::GetWindowDrawList();
  list->AddRectFilled(at, end, badge_color(badge), size.y * 0.5F);
  list->AddText({at.x + pad.x, at.y + pad.y}, IM_COL32(255, 255, 255, 240), text.data(),
                text.data() + text.size());
  ImGui::Dummy({end.x - at.x, ImGui::GetFrameHeight()});
}

void engine_row(EngineRow& row) {
  ImGui::PushID(static_cast<int>(row.offer.id));
  const bool locked = row.offer.id == core::EngineId::Vulkan || !row.usable;
  ImGui::BeginDisabled(locked);
  ImGui::Checkbox("##chosen", &row.chosen);
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  const std::string_view label = engine_label(row.offer.id);
  ImGui::TextUnformatted(label.data(), label.data() + label.size());
  for (const core::Badge badge : row.offer.badges) {
    ImGui::SameLine();
    chip(badge);
  }
  ImGui::Indent(ImGui::GetFrameHeightWithSpacing());
  note(std::format("{} · {}", reason_label(row.offer.reason), row.status));
  // La licence d'un moteur coché à télécharger : lue et acceptée par l'utilisateur, sinon le
  // moteur reste hors du choix.
  if (row.download && row.chosen && !row.license.empty()) {
    ImGui::Checkbox(i18n::tr("I accept its licence"), &row.license_accepted);
    ImGui::SameLine();
    ImGui::TextLinkOpenURL(i18n::tr("Read the licence"), row.license.c_str());
    if (!row.license_accepted)
      ImGui::TextColored(kWarnColor, "%s", i18n::tr("Not installed until its licence is accepted."));
  }
  ImGui::Unindent(ImGui::GetFrameHeightWithSpacing());
  ImGui::Spacing();
  ImGui::PopID();
}

// Une ligne d'un moteur téléchargeable : installé ici (arbre de construction, ou extrait), à
// télécharger, ou absent de l'index de cette version.
void downloadable_row(EngineRow& row, const EnginesHere& here) {
  const bool built = row.offer.id == core::EngineId::TensorRt && here.tensorrt;
  const auto found = std::ranges::find(here.downloads, row.offer.id, &EngineDownload::id);
  const EngineDownload* download = found == here.downloads.end() ? nullptr : &*found;
  if (built || (download != nullptr && download->installed)) {
    row.usable = true;
    row.status = i18n::tr("installed");
  } else if (download != nullptr) {
    row.license = download->license;
    row.bytes = download->bytes;
    const bool fits = download->needed == 0 || download->available >= download->needed;
    row.usable = fits;
    row.download = fits;
    // Avant tout téléchargement : ce qui sera tiré, la place que l'installation demande en tout, et celle qui
    // reste.
    row.status =
        download->needed == 0
            ? i18n::tr_format("to download ({})", format_size(download->bytes))
            : i18n::tr_format("to download ({}) · needs {} of space, {} free", format_size(download->bytes),
                              format_size(download->needed), format_size(download->available));
    if (!fits)
      row.status = i18n::tr_format("not enough disk space: {} needed ({} to download), {} left",
                                   format_size(download->needed), format_size(download->bytes),
                                   format_size(download->available));
  } else {
    row.status = i18n::tr("not published for this version yet");
  }
}

// Les préférés qui découlent des moteurs choisis : le plus rapide pour les réseaux.
void prefer_chosen(core::Settings& settings) {
  core::EngineId networks = core::EngineId::Vulkan;
  for (const core::EngineId id : {core::EngineId::Metal, core::EngineId::OpenVino, core::EngineId::MiGraphX,
                                  core::EngineId::TensorRt}) {
    if (chosen(settings, id)) networks = id;
  }
  const auto usage = [](core::Usage u) { return static_cast<std::size_t>(u); };
  settings.preferred[usage(core::Usage::Upscale)] = networks;
  settings.preferred[usage(core::Usage::Interpolate)] = networks;
}

}  // namespace

bool chosen(const core::Settings& settings, core::EngineId id) {
  return std::ranges::contains(settings.chosen_engines, id);
}

std::vector<std::filesystem::path> backends_within(std::span<const std::filesystem::path> candidates,
                                                   const core::Settings& engines) {
  // Chaque bibliothèque d'un moteur facultatif ne s'essaie que choisie ; Vulkan, toujours.
  std::vector<std::filesystem::path> out;
  for (const std::filesystem::path& path : candidates) {
    const std::optional<core::EngineId> id = engine_of_library(path);
    if (!id || *id == core::EngineId::Vulkan || chosen(engines, *id)) out.push_back(path);
  }
  return out;
}

namespace {}  // namespace

bool amd_matrix_cores(std::string_view gpu_name) {
  const std::string name = core::ascii_lower(gpu_name);
  // RADV nomme la puce : NAVI3x pour RDNA3, GFX1200 et GFX1201 pour RDNA4 (Mesa ne leur donne
  // pas de nom de code).
  constexpr std::array kWmmaChips{"navi31", "navi32", "navi33", "gfx11", "gfx12"};
  if (name.contains("(radv ")) {
    return std::ranges::any_of(kWmmaChips, [&](std::string_view chip) { return name.contains(chip); });
  }
  if (name.contains("pro w7") || name.contains("r9700")) return true;
  // Le pilote AMD ne donne que le produit : toutes les RX 7000 (portables comprises) et 9000.
  for (std::size_t at = name.find("rx "); at != std::string::npos; at = name.find("rx ", at + 1)) {
    const std::string_view model = std::string_view(name).substr(at + 3);
    if (model.size() >= 4 && (model[0] == '7' || model[0] == '9') &&
        std::isdigit(static_cast<unsigned char>(model[1])))
      return true;
  }
  return false;
}

core::SystemInfo system_of(const GpuFacts& gpu, core::Os os) {
  return core::SystemInfo{.os = os,
                          .gpus = {core::GpuInfo{.vendor = gpu.vendor,
                                                 .name = gpu.name,
                                                 .vram_bytes = gpu.vram_bytes,
                                                 .tensor_cores = gpu.tensor_cores}}};
}

std::string rebar_text(const GpuFacts& gpu) {
  constexpr std::uint64_t kDedicated = 3 * kGiB;  // en dessous, mémoire partagée ou carte sans intérêt
  if ((gpu.vendor != GpuVendor::Nvidia && gpu.vendor != GpuVendor::Amd) || gpu.vram_bytes < kDedicated ||
      gpu.host_visible_vram_bytes == 0)
    return {};
  const bool wide = gpu.host_visible_vram_bytes >= gpu.vram_bytes / 4 * 3;
  const double size = static_cast<double>(gpu.host_visible_vram_bytes);
  const std::string text = gpu.host_visible_vram_bytes >= kGiB
                               ? std::format("{} Go", std::lround(size / static_cast<double>(kGiB)))
                               : std::format("{} Mo", std::lround(size / (1024.0 * 1024.0)));
  return std::format("{}, {}", wide ? "actif" : "inactif", text);
}

std::string gpu_line(const GpuFacts& gpu) {
  std::string line = gpu.name.empty() ? std::string(i18n::tr("Unknown GPU")) : gpu.name;
  if (gpu.vram_bytes > 0) {
    line += std::format(" · {} Go", static_cast<int>(std::lround(static_cast<double>(gpu.vram_bytes) /
                                                                 static_cast<double>(kGiB))));
  }
  if (const std::string rebar = rebar_text(gpu); !rebar.empty()) line += " · Resizable BAR " + rebar;
  if (gpu.tensor_cores) line += i18n::tr(" · Tensor cores");
  if (gpu.wmma) line += i18n::tr(" · matrix units (WMMA)");
  return line;
}

std::optional<std::pair<int, int>> nvidia_driver(const GpuFacts& gpu) {
  if (gpu.vendor != GpuVendor::Nvidia || gpu.driver.empty()) return std::nullopt;
  std::uint64_t raw = 0;
  for (const char c : gpu.driver) {
    if (c < '0' || c > '9') return std::nullopt;
    raw = (raw * 10) + static_cast<std::uint64_t>(c - '0');
    if (raw > 0xffffffffULL) return std::nullopt;
  }
  // NVIDIA code driverVersion en 10.8.8.6 bits : 580.82 vaut (580 << 22) | (82 << 14).
  return std::pair{static_cast<int>(raw >> 22U), static_cast<int>((raw >> 14U) & 0xffU)};
}

FirstRunView first_run_view(const GpuFacts& gpu, const EnginesHere& here, const core::Settings& current) {
  FirstRunView view{.gpu = gpu_line(gpu)};
  for (const core::EngineOffer& offer : core::recommend_engines(system_of(gpu, here.os))) {
    EngineRow row{.offer = offer};
    switch (offer.id) {
      case core::EngineId::Vulkan:
        row.usable = true;
        row.status = i18n::tr("bundled with the player");
        break;
      case core::EngineId::TensorRt:
      case core::EngineId::MiGraphX:
      case core::EngineId::OpenVino:
        downloadable_row(row, here);
        // P3.6 : le pilote se lit avant 650 Mo de téléchargement ; trop ancien, la case reste verrouillée.
        if (offer.id == core::EngineId::TensorRt && row.download) {
          if (const auto version = nvidia_driver(gpu); version && version->first < kTensorRtDriverMajor) {
            row.usable = false;
            row.download = false;
            row.status = i18n::tr_format("NVIDIA driver {} or newer required, you have {}.{:02}",
                                         kTensorRtDriverMajor, version->first, version->second);
          }
        }
        break;
      case core::EngineId::Metal:  // Core ML, dans AJP.app
        row.usable = here.coreml;
        row.status = here.coreml ? i18n::tr("bundled with the player") : i18n::tr("not in this package");
        break;
      default:
        row.status = i18n::tr("cannot be installed in this version yet");
        break;
    }
    const bool wanted = current.first_run_done ? chosen(current, offer.id) : offer.preselected;
    // Une licence ne se coche jamais d'office : le moteur qui en demande une attend l'accord.
    row.chosen = offer.id == core::EngineId::Vulkan || (wanted && row.usable && row.license.empty());
    view.engines.push_back(std::move(row));
  }
  return view;
}

core::Settings settings_after(const FirstRunView& view, core::Settings settings) {
  settings.first_run_done = true;
  settings.chosen_engines.clear();
  settings.installed_engines.clear();
  for (const EngineRow& row : view.engines) {
    // Un moteur à télécharger compte comme installé une fois coché (install_chosen le télécharge,
    // without_engine le retire s'il échoue) ; avec sa licence acceptée seulement.
    const bool taken = row.usable && (row.chosen || row.offer.id == core::EngineId::Vulkan) &&
                       (!row.download || row.license.empty() || row.license_accepted);
    if (row.usable && (!row.download || taken)) settings.installed_engines.push_back(row.offer.id);
    if (taken) settings.chosen_engines.push_back(row.offer.id);
  }
  prefer_chosen(settings);
  return settings;
}

core::Settings without_engine(core::Settings settings, core::EngineId id) {
  std::erase(settings.chosen_engines, id);
  std::erase(settings.installed_engines, id);
  prefer_chosen(settings);
  return settings;
}

PlaySettings within_engines(PlaySettings settings, const core::Settings& engines,
                            std::vector<std::string>& notes) {
  const bool inference = std::ranges::any_of(
      std::array{core::EngineId::TensorRt, core::EngineId::Vulkan, core::EngineId::MiGraphX,
                 core::EngineId::OpenVino, core::EngineId::Metal},
      [&](core::EngineId id) { return chosen(engines, id); });
  if (!inference && (settings.upscale || settings.rife_factor >= 2)) {
    settings.upscale = false;
    settings.rife_factor = 0;
    settings.rife_factor_den = 1;
    notes.emplace_back(i18n::tr("neither TensorRT nor Vulkan chosen: no SPAN or RIFE"));
  }
  return settings;
}

std::string chosen_line(const core::Settings& engines) {
  std::string line;
  for (const core::EngineId id : engines.chosen_engines)
    line += line.empty() ? std::string(engine_label(id)) : ", " + std::string(engine_label(id));
  return line;
}

std::string_view engine_label(core::EngineId id) {
  switch (id) {
    case core::EngineId::Vulkan:
      return "Vulkan";
    case core::EngineId::TensorRt:
      return "TensorRT";
    case core::EngineId::MiGraphX:
      return core::kHostOs == core::Os::Windows ? "MIGraphX (Windows ML)" : "MIGraphX (ROCm)";
    case core::EngineId::OpenVino:
      return "OpenVINO";
    case core::EngineId::DirectMl:
      return "DirectML";
    case core::EngineId::Metal:
      return "Core ML (Apple)";
  }
  return "?";
}

std::string_view badge_label(core::Badge badge) {
  switch (badge) {
    case core::Badge::Recommended:
      return i18n::tr("Recommended");
    case core::Badge::Compatibility:
      return i18n::tr("Compatibility");
    case core::Badge::NvidiaOnly:
      return "NVIDIA";
    case core::Badge::AmdOnly:
      return "AMD";
    case core::Badge::IntelOnly:
      return "Intel";
    case core::Badge::AppleOnly:
      return "Apple";
    case core::Badge::WindowsOnly:
      return "Windows";
    case core::Badge::LinuxOnly:
      return "Linux";
  }
  return "?";
}

std::string_view reason_label(core::Reason reason) {
  switch (reason) {
    case core::Reason::AlwaysIncluded:
      return i18n::tr("Always available");
    case core::Reason::FastestForThisGpu:
      return i18n::tr("Fastest for this GPU");
    case core::Reason::CompareWithVulkan:
      return i18n::tr("Compare with Vulkan");
    case core::Reason::NoMatchingGpu:
      return i18n::tr("Does not suit this GPU");
  }
  return "?";
}

std::vector<std::string> engines_summary(const std::vector<EngineRow>& rows) {
  std::string kept;
  std::uint64_t bytes = 0;
  std::vector<std::string> refused;
  for (const EngineRow& row : rows) {
    const std::string_view label = engine_label(row.offer.id);
    // Le moteur recommandé pour cette carte reste décoché (sa licence ne se coche jamais d'office) : la
    // calibration mesurerait l'autre sans le dire (A25-02).
    if (row.usable && !row.chosen && std::ranges::contains(row.offer.badges, core::Badge::Recommended) &&
        row.offer.id != core::EngineId::Vulkan)
      refused.push_back(i18n::tr_format(
          "{}: recommended for this card but not selected. The calibration will measure a slower engine.",
          label));
    if (!row.chosen || !row.usable) continue;
    if (row.download && !row.license.empty() && !row.license_accepted) {
      refused.push_back(i18n::tr_format("{}: licence not accepted, it will not be installed", label));
      continue;
    }
    kept += (kept.empty() ? "" : ", ") + std::string(label);
    if (row.download) bytes += row.bytes;
  }
  std::vector<std::string> out;
  out.push_back(bytes > 0 ? i18n::tr_format("Engines: {} ({} to download)", kept, format_size(bytes))
                          : i18n::tr_format("Engines: {}", kept));
  out.insert(out.end(), refused.begin(), refused.end());
  return out;
}

namespace {

void radio_note(const char* label, const char* detail, bool selected, bool& pick, bool enabled = true) {
  ImGui::BeginDisabled(!enabled);
  if (ImGui::RadioButton(label, selected)) pick = true;
  ImGui::EndDisabled();
  if (!enabled)
    panel::tip(detail);
  else
    note(detail);
}

// Le contenu d'un écran de l'assistant.
void draw_wizard_step(FirstRunView& view, Wizard& wizard) {
  switch (current(wizard)) {
    case WizardStep::Welcome:
      section(i18n::tr("Welcome"));
      panel::wrapped(view.gpu.c_str());
      note(view.tier_line);
      if (!view.display_line.empty()) note(view.display_line);
      note(i18n::tr(
          "The player picks the settings for this machine; everything is in Settings (p) afterwards."));
      {
        // La langue de l'interface (P3.18), détectée au lancement, confirmée ici ; prise en direct.
        const bool french = i18n::language() == i18n::Language::French;
        if (ImGui::BeginCombo(i18n::tr("Interface language"), french ? "Français" : "English")) {  // i18n-ok
          if (ImGui::Selectable("English", !french)) wizard.interface_language = "en";             // i18n-ok
          if (ImGui::Selectable("Français", french)) wizard.interface_language = "fr";             // i18n-ok
          ImGui::EndCombo();
        }
        if (!wizard.interface_language.empty())
          i18n::set_language(i18n::language_from_code(wizard.interface_language));
      }
      break;
    case WizardStep::Engines:
      section(i18n::tr("Engines"));
      for (EngineRow& row : view.engines) engine_row(row);
      note(
          i18n::tr("Vulkan works on every GPU. Checked engines install at the end of the assistant, verified "
                   "by hash; "
                   "Vulkan takes over if one fails."));
      break;
    case WizardStep::Languages:
      section(i18n::tr("Languages"));
      language_choice(i18n::tr("Audio language"), i18n::tr("From the file"), wizard.audio_language);
      language_choice(i18n::tr("Subtitle language"), i18n::tr("None"), wizard.subtitle_language);
      break;
    case WizardStep::Updates: {
      section(i18n::tr("Updates"));
      bool pick = false;
      radio_note(i18n::tr("Notify me"), i18n::tr("A notification; you start the update yourself."),
                 wizard.update_mode == UpdateMode::Notify, pick);
      if (pick) wizard.update_mode = UpdateMode::Notify;
      pick = false;
      radio_note(i18n::tr("Automatic"), i18n::tr("Downloaded, installed at next launch."),
                 wizard.update_mode == UpdateMode::Auto, pick);
      if (pick) wizard.update_mode = UpdateMode::Auto;
      pick = false;
      radio_note(i18n::tr("Never"), i18n::tr("No checks."), wizard.update_mode == UpdateMode::Never, pick);
      if (pick) wizard.update_mode = UpdateMode::Never;
      section(i18n::tr("Default player"));
      ImGui::Checkbox(i18n::tr("Make it the default video player"), &wizard.make_default);
      note(i18n::tr("AJP is added to \"Open with\" in any case."));
      break;
    }
    case WizardStep::Summary:
      section(i18n::tr("Ready"));
      for (const std::string& line : engines_summary(view.engines)) ImGui::BulletText("%s", line.c_str());
      for (const std::string& line : wizard_summary(wizard)) ImGui::BulletText("%s", line.c_str());
      section(i18n::tr("Machine calibration"));
      note(
          i18n::tr("\"Calibrate now\" tests several settings on a very demanding sequence: picture "
                   "resolution, SPAN upscaling and RIFE smoothness, and keeps the best combination this "
                   "machine sustains. Only a progress bar is shown; it takes a few minutes, longer the first "
                   "time (the engines are built once)."));
      // Calculée au moment de l'afficher, d'après les moteurs cochés maintenant : TensorRT choisi à l'écran
      // des moteurs construit ses moteurs pendant l'essai (A25-03).
      note(i18n::tr_format("Estimated for this machine: about {} minutes.",
                           calibration_minutes(std::ranges::any_of(view.engines,
                                                                   [](const EngineRow& row) {
                                                                     return row.offer.id ==
                                                                                core::EngineId::TensorRt &&
                                                                            row.chosen && row.usable;
                                                                   }),
                                               view.laptop)));
      note(
          i18n::tr("\"Set it up myself\": the Auto preset estimated for this card, changeable in Video > "
                   "Post-processing."));
      note(i18n::tr("Everything can be changed later in Settings (p). To redo this: ajp --first-run."));
      break;
  }
}

void draw_wizard(FirstRunView& view, float scale) {
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 24.0F * scale;
  const ImVec2 size{std::min(860.0F * scale, io.DisplaySize.x - 2.0F * pad),
                    std::min(640.0F * scale, io.DisplaySize.y - 2.0F * pad)};
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  Wizard& wizard = view.wizard;
  if (ImGui::Begin(i18n::tr("Welcome"), nullptr, kFlags)) {
    note(progress_text(wizard));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kAccent);  // la couleur d'accent, pas le jaune par défaut
    ImGui::ProgressBar(static_cast<float>(wizard.at + 1) / static_cast<float>(wizard.steps.size()),
                       {-1.0F, 6.0F * scale}, "");
    ImGui::PopStyleColor();
    const float footer = ImGui::GetFrameHeight() * 1.6F + ImGui::GetStyle().ItemSpacing.y * 3.0F;
    ImGui::BeginChild("##ecran", {0.0F, -footer});
    draw_wizard_step(view, wizard);
    ImGui::EndChild();
    ImGui::Separator();
    const float tall = ImGui::GetFrameHeight() * 1.4F;
    ImGui::BeginDisabled(is_first(wizard));
    if (ImGui::Button(i18n::tr("Back"), {0.0F, tall})) previous(wizard);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (!is_last(wizard)) {
      if (ImGui::Button(i18n::tr("Skip"), {0.0F, tall})) skip(wizard);
      ImGui::SameLine();
    }
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.05F, 0.07F, 0.12F, 1.0F});
    if (is_last(wizard)) {
      if (ImGui::Button(i18n::tr("Calibrate now"), {0.0F, tall})) {
        wizard.calibrate = true;
        view.done = true;
      }
    } else if (ImGui::Button(i18n::tr("Next"), {0.0F, tall})) {
      next(wizard);
    }
    ImGui::PopStyleColor(2);
    // Le bouton secondaire de Prêt : sans calibration, le niveau Auto estimé pour cette carte (P3.1).
    if (is_last(wizard)) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Set it up myself"), {0.0F, tall})) {
        set_up_myself(wizard);
        view.done = true;
      }
    }
  }
  ImGui::End();
}

}  // namespace

void draw_first_run(FirstRunView& view, float scale) {
  if (!view.revisit && !view.wizard.steps.empty()) {
    draw_wizard(view, scale);
    return;
  }
  // Le bouton qui lance les téléchargements porte le nom du bouton de cet écran.
  const std::string footer_text = i18n::tr_format(
      "Vulkan works on every GPU. Checked engines install after \"{}\", verified by hash; Vulkan takes over "
      "if one fails.",
      view.revisit ? i18n::tr("Resume playback") : i18n::tr("Start"));
  const char* const kFooter = footer_text.c_str();
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 24.0F * scale;
  const ImVec2 size{std::min(1080.0F * scale, io.DisplaySize.x - 2.0F * pad),
                    std::min(760.0F * scale, io.DisplaySize.y - 2.0F * pad)};
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin(view.revisit ? i18n::tr("Engines") : i18n::tr("Welcome"), nullptr, kFlags)) {
    ImGui::TextUnformatted(view.gpu.c_str());
    note(view.revisit ? i18n::tr("Checked engines install, then playback resumes at the same spot.")
                      : i18n::tr("The player chose for this GPU; change anything later in Settings (p)."));
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 room = ImGui::GetContentRegionAvail();
    const float footer = ImGui::CalcTextSize(kFooter, nullptr, false, room.x).y +
                         ImGui::GetFrameHeight() * 1.4F + style.ItemSpacing.y * 4.0F;
    const float body = std::max(room.y - footer, ImGui::GetFrameHeight() * 4.0F);
    ImGui::BeginChild("##moteurs", {0.0F, body});
    section(i18n::tr("Engines"));
    for (EngineRow& row : view.engines) engine_row(row);
    ImGui::EndChild();
    ImGui::Separator();
    note(kFooter);
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.05F, 0.07F, 0.12F, 1.0F});
    const char* start = view.revisit ? i18n::tr("Resume playback") : i18n::tr("Start");
    const float tall = ImGui::GetFrameHeight() * 1.4F;
    if (ImGui::Button(start, {ImGui::CalcTextSize(start).x + style.FramePadding.x * 6.0F, tall}))
      view.done = true;
    ImGui::PopStyleColor(2);
    if (view.revisit) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Cancel"), {0.0F, tall})) view.cancelled = true;
    }
  }
  ImGui::End();
}

}  // namespace ajp::app::play
