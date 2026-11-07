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
#include "app/play/panel_style.h"
#include "core/engines.h"

namespace ajp::app::play {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

void note(const std::string& text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextWrapped("%s", text.c_str());
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
  }
  ImGui::Unindent(ImGui::GetFrameHeightWithSpacing());
  ImGui::Spacing();
  ImGui::PopID();
}

// Une ligne d'un moteur téléchargeable : installé ici (arbre de construction, ou extrait), à
// télécharger, ou absent de l'index de cette version.
void downloadable_row(EngineRow& row, const EnginesHere& here) {
  const bool built = (row.offer.id == core::EngineId::TensorRt && here.tensorrt) ||
                     (row.offer.id == core::EngineId::MiGraphX && here.migraphx) ||
                     (row.offer.id == core::EngineId::OpenVino && here.openvino);
  const auto found = std::ranges::find(here.downloads, row.offer.id, &EngineDownload::id);
  const EngineDownload* download = found == here.downloads.end() ? nullptr : &*found;
  const std::string missing =
      download != nullptr && !download->missing.empty() ? std::format(" · {}", download->missing) : "";
  if (built || (download != nullptr && download->installed)) {
    row.usable = true;
    row.status = i18n::tr("installed") + missing;
  } else if (download != nullptr) {
    row.license = download->license;
    const bool fits = download->needed == 0 || download->available >= download->needed;
    row.usable = fits;
    row.download = fits;
    // Avant tout téléchargement : ce qui sera tiré, la place que l'installation demande en tout, et celle qui
    // reste.
    row.status =
        download->needed == 0
            ? i18n::tr_format("to download ({}){}", format_size(download->bytes), missing)
            : i18n::tr_format("to download ({}) · needs {} of space, {} free{}", format_size(download->bytes),
                              format_size(download->needed), format_size(download->available), missing);
    if (!fits)
      row.status = i18n::tr_format("not enough disk space: {} needed ({} to download), {} left",
                                   format_size(download->needed), format_size(download->bytes),
                                   format_size(download->available)) +
                   missing;
  } else {
    row.status =
        row.offer.id == core::EngineId::TensorRt && here.os == core::Os::Linux
            ? i18n::tr("not published (NVIDIA licence): build it on this machine, see docs/nvidia-linux.md")
            : i18n::tr("not published for this version yet");
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

namespace {

std::string lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

}  // namespace

bool amd_matrix_cores(std::string_view gpu_name) {
  const std::string name = lower(gpu_name);
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

bool rocm_listed(std::string_view gpu_name) {
  const std::string name = lower(gpu_name);
  // RADV nomme la puce (« AMD Radeon RX 6500M (RADV NAVI24) ») : elle seule décide. RDNA4 y
  // porte son nom GFX (« AMD Radeon RX 9070 XT (RADV GFX1201) »).
  // Les Ryzen AI 9 HX et Ryzen AI Max (GFX1150, GFX1151) depuis ROCm 7.2.1 ; pas le Ryzen AI 7
  // (Radeon 860M, GFX1152).
  constexpr std::array kListedChips{"navi21",  "navi31",  "navi32",  "navi33",  "navi44",
                                    "navi48",  "gfx1100", "gfx1101", "gfx1102", "gfx1150",
                                    "gfx1151", "gfx1200", "gfx1201"};
  if (name.contains("(radv ")) {
    return std::ranges::any_of(kListedChips, [&](std::string_view chip) { return name.contains(chip); });
  }
  if (name.contains("instinct") || name.contains("pro w6800") || name.contains("pro w7")) return true;
  constexpr std::array kListedApus{"880m", "890m", "8050s", "8060s"};
  if (std::ranges::any_of(kListedApus, [&](std::string_view apu) { return name.contains(apu); })) return true;
  // AMDVLK et Windows ne donnent que le produit : RX 6800 à 6950 de bureau (les « M » et « S »
  // portables sont des Navi 22), toutes les RX 7000 et 9000 de bureau.
  for (std::size_t at = name.find("rx "); at != std::string::npos; at = name.find("rx ", at + 1)) {
    const std::string_view model = std::string_view(name).substr(at + 3);
    if (model.size() < 4 || !std::isdigit(static_cast<unsigned char>(model[0]))) continue;
    const bool mobile = model.size() > 4 && (model[4] == 'm' || model[4] == 's');
    if (mobile) return false;
    if (model.starts_with("68") || model.starts_with("69") || model.starts_with('7') ||
        model.starts_with('9'))
      return true;
  }
  return false;
}

bool winml_listed(std::string_view gpu_name) {
  if (amd_matrix_cores(gpu_name)) return true;  // RX 7000, RX 9000, Radeon Pro W7000, R9700
  const std::string name = lower(gpu_name);
  // Les APU RDNA 3 (Phoenix, Hawk Point), RDNA 3.5 (Strix, Krackan, Strix Halo), que le pilote
  // AMD nomme par leur produit.
  constexpr std::array kRdna3Apus{"740m", "760m", "780m",  "820m",  "840m", "860m",
                                  "880m", "890m", "8040s", "8050s", "8060s"};
  return std::ranges::any_of(kRdna3Apus, [&](std::string_view apu) { return name.contains(apu); });
}

core::SystemInfo system_of(const GpuFacts& gpu, core::Os os) {
  // OpenVINO refuse lui-même un GPU qu'il ne sert pas (repli sur Vulkan) : le fabricant suffit.
  // MIGraphX aussi, mais après un téléchargement de ROCm : la liste officielle le filtre avant.
  return core::SystemInfo{
      .os = os,
      .gpus = {core::GpuInfo{.vendor = gpu.vendor,
                             .name = gpu.name,
                             .vram_bytes = gpu.vram_bytes,
                             .tensor_cores = gpu.tensor_cores,
                             .rocm_supported = gpu.vendor == GpuVendor::Amd && rocm_listed(gpu.name),
                             .winml_migraphx = gpu.vendor == GpuVendor::Amd && winml_listed(gpu.name),
                             .openvino_gpu = gpu.vendor == GpuVendor::Intel}}};
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
      ImGui::TextWrapped("%s", view.gpu.c_str());
      note(view.tier_line);
      if (!view.display_line.empty()) note(view.display_line);
      note(i18n::tr(
          "The player picks the settings for this machine; everything is in Settings (p) afterwards."));
      break;
    case WizardStep::Engines:
      section(i18n::tr("Engines"));
      for (EngineRow& row : view.engines) engine_row(row);
      note(
          i18n::tr("Vulkan works on every GPU. Checked engines install after \"Finish\", verified by hash; "
                   "Vulkan takes over if one fails."));
      break;
    case WizardStep::Chain:
      section(i18n::tr("Machine calibration"));
      note(
          i18n::tr("After \"Finish\", the player tests several settings on a very demanding sequence: "
                   "picture resolution, SPAN upscaling and RIFE smoothness. It keeps the best combination "
                   "this machine sustains, SPAN first."));
      note(
          i18n::tr("Only a progress bar is shown; it takes a few minutes, longer the first time on a "
                   "laptop (the engines are built once)."));
      note(i18n::tr("Skipped: the Auto preset. Redo it later in Video > Post-processing."));
      break;
    case WizardStep::Hdr:    // retiré : le HDR suit le système
    case WizardStep::Grade:  // retiré
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
                 wizard.update_mode == "notify", pick);
      if (pick) wizard.update_mode = "notify";
      pick = false;
      radio_note(i18n::tr("Automatic"), i18n::tr("Downloaded, installed at next launch."),
                 wizard.update_mode == "auto", pick);
      if (pick) wizard.update_mode = "auto";
      pick = false;
      radio_note(i18n::tr("Never"), i18n::tr("No checks."), wizard.update_mode == "never", pick);
      if (pick) wizard.update_mode = "never";
      break;
    }
    case WizardStep::Summary:
      section(i18n::tr("Summary"));
      for (const std::string& line : wizard_summary(wizard)) ImGui::BulletText("%s", line.c_str());
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
      if (ImGui::Button(i18n::tr("Finish"), {0.0F, tall})) view.done = true;
    } else if (ImGui::Button(i18n::tr("Next"), {0.0F, tall})) {
      next(wizard);
    }
    ImGui::PopStyleColor(2);
    // Le bouton secondaire du résumé : sans calibration, le niveau Auto estimé pour cette carte (P3.1).
    if (is_last(wizard) && wants_machine_calibration(wizard)) {
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
