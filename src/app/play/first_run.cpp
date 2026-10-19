#include "app/play/first_run.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <utility>

#include <imgui.h>

namespace ajp::app::play {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
// 11 Go et plus : la chaîne 1440p ×3 et ses moteurs tiennent avec la marge du bureau.
constexpr std::uint64_t kQualityVram = 11 * kGiB;

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
  ImGui::Unindent(ImGui::GetFrameHeightWithSpacing());
  ImGui::Spacing();
  ImGui::PopID();
}

void profile_choice(ProfilesView& view) {
  for (int slot = 1; slot <= 9; ++slot) {
    const std::optional<ProfileRow>& row = view.rows[static_cast<std::size_t>(slot - 1)];
    if (!row) continue;
    ImGui::PushID(slot);
    ImGui::AlignTextToFramePadding();
    if (ImGui::RadioButton(std::format("{}  {}", slot, row->name).c_str(), view.start == slot))
      view.start = slot;
    ImGui::Indent(ImGui::GetFrameHeightWithSpacing());
    note(row->summary);
    ImGui::Unindent(ImGui::GetFrameHeightWithSpacing());
    ImGui::PopID();
  }
}

bool chosen(const core::Settings& settings, core::EngineId id) {
  return std::ranges::contains(settings.chosen_engines, id);
}

}  // namespace

core::SystemInfo system_of(const GpuFacts& gpu) {
  return core::SystemInfo{.os = core::Os::Linux,
                          .gpus = {core::GpuInfo{.vendor = gpu.vendor,
                                                 .name = gpu.name,
                                                 .vram_bytes = gpu.vram_bytes,
                                                 .tensor_cores = gpu.tensor_cores}}};
}

std::string gpu_line(const GpuFacts& gpu) {
  std::string line = gpu.name.empty() ? std::string("GPU inconnu") : gpu.name;
  if (gpu.vram_bytes > 0) {
    line += std::format(" · {} Go", static_cast<int>(std::lround(static_cast<double>(gpu.vram_bytes) /
                                                                 static_cast<double>(kGiB))));
  }
  if (gpu.tensor_cores) line += " · cœurs Tensor";
  return line;
}

int start_profile_for(const GpuFacts& gpu) {
  if (gpu.vendor != GpuVendor::Nvidia || !gpu.tensor_cores) return 9;
  return gpu.vram_bytes >= kQualityVram ? 1 : 7;
}

FirstRunView first_run_view(const GpuFacts& gpu, const EnginesHere& here, ProfilesView profiles,
                            const core::Settings& current) {
  FirstRunView view{.gpu = gpu_line(gpu), .profiles = std::move(profiles)};
  for (const core::EngineOffer& offer : core::recommend_engines(system_of(gpu))) {
    EngineRow row{.offer = offer};
    switch (offer.id) {
      case core::EngineId::Vulkan:
        row.usable = true;
        row.status = "livré avec le lecteur";
        break;
      case core::EngineId::TensorRt:
        row.usable = here.tensorrt;
        row.status = here.tensorrt ? "installé" : "absent de ce poste";
        break;
      case core::EngineId::MaxineVfx:
        row.usable = here.maxine;
        row.status = here.maxine_status;
        break;
      default:
        row.status = "pas encore installable dans cette version";
        break;
    }
    const bool wanted = current.first_run_done ? chosen(current, offer.id) : offer.preselected;
    row.chosen = offer.id == core::EngineId::Vulkan || (wanted && row.usable);
    view.engines.push_back(std::move(row));
  }
  return view;
}

core::Settings settings_after(const FirstRunView& view, core::Settings settings) {
  settings.first_run_done = true;
  settings.chosen_engines.clear();
  settings.installed_engines.clear();
  for (const EngineRow& row : view.engines) {
    if (row.usable) settings.installed_engines.push_back(row.offer.id);
    if (row.usable && (row.chosen || row.offer.id == core::EngineId::Vulkan))
      settings.chosen_engines.push_back(row.offer.id);
  }
  const bool tensorrt = chosen(settings, core::EngineId::TensorRt);
  const auto usage = [](core::Usage u) { return static_cast<std::size_t>(u); };
  settings.preferred[usage(core::Usage::Upscale)] =
      tensorrt ? core::EngineId::TensorRt : core::EngineId::Vulkan;
  settings.preferred[usage(core::Usage::Interpolate)] =
      tensorrt ? core::EngineId::TensorRt : core::EngineId::Vulkan;
  settings.preferred[usage(core::Usage::TrueHdr)] =
      chosen(settings, core::EngineId::MaxineVfx) ? std::optional(core::EngineId::MaxineVfx) : std::nullopt;
  return settings;
}

PlaySettings within_engines(PlaySettings settings, const core::Settings& engines,
                            std::vector<std::string>& notes) {
  if (!chosen(engines, core::EngineId::TensorRt) && (settings.upscale || settings.rife_factor >= 2)) {
    settings.upscale = false;
    settings.rife_factor = 0;
    notes.emplace_back("TensorRT non choisi : ni SPAN ni RIFE (pas encore d'inférence Vulkan ici)");
  }
  if (!chosen(engines, core::EngineId::MaxineVfx) && settings.true_hdr) {
    settings.true_hdr = false;
    notes.emplace_back("NVIDIA Maxine non choisi : pas de RTX HDR");
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
    case core::EngineId::RtxVideoSdk:
      return "RTX Video SDK";
    case core::EngineId::MaxineVfx:
      return "NVIDIA Maxine (RTX HDR)";
    case core::EngineId::MiGraphX:
      return "MIGraphX (ROCm)";
    case core::EngineId::OpenVino:
      return "OpenVINO";
    case core::EngineId::DirectMl:
      return "DirectML";
    case core::EngineId::Metal:
      return "Metal";
  }
  return "?";
}

std::string_view badge_label(core::Badge badge) {
  switch (badge) {
    case core::Badge::Recommended:
      return "Recommandé";
    case core::Badge::Compatibility:
      return "Compatibilité";
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
      return "Toujours présent";
    case core::Reason::FastestForThisGpu:
      return "Le plus rapide pour ce GPU";
    case core::Reason::RtxHdrForThisGpu:
      return "RTX HDR sur ce GPU";
    case core::Reason::CompareWithVulkan:
      return "À comparer avec Vulkan";
    case core::Reason::NoMatchingGpu:
      return "Ne convient pas à ce GPU";
  }
  return "?";
}

void draw_first_run(FirstRunView& view, float scale) {
  static constexpr const char* kFooter =
      "Dans cette version, SPAN et RIFE passent par TensorRT et RTX HDR par NVIDIA Maxine ; sans eux, "
      "l'image "
      "reste celle de la source.";
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 24.0F * scale;
  const ImVec2 size{std::min(1080.0F * scale, io.DisplaySize.x - 2.0F * pad),
                    std::min(760.0F * scale, io.DisplaySize.y - 2.0F * pad)};
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin("Bienvenue", nullptr, kFlags)) {
    ImGui::TextUnformatted(view.gpu.c_str());
    note("Le lecteur a choisi pour ce GPU ; tout se change ensuite dans les paramètres (p).");
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 room = ImGui::GetContentRegionAvail();
    const float footer = ImGui::CalcTextSize(kFooter, nullptr, false, room.x).y +
                         ImGui::GetFrameHeight() * 1.4F + style.ItemSpacing.y * 4.0F;
    const float body = std::max(room.y - footer, ImGui::GetFrameHeight() * 4.0F);
    ImGui::BeginChild("##moteurs", {(room.x - style.ItemSpacing.x) * 0.5F, body});
    section("Moteurs");
    for (EngineRow& row : view.engines) engine_row(row);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##profils", {0.0F, body});
    section("Profil de démarrage");
    profile_choice(view.profiles);
    ImGui::EndChild();
    ImGui::Separator();
    note(kFooter);
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.05F, 0.07F, 0.12F, 1.0F});
    const ImVec2 button{ImGui::CalcTextSize("Commencer").x + style.FramePadding.x * 6.0F,
                        ImGui::GetFrameHeight() * 1.4F};
    if (ImGui::Button("Commencer", button)) view.done = true;
    ImGui::PopStyleColor(2);
  }
  ImGui::End();
}

}  // namespace ajp::app::play
