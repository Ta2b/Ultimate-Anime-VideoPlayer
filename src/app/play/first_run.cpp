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

namespace ajp::app::play {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
// 11 Go et plus : la chaîne 1440p ×3 et ses moteurs tiennent avec la marge du bureau.
constexpr std::uint64_t kQualityVram = 11 * kGiB;
// 4 Go annoncés (un peu moins une fois le pilote servi) : le profil 7, 540p ×2 et RIFE ×2.
constexpr std::uint64_t kLightVram = 3 * kGiB + kGiB / 2;

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
    ImGui::Checkbox("J'accepte sa licence", &row.license_accepted);
    ImGui::SameLine();
    ImGui::TextLinkOpenURL("Lire la licence", row.license.c_str());
  }
  if (row.download && row.chosen && row.needs_key) {
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 22.0F);
    ImGui::InputText("Clé d'API NGC", row.key.data(), row.key.size(), ImGuiInputTextFlags_Password);
    ImGui::SameLine();
    ImGui::TextLinkOpenURL("Obtenir une clé", "https://org.ngc.nvidia.com/setup/api-keys");
  }
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
    row.status = "installé" + missing;
  } else if (download != nullptr) {
    row.usable = true;
    row.download = true;
    row.license = download->license;
    row.status = std::format("à télécharger ({} Mo){}", (download->bytes + (1U << 20) - 1) >> 20, missing);
  } else {
    row.status = "pas encore publié pour cette version";
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
  settings.preferred[usage(core::Usage::TrueHdr)] =
      chosen(settings, core::EngineId::MaxineVfx) ? std::optional(core::EngineId::MaxineVfx) : std::nullopt;
}

}  // namespace

bool chosen(const core::Settings& settings, core::EngineId id) {
  return std::ranges::contains(settings.chosen_engines, id);
}

std::vector<std::filesystem::path> backends_within(std::span<const std::filesystem::path> candidates,
                                                   const core::Settings& engines) {
  // Chaque bibliothèque d'un moteur facultatif ne s'essaie que choisie ; Vulkan, toujours.
  constexpr std::array<std::pair<std::string_view, core::EngineId>, 4> kOptional{{
      {"tensorrt", core::EngineId::TensorRt},
      {"migraphx", core::EngineId::MiGraphX},
      {"openvino", core::EngineId::OpenVino},
      {"coreml", core::EngineId::Metal},  // le moteur natif du Mac : Core ML derrière MoltenVK
  }};
  std::vector<std::filesystem::path> out;
  for (const std::filesystem::path& path : candidates) {
    const std::string file = path.filename().string();
    const bool allowed = std::ranges::all_of(kOptional, [&](const auto& engine) {
      return !file.contains(engine.first) || chosen(engines, engine.second);
    });
    if (allowed) out.push_back(path);
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
  constexpr std::array kListedChips{"navi21",  "navi31",  "navi32",  "navi33",  "navi44", "navi48",
                                    "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"};
  if (name.contains("(radv ")) {
    return std::ranges::any_of(kListedChips, [&](std::string_view chip) { return name.contains(chip); });
  }
  if (name.contains("instinct") || name.contains("pro w6800") || name.contains("pro w7")) return true;
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

core::SystemInfo system_of(const GpuFacts& gpu) {
#if defined(_WIN32)
  constexpr core::Os kOs = core::Os::Windows;
#elif defined(__APPLE__)
  constexpr core::Os kOs = core::Os::MacOs;
#else
  constexpr core::Os kOs = core::Os::Linux;
#endif
  // OpenVINO refuse lui-même un GPU qu'il ne sert pas (repli sur Vulkan) : le fabricant suffit.
  // MIGraphX aussi, mais après un téléchargement de ROCm : la liste officielle le filtre avant.
  return core::SystemInfo{
      .os = kOs,
      .gpus = {core::GpuInfo{.vendor = gpu.vendor,
                             .name = gpu.name,
                             .vram_bytes = gpu.vram_bytes,
                             .tensor_cores = gpu.tensor_cores,
                             .rocm_supported = gpu.vendor == GpuVendor::Amd && rocm_listed(gpu.name),
                             .openvino_gpu = gpu.vendor == GpuVendor::Intel}}};
}

std::string gpu_line(const GpuFacts& gpu) {
  std::string line = gpu.name.empty() ? std::string("GPU inconnu") : gpu.name;
  if (gpu.vram_bytes > 0) {
    line += std::format(" · {} Go", static_cast<int>(std::lround(static_cast<double>(gpu.vram_bytes) /
                                                                 static_cast<double>(kGiB))));
  }
  if (gpu.tensor_cores) line += " · cœurs Tensor";
  if (gpu.wmma) line += " · unités matricielles (WMMA)";
  return line;
}

int start_profile_for(const GpuFacts& gpu) {
  if (gpu.vendor == GpuVendor::Nvidia && gpu.tensor_cores) return gpu.vram_bytes >= kQualityVram ? 1 : 7;
  // Une RDNA3 ou RDNA4 de 12 Go et plus (RX 7900, RX 9070 XT) : matrices coopératives du moteur
  // Vulkan ou MIGraphX, au niveau d'une RTX 4070 Ti.
  if (gpu.vendor == GpuVendor::Amd && gpu.wmma && gpu.vram_bytes >= kQualityVram) return 1;
  // Sans cœurs Tensor, le moteur Vulkan tient 540p ×2 sur une carte dédiée de 4 Go (RX 6500) ;
  // un GPU Intel (intégré le plus souvent) ou plus petit reste sur la source.
  return gpu.vendor != GpuVendor::Intel && gpu.vram_bytes >= kLightVram ? 7 : 9;
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
      case core::EngineId::MiGraphX:
      case core::EngineId::OpenVino:
        downloadable_row(row, here);
        break;
      case core::EngineId::Metal:  // Core ML, dans AJP.app
        row.usable = here.coreml;
        row.status = here.coreml ? "livré avec le lecteur" : "absent de ce paquet";
        break;
      case core::EngineId::MaxineVfx:
        row.usable = here.maxine || !here.maxine_license.empty();
        row.status = here.maxine_status;
        if (!here.maxine && !here.maxine_license.empty()) {
          // Le SDK ne se redistribue pas : le lecteur le télécharge depuis NGC avec la clé de
          // l'utilisateur, une fois sa licence acceptée.
          row.download = true;
          row.needs_key = true;
          row.license = here.maxine_license;
          row.status = "à installer depuis NVIDIA NGC (clé d'API d'un compte NVIDIA gratuit)";
        }
        break;
      default:
        row.status = "pas encore installable dans cette version";
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
                       (!row.download || row.license.empty() || row.license_accepted) &&
                       (!row.needs_key || row.key.front() != '\0');
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
    notes.emplace_back("ni TensorRT ni Vulkan choisis : ni SPAN ni RIFE");
  }
  if (!chosen(engines, core::EngineId::MaxineVfx) && settings.true_hdr) {
    settings.true_hdr = false;
    settings.hdr_native = true;
    notes.emplace_back("NVIDIA Maxine non choisi : HDR adaptatif (tous GPU) au lieu de RTX HDR");
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
      return "Core ML (Apple)";
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
      "Vulkan sert tous les GPU ; les moteurs cochés à télécharger s'installent après « Commencer », "
      "vérifiés par leur empreinte, et Vulkan prend le relais de celui qui échoue.";
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
