#include "app/play/model_choice.h"

#include <algorithm>
#include <string_view>
#include <system_error>
#include <utility>

#include "core/dir_entries.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

// Le défaut trouvé, la liste de son dossier (fichiers qui commencent par prefix), et le fichier
// choisi au menu s'il existe à côté.
ModelChoice choose_in(const std::filesystem::path& found, std::string_view prefix,
                      const std::string& chosen_name, bool option_given) {
  ModelChoice choice{.path = found, .default_name = found.filename().string()};
  std::error_code error;
  for (const auto& entry : core::dir_entries(found.parent_path(), error)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with(prefix) && name.ends_with(".onnx")) choice.available.push_back(name);
  }
  std::ranges::sort(choice.available);
  if (!option_given && !chosen_name.empty()) {
    const std::filesystem::path chosen = found.parent_path() / chosen_name;
    if (std::filesystem::exists(chosen, error)) {
      choice.path = chosen;
    } else {
      log_channel("app")->warn("modèle {} introuvable : {}", chosen_name, choice.default_name);
    }
  }
  return choice;
}

// B18-06 : un modèle pris hors du paquet (installation d'AnimeJaNai faite à la main) se dit au journal : la
// qualité de l'image et le moteur construit dépendent alors d'un fichier que le lecteur ne vérifie pas.
void note_outside_bundle(const std::filesystem::path& found, const std::filesystem::path& bundled_dir,
                         bool option_given) {
  if (option_given || bundled_dir.empty() || found.parent_path() == bundled_dir) return;
  log_channel("app")->info("modèle pris hors du paquet : {} (dossier {})", core::to_utf8(found.filename()),
                           core::to_utf8(found.parent_path().filename()));
}

}  // namespace

std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings,
                                                 const std::filesystem::path& bundled) {
  PlayOptions defaults = options;
  if (!options.was_given("--model")) defaults.model.reset();
  auto found = find_model(defaults, default_model_candidates(home, bundled));
  if (!found) return std::unexpected(std::move(found.error()));
  note_outside_bundle(*found, bundled.empty() ? bundled : bundled / "onnx", options.was_given("--model"));
  // Le catalogue réduit : un modèle hors catalogue (hérité) ne se choisit plus et n'entre pas au menu ;
  // --model reste libre.
  const std::string chosen = std::ranges::contains(kSpanCatalogue, settings.model) ? settings.model : "";
  ModelChoice choice = choose_in(*found, "2x_", chosen, options.was_given("--model"));
  std::erase_if(choice.available,
                [](const std::string& name) { return !std::ranges::contains(kSpanCatalogue, name); });
  return choice;
}

std::expected<ModelChoice, Refusal> choose_rife_model(const PlayOptions& options,
                                                      const std::filesystem::path& home,
                                                      const PlaySettings& settings,
                                                      std::span<const std::string_view> preference,
                                                      const std::filesystem::path& bundled) {
  auto found = find_rife_model(options, default_rife_candidates(home, preference, bundled));
  if (!found) return std::unexpected(std::move(found.error()));
  note_outside_bundle(*found, bundled.empty() ? bundled : bundled / "rife",
                      options.was_given("--rife-model"));
  const std::string chosen =
      std::ranges::contains(kRifeCatalogue, settings.rife_model) ? settings.rife_model : "";
  return choose_in(*found, "rife_v", chosen, options.was_given("--rife-model"));
}

std::span<const std::string_view> rife_preference(const GpuFacts& gpu) {
  if ((gpu.vendor == GpuVendor::Nvidia && gpu.tensor_cores) || (gpu.vendor == GpuVendor::Amd && gpu.wmma))
    return kRifeForRtx;
  return kRifeForOthers;
}

std::vector<RifeEntry> rife_menu(const std::vector<std::string>& available) {
  std::vector<RifeEntry> menu;
  menu.reserve(kRifeCatalogue.size());
  for (const std::string_view file : kRifeCatalogue)
    menu.push_back({.file = std::string(file), .present = std::ranges::contains(available, file)});
  return menu;
}

std::filesystem::path tier_variant(const std::filesystem::path& model, PrecisionTier tier) {
  if (tier == PrecisionTier::Balanced) return model;
  std::filesystem::path variant = model;
  variant.replace_filename(model.stem().string() + (tier == PrecisionTier::Performance ? "_fp8" : "_fp32") +
                           model.extension().string());
  std::error_code error;
  return std::filesystem::is_regular_file(variant, error) ? variant : model;
}

PrecisionTier model_tier(const std::optional<std::filesystem::path>& model) {
  if (!model) return PrecisionTier::Balanced;
  const std::string stem = core::to_utf8(model->stem());
  if (stem.ends_with("_fp8")) return PrecisionTier::Performance;
  return stem.ends_with("_fp32") ? PrecisionTier::Quality : PrecisionTier::Balanced;
}

TierSupport sibling_tiers(const std::filesystem::path& model, bool fp8_engine) {
  return TierSupport{.performance = fp8_engine && tier_variant(model, PrecisionTier::Performance) != model,
                     .quality = tier_variant(model, PrecisionTier::Quality) != model};
}

}  // namespace ajp::app::play
