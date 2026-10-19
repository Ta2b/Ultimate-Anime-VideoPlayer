#include "app/play/model_choice.h"

#include <algorithm>
#include <string_view>
#include <system_error>
#include <utility>

#include "core/log.h"

namespace ajp::app::play {
namespace {

// Le défaut trouvé, la liste de son dossier (fichiers qui commencent par prefix), et le fichier
// choisi au menu s'il existe à côté.
ModelChoice choose_in(std::filesystem::path found, std::string_view prefix, const std::string& chosen_name,
                      bool option_given) {
  ModelChoice choice{.path = found, .default_name = found.filename().string()};
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(found.parent_path(), error)) {
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

}  // namespace

std::expected<ModelChoice, Refusal> choose_model(const PlayOptions& options,
                                                 const std::filesystem::path& home,
                                                 const PlaySettings& settings) {
  PlayOptions defaults = options;
  if (!options.was_given("--model")) defaults.model.reset();
  auto found = find_model(defaults, default_model_candidates(home));
  if (!found) return std::unexpected(std::move(found.error()));
  return choose_in(std::move(*found), "2x_", settings.model, options.was_given("--model"));
}

std::expected<ModelChoice, Refusal> choose_rife_model(const PlayOptions& options,
                                                      const std::filesystem::path& home,
                                                      const PlaySettings& settings) {
  auto found = find_rife_model(options, default_rife_candidates(home));
  if (!found) return std::unexpected(std::move(found.error()));
  return choose_in(std::move(*found), "rife_v", settings.rife_model, options.was_given("--rife-model"));
}

}  // namespace ajp::app::play
