#include "app/play/anime4k_choice.h"

namespace ajp::app::play {

std::optional<render::Anime4kSize> anime4k_size(std::string_view model) {
  if (model == kAnime4kFast) return render::Anime4kSize::Fast;
  if (model == kAnime4kQuality) return render::Anime4kSize::Quality;
  return std::nullopt;
}

std::string anime4k_label(render::Anime4kSize size) {
  return size == render::Anime4kSize::Quality ? "Anime4K HQ" : "Anime4K Fast";
}

std::optional<render::Anime4kChoice> anime4k_choice(std::string_view model, bool upscales, int source_height,
                                                    int work_height) {
  const std::optional<render::Anime4kSize> size = anime4k_size(model);
  if (!size || !upscales) return std::nullopt;
  return render::Anime4kChoice{.size = *size, .mode = render::anime4k_mode(source_height, work_height)};
}

}  // namespace ajp::app::play
