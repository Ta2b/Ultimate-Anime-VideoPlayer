#include "render/anime4k.h"

#include <format>
#include <string>

#include "render/anime4k_shaders.h"

namespace ajp::render {

std::string_view name(Anime4kMode mode) {
  switch (mode) {
    case Anime4kMode::A:
      return "A";
    case Anime4kMode::B:
      return "B";
    case Anime4kMode::C:
      return "C";
  }
  return "A";
}

std::string_view name(Anime4kSize size) {
  return size == Anime4kSize::Quality ? "VL" : "M";
}

Anime4kMode anime4k_mode(int source_height, int work_height) {
  if (source_height <= 0) return Anime4kMode::A;
  // Réduite : le rapport dit ce que la réduction a laissé (Anime4K : « 1080p->720p downscaled » en B,
  // « 1080p->480p downscaled » en C). 1,75 sépare 1080 → 720 (1,5) de 1080 → 540 (2).
  if (work_height > 0 && work_height < source_height) {
    const double ratio = static_cast<double>(source_height) / work_height;
    return ratio >= 1.75 ? Anime4kMode::C : Anime4kMode::B;
  }
  if (source_height >= 600 && source_height < 900) return Anime4kMode::B;
  return Anime4kMode::A;
}

std::vector<std::string_view> anime4k_texts(Anime4kChoice choice) {
  const bool vl = choice.size == Anime4kSize::Quality;
  std::vector<std::string_view> texts{shaders::Anime4K_Clamp_Highlights};
  switch (choice.mode) {
    case Anime4kMode::A:
      texts.push_back(vl ? shaders::Anime4K_Restore_CNN_VL : shaders::Anime4K_Restore_CNN_M);
      texts.push_back(vl ? shaders::Anime4K_Upscale_CNN_x2_VL : shaders::Anime4K_Upscale_CNN_x2_M);
      break;
    case Anime4kMode::B:
      texts.push_back(vl ? shaders::Anime4K_Restore_CNN_Soft_VL : shaders::Anime4K_Restore_CNN_Soft_M);
      texts.push_back(vl ? shaders::Anime4K_Upscale_CNN_x2_VL : shaders::Anime4K_Upscale_CNN_x2_M);
      break;
    case Anime4kMode::C:
      texts.push_back(vl ? shaders::Anime4K_Upscale_Denoise_CNN_x2_VL
                         : shaders::Anime4K_Upscale_Denoise_CNN_x2_M);
      break;
  }
  return texts;
}

std::expected<Anime4k, Refusal> Anime4k::create(pl_gpu gpu, Anime4kChoice choice) {
  std::vector<const pl_hook*> hooks;
  for (const std::string_view text : anime4k_texts(choice)) {
    const pl_hook* hook = pl_mpv_user_shader_parse(gpu, text.data(), text.size());
    if (hook == nullptr) {
      for (const pl_hook*& made : hooks) pl_mpv_user_shader_destroy(&made);
      // La première ligne //!DESC du shader refusé le nomme.
      const std::size_t at = text.find("//!DESC ");
      const std::string_view desc = at == std::string_view::npos
                                        ? text.substr(0, 0)
                                        : text.substr(at + 8, text.find('\n', at) - at - 8);
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed,
                                     std::format("pl_mpv_user_shader_parse (Anime4K, {})", desc)});
    }
    hooks.push_back(hook);
  }
  return Anime4k(choice, std::move(hooks));
}

Anime4k::Anime4k(Anime4k&& other) noexcept
    : choice_(other.choice_), hooks_(std::exchange(other.hooks_, {})) {}

Anime4k& Anime4k::operator=(Anime4k&& other) noexcept {
  if (this != &other) {
    destroy();
    choice_ = other.choice_;
    hooks_ = std::exchange(other.hooks_, {});
  }
  return *this;
}

Anime4k::~Anime4k() {
  destroy();
}

void Anime4k::destroy() {
  for (const pl_hook*& hook : hooks_) pl_mpv_user_shader_destroy(&hook);
  hooks_.clear();
}

bool is_anime4k_pass(std::string_view description) {
  return description.find("Anime4K-") != std::string_view::npos;
}

}  // namespace ajp::render
