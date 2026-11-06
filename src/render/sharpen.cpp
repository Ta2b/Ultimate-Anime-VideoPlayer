#include "render/sharpen.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>

#include "render/shaders.h"

namespace ajp::render {

std::expected<Sharpen, Refusal> Sharpen::create(pl_gpu gpu) {
  const std::string_view text = shaders::sharpen;
  const pl_hook* hook = pl_mpv_user_shader_parse(gpu, text.data(), text.size());
  if (hook == nullptr)
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_mpv_user_shader_parse (sharpen.hook)"});
  for (int i = 0; i < hook->num_parameters; ++i) {
    const pl_hook_par& par = hook->parameters[i];
    if (par.type == PL_VAR_FLOAT && std::string_view(par.name) == "sharpness") return Sharpen(hook, par.data);
  }
  pl_mpv_user_shader_destroy(&hook);
  return std::unexpected(Refusal{RefusalCode::FieldMissing, "//!PARAM sharpness absent de sharpen.hook"});
}

Sharpen::Sharpen(Sharpen&& other) noexcept
    : hook_(std::exchange(other.hook_, nullptr)),
      param_(std::exchange(other.param_, nullptr)),
      strength_(other.strength_) {}

Sharpen& Sharpen::operator=(Sharpen&& other) noexcept {
  if (this != &other) {
    if (hook_ != nullptr) pl_mpv_user_shader_destroy(&hook_);
    hook_ = std::exchange(other.hook_, nullptr);
    param_ = std::exchange(other.param_, nullptr);
    strength_ = other.strength_;
  }
  return *this;
}

Sharpen::~Sharpen() {
  if (hook_ != nullptr) pl_mpv_user_shader_destroy(&hook_);
}

void Sharpen::set_strength(double strength) {
  strength_ = std::isnan(strength) ? 0.0 : std::clamp(strength, 0.0, 1.0);
  if (param_ != nullptr) param_->f = static_cast<float>(strength_);
}

}  // namespace ajp::render
