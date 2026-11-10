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
  pl_var_data* strength = nullptr;
  pl_var_data* linear = nullptr;
  for (int i = 0; i < hook->num_parameters; ++i) {
    const pl_hook_par& par = hook->parameters[i];
    if (par.type != PL_VAR_FLOAT) continue;
    if (std::string_view(par.name) == "sharpness") strength = par.data;
    if (std::string_view(par.name) == "sharpen_linear") linear = par.data;
  }
  if (strength != nullptr && linear != nullptr) return Sharpen(hook, strength, linear);
  pl_mpv_user_shader_destroy(&hook);
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "//!PARAM sharpness ou sharpen_linear absent de sharpen.hook"});
}

Sharpen::Sharpen(Sharpen&& other) noexcept
    : hook_(std::exchange(other.hook_, nullptr)),
      param_(std::exchange(other.param_, nullptr)),
      linear_(std::exchange(other.linear_, nullptr)),
      strength_(other.strength_) {}

Sharpen& Sharpen::operator=(Sharpen&& other) noexcept {
  if (this != &other) {
    if (hook_ != nullptr) pl_mpv_user_shader_destroy(&hook_);
    hook_ = std::exchange(other.hook_, nullptr);
    param_ = std::exchange(other.param_, nullptr);
    linear_ = std::exchange(other.linear_, nullptr);
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
