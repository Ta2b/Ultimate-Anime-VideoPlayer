// Netteté : le hook mpv embarqué sharpen.hook (masque flou adaptatif sur MAIN), chargé par
// pl_mpv_user_shader_parse. Le compositeur de app/play le passe à pl_render_image devant le grade.
#pragma once

#include <expected>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"

namespace ajp::render {

class Sharpen {
 public:
  // VendorCallFailed : le hook ne se lit pas ; FieldMissing : son //!PARAM sharpness manque.
  static std::expected<Sharpen, Refusal> create(pl_gpu gpu);

  Sharpen(Sharpen&& other) noexcept;
  Sharpen& operator=(Sharpen&& other) noexcept;
  Sharpen(const Sharpen&) = delete;
  Sharpen& operator=(const Sharpen&) = delete;
  ~Sharpen();

  // La force, de 0 à 1 (bornée) ; prise en compte au prochain rendu.
  void set_strength(double strength);
  double strength() const { return strength_; }
  // Le hook à poser, nullptr à force 0 : l'image n'est alors pas touchée.
  const pl_hook* hook() const { return strength_ > 0.0 ? hook_ : nullptr; }

 private:
  Sharpen(const pl_hook* hook, pl_var_data* param) : hook_(hook), param_(param) {}
  const pl_hook* hook_ = nullptr;
  pl_var_data* param_ = nullptr;
  double strength_ = 0.0;
};

}  // namespace ajp::render
