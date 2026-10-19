// Grade : les hooks mpv du grade chargés par pl_mpv_user_shader_parse, et leurs paramètres.
// Interne à render : le compositeur les passe à pl_render_image sur l'étape OUTPUT.
#pragma once

#include <array>
#include <expected>
#include <memory>
#include <span>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"
#include "render/grade_params.h"

namespace ajp::render {

class Grade {
 public:
  // RenderFailed : un hook ne se lit pas ; FeatureUnavailable : un //!PARAM attendu manque.
  static std::expected<Grade, Refusal> create(pl_gpu gpu);

  Grade(Grade&&) noexcept;
  Grade& operator=(Grade&&) noexcept;
  Grade(const Grade&) = delete;
  Grade& operator=(const Grade&) = delete;
  ~Grade();

  // Écrit chaque valeur dans le //!PARAM du même nom ; pris en compte au prochain rendu.
  void set_params(const GradeParams& params);
  const GradeParams& params() const;

  // Dans l'ordre d'exécution : les neuf passes du grade, puis la dixième (vrai noir).
  std::span<const pl_hook* const> hooks() const;
  // La mire d'étalonnage, qui remplace l'image par des patchs en luminance absolue (PQ).
  const pl_hook* calibration() const;

 private:
  struct Impl;
  explicit Grade(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
