// Grade : les hooks mpv du grade chargés par pl_mpv_user_shader_parse, et leurs paramètres.
// Interne à render : le compositeur les passe à pl_render_image sur l'étape OUTPUT.
#pragma once

#include <array>
#include <expected>
#include <memory>
#include <span>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"
#include "render/cube_lut.h"
#include "render/grade_params.h"

namespace ajp::render {

class Grade {
 public:
  // VendorCallFailed : un hook ne se lit pas ; FieldMissing : un //!PARAM attendu manque.
  static std::expected<Grade, Refusal> create(pl_gpu gpu);

  Grade(Grade&&) noexcept;
  Grade& operator=(Grade&&) noexcept;
  Grade(const Grade&) = delete;
  Grade& operator=(const Grade&) = delete;
  ~Grade();

  // Écrit chaque valeur dans le //!PARAM du même nom ; pris en compte au prochain rendu.
  void set_params(const GradeParams& params);
  const GradeParams& params() const;

  // Le look : la LUT 3D lue par anime-hdr-look.hook (nullptr : une identité 2x2x2, le look est alors sauté
  // avec look_mix = 0). Refait le hook du look, donc à n'appeler qu'au changement de LUT ; l'intensité passe
  // par set_params. Refus si libplacebo ne lit pas la texture.
  std::expected<void, Refusal> set_look(const CubeLut* lut);

  // Dans l'ordre d'exécution : les neuf passes du grade, le look, puis la dixième (vrai noir).
  std::span<const pl_hook* const> hooks() const;
  // La mire d'étalonnage, qui remplace l'image par des patchs en luminance absolue (PQ).
  const pl_hook* calibration() const;

 private:
  struct Impl;
  explicit Grade(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
