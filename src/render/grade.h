// Grade : les hooks mpv du grade chargés par pl_mpv_user_shader_parse, et leurs paramètres.
// Interne à render : le compositeur les passe à pl_render_image sur l'étape OUTPUT.
#pragma once

#include <array>
#include <expected>
#include <memory>
#include <optional>
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
  // Les images vidéo distinctes par seconde à l'écran (//!PARAM video_fps, 24 par défaut) : les lissages du
  // grade, réglés pour une image à 24 i/s, gardent ainsi leur durée en secondes. Hors de GradeParams : la
  // cadence n'est pas un réglage et change sans recharger le grade.
  void set_video_fps(double fps);
  // --hdr-metadata : la compression que KWin faisait sans métadonnées, faite par le grade (dernière passe),
  // pour la luminosité SDR de KDE `reference` et le pic `peak` de l'écran ; nullopt : passe sautée.
  struct KwinMatch {
    double reference = 200.0;
    double peak = 1000.0;
    friend bool operator==(const KwinMatch&, const KwinMatch&) = default;
  };
  void set_kwin_match(std::optional<KwinMatch> match);

  // Le look : jusqu'à trois LUT 3D enchaînées, lues par anime-hdr-look.hook (nullptr : une identité 2x2x2, la
  // LUT est alors sautée avec son intensité à 0). Refait le hook du look, donc à n'appeler qu'au changement
  // de LUT ; les intensités passent par set_params. Refus si libplacebo ne lit pas la texture.
  std::expected<void, Refusal> set_looks(const std::array<const CubeLut*, kLookSlots>& luts);

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
