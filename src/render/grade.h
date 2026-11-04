// Grade : les hooks mpv du grade chargés par pl_mpv_user_shader_parse, et leurs paramètres.
// Interne à render : le compositeur les passe à pl_render_image sur l'étape OUTPUT.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"
#include "render/cube_lut.h"
#include "render/grade_params.h"

namespace ajp::render {

// Le corps du grade : en trois passes, la scène donnée par le lecteur (GradeParams::fs_apl, fs_bright,
// sh_*) ; ou celui d'avant en neuf passes, qui mesure la scène lui-même (--legacy-grade, une version).
// Ou la méthode « HDR anime » (--hdr-anime) : surfaces ancrées, seuls les émetteurs montent, appliquée par un
// gain lisse (anime-hdr-anime) ; elle réutilise les deux premières passes du grade en trois passes.
// Fused : le grade en trois passes, le look lu dans la passe du grade ; FusedSplit : les mêmes passes, le
// look dans sa passe à part (la chaîne d'avant 2.2, --legacy-grade=split) ; Legacy : neuf passes ; Anime :
// --hdr-anime.
enum class GradeBody : std::uint8_t { Fused, Legacy, Anime, FusedSplit };

class Grade {
 public:
  // VendorCallFailed : un hook ne se lit pas ; FieldMissing : un //!PARAM attendu manque.
  static std::expected<Grade, Refusal> create(pl_gpu gpu, GradeBody body = GradeBody::Fused,
                                              bool skin_mask = true);

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

  // Dans l'ordre d'exécution : les passes du grade (trois, ou neuf pour celui d'avant), le look, puis le vrai
  // noir.
  std::span<const pl_hook* const> hooks() const;
  // La valeur posée d'un //!PARAM de la passe du grade (après set_params, set_video_fps, set_kwin_match),
  // nullopt s'il n'y est pas : le grade de l'anneau (kernels/spec/grade.h) en reprend les valeurs.
  std::optional<float> param(std::string_view name) const;
  // La pile de LUT composée que lit le hook (look_mix à 1, 65 points), nullptr sans look actif.
  const CubeLut* look_stack() const;
  // Faux : le masque peau est coupé (sans --skin-mask).
  bool skin_mask() const;
  // La mire d'étalonnage, qui remplace l'image par des patchs en luminance absolue (PQ).
  const pl_hook* calibration() const;

 private:
  struct Impl;
  explicit Grade(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
