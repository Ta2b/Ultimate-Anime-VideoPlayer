// Grade : les hooks mpv du grade chargés par pl_mpv_user_shader_parse, et leurs paramètres.
// Interne à render : le compositeur les passe à pl_render_image sur l'étape OUTPUT.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"
#include "render/cube_lut.h"
#include "render/grade_params.h"

namespace ajp::render {

class Grade {
 public:
  // VendorCallFailed : un hook ne se lit pas ; FieldMissing : un //!PARAM attendu manque.
  // Le grade HDR du lecteur, en trois passes sur OUTPUT (anime-hdr-anime), puis le look et le vrai noir.
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

  // P4.9 : la pile de LUT se compose et son hook se lit sur un thread (65³ points, 4,4 Mo de texte) ; le
  // lecteur l'active, adopt() pose le résultat prêt sur le thread qui rend. Coupé (le défaut, les tests et
  // la mesure de rampe) : set_params compose sur place.
  void set_background(bool on);
  // Le hook du look composé en tâche de fond, s'il est prêt : vrai quand il remplace l'ancien (les hooks
  // ont changé).
  bool adopt();
  // À tenir pendant la soumission d'une présentation (pl_swapchain_submit_frame). libplacebo enregistre les
  // commandes d'un pl_gpu sous verrou, sauf celle que la swapchain prend pour présenter (pl_vk_steal_cmd) :
  // une texture de look créée par le thread de fond entre cette prise et sa soumission désordonne le
  // sémaphore de la file (assertion de vk_cmd_submit, « pool->sync[…].value == cmd->sync.value », 4
  // lancements sur 120 sous lavapipe). Le thread de fond tient le même verrou pendant qu'il lit son hook.
  std::unique_lock<std::mutex> gpu_guard();

  // Dans l'ordre d'exécution : la passe du grade (ses trois passes dans un hook), le look, puis le vrai noir.
  std::span<const pl_hook* const> hooks() const;
  // La valeur posée d'un //!PARAM de la passe du grade (après set_params, set_video_fps, set_kwin_match),
  // nullopt s'il n'y est pas.
  std::optional<float> param(std::string_view name) const;
  // La pile de LUT composée que lit le hook (look_mix à 1, 65 points), nullptr sans look actif.
  const CubeLut* look_stack() const;
  // La mire d'étalonnage, qui remplace l'image par des patchs en luminance absolue (PQ).
  const pl_hook* calibration() const;

 private:
  struct Impl;
  explicit Grade(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
