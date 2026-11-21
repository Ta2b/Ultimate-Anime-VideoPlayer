// La passe du masque des effets (render/effect_mask.h) : sur OUTPUT, avant la passe anime du grade, elle pose
// le masque dans l'alpha de l'image (lu par 3-anime.hook quand fx_on vaut 1, puis remplacé par la montée des
// émetteurs). Elle n'est pas une passe à elle : quelques lignes ajoutées à la couleur en attente
// (PL_HOOK_SIG_COLOR), que libplacebo rend de toute façon pour la passe anime. Le masque est une petite
// texture (une valeur par tuile de la carte de la scène, r8 ou le premier format filtrable que le GPU offre),
// lue en bilinéaire si le format le permet. Interne à render : Grade la pose devant ses hooks quand le masque
// n'est pas neutre.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"

namespace kirari::render {

class EffectPass {
 public:
  // Refus si le GPU n'offre aucun format échantillonnable inscriptible (r8, rgba8, r32f, rgba32f), ou pas de
  // sommet vec2.
  static std::expected<std::unique_ptr<EffectPass>, Refusal> create(pl_gpu gpu);
  ~EffectPass();
  EffectPass(const EffectPass&) = delete;
  EffectPass& operator=(const EffectPass&) = delete;
  EffectPass(EffectPass&&) = delete;
  EffectPass& operator=(EffectPass&&) = delete;

  // Les valeurs de 0 à 1, cols × rows, ligne par ligne ; cover : EffectTiles::cover_x et cover_y. Envoyées au
  // GPU au rendu suivant.
  void set(int cols, int rows, std::span<const float> values, double cover_x, double cover_y);
  void set_guide(int cols, int rows, std::span<const float> values, double cover_x, double cover_y);
  const pl_hook* hook() const;

 private:
  struct Impl;
  explicit EffectPass(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace kirari::render
