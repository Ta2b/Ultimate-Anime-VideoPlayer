// La passe du masque des effets (render/effect_mask.h) : sur OUTPUT, avant la passe anime du grade, elle pose
// le masque dans l'alpha de l'image (lu par 3-anime.hook quand fx_on vaut 1, puis remplacé par la montée des
// émetteurs). Le masque est une petite texture (une valeur par tuile de la carte de la scène, r8), lue en
// bilinéaire. Interne à render : Grade la pose devant ses hooks quand le masque n'est pas neutre.
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
  // Refus si le GPU n'a pas de r8 filtrable.
  static std::expected<std::unique_ptr<EffectPass>, Refusal> create(pl_gpu gpu);
  ~EffectPass();
  EffectPass(const EffectPass&) = delete;
  EffectPass& operator=(const EffectPass&) = delete;
  EffectPass(EffectPass&&) = delete;
  EffectPass& operator=(EffectPass&&) = delete;

  // Les valeurs de 0 à 1, cols × rows, ligne par ligne ; cover : EffectTiles::cover_x et cover_y. Envoyées au
  // GPU au rendu suivant.
  void set(int cols, int rows, std::span<const float> values, double cover_x, double cover_y);
  const pl_hook* hook() const;

 private:
  struct Impl;
  explicit EffectPass(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace kirari::render
