// Ce que l'essai au chargement du GMFSS natif (gmfss_trial.h, docs/gmfss-natif.md) voit d'un réseau natif, le
// GridNet de la fusion ou la paire : des morceaux dont les sommes se choisissent à part (gmfss_mix.h), et le
// compte du réglage de leurs convolutions au banc (gmfss_conv_plan.cuh). En-tête sans std::expected : il est
// lu par nvcc en C++20.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/native_sums.h"

namespace kirari::gpu::cuda {

// Le réglage des convolutions : chacune pour les trois sommes ; une forme seule possible n'est pas mesurée,
// une couche de même noyau et de même géométrie qu'une autre déjà réglée en reprend les formes. notes : les
// couches dont des formes ont été écartées (sortie différente de celle de la forme de référence).
struct GmfssTuning {
  int layers = 0;
  int measured = 0;  // formes mesurées au banc, toutes sommes
  int rejected = 0;
  int single = 0;  // (couche, sommes) à une seule forme
  int reused = 0;  // couches reprises
  std::vector<std::string> notes;
};

class GmfssNativeParts {
 public:
  virtual ~GmfssNativeParts() = default;
  virtual int pieces() const = 0;
  virtual std::string piece_name(int piece) const = 0;
  // Le morceau a-t-il une tête : des convolutions qui écrivent une sortie du réseau ?
  virtual bool has_head(int piece) const = 0;
  // Les sommes de chaque morceau (precisions[morceau]), la tête à part ; elles valent pour les lancements
  // suivants.
  virtual void use(std::span<const RifeNativePrecision> precisions) = 0;
  // Libère les poids rangés pour les tuiles que les sommes posées n'emploient pas : use ne doit plus en
  // changer.
  virtual void trim() = 0;
  virtual const GmfssTuning& tuning() const = 0;
  virtual std::int64_t workspace_bytes() const = 0;
};

}  // namespace kirari::gpu::cuda
