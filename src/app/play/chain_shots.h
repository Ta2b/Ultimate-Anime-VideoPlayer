// La pré-analyse des plans dans la lecture (media/shot_probe.h) : lancée à l'ouverture d'une vidéo avec le
// grade adaptatif, elle survit aux réouvertures de la chaîne sur le même fichier ; le grade adaptatif
// (chain_hdr.cpp, adaptive_tick) y lit le plan de l'image qui vient.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

#include "app/play/settings.h"
#include "gpu/kernel_ops.h"
#include "media/shot_probe.h"
#include "media/shot_timeline.h"

namespace ajp::app::play {

struct Chain;

// Ce qui vit d'une ouverture à l'autre (Player) : la chronologie du fichier et son analyse.
struct ShotWork {
  std::filesystem::path file;
  std::unique_ptr<media::ShotTimeline> timeline;
  std::unique_ptr<media::ShotProbe> probe;
};

// À l'ouverture d'une chaîne : relance l'analyse pour un autre fichier, la garde pour le même ; rien sans
// grade adaptatif (l'analyse en cours s'arrête).
void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us);

// Le plan de l'image qui suit la dernière préparée, s'il est analysé : ses valeurs sur l'échelle de RTX HDR
// et son numéro (il change à chaque coupe). Cale aussi l'analyse sur la lecture (avance, sauts).
struct ShotScene {
  gpu::KernelOps::SceneStats stats;
  std::uint64_t serial = 0;
  bool fade = false;
};
std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us);

// La luminance moyenne de la vidéo selon les plans analysés, une fois assez couverte (une minute, ou toute
// la vidéo) ; vide sinon.
std::optional<double> shots_series_luma(const Chain& to);

}  // namespace ajp::app::play
