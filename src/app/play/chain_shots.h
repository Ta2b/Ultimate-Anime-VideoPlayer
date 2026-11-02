// La pré-analyse des plans dans la lecture (media/shot_probe.h) : lancée à l'ouverture d'une vidéo avec le
// grade adaptatif, elle survit aux réouvertures de la chaîne et aux vidéos suivantes ; le grade adaptatif
// (chain_hdr.cpp, adaptive_tick) y lit le plan de l'image qui vient. Une vidéo analysée entière va au cache
// (media/shot_cache.h), seule la vidéo lue s'analyse.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "app/play/settings.h"
#include "gpu/kernel_ops.h"
#include "media/shot_probe.h"
#include "media/shot_timeline.h"

namespace ajp::app::play {

struct Chain;

// Une vidéo et son analyse. Remplacée ou détruite en route, son analyse s'arrête et ce qu'elle a fait va au
// cache, marqué incomplet : la prochaine ouverture de la vidéo la reprend.
struct ShotJob {
  std::filesystem::path cache_dir;  // vide : pas de cache
  std::filesystem::path file;
  std::string key;  // son empreinte dans le cache
  std::unique_ptr<media::ShotTimeline> timeline;
  std::unique_ptr<media::ShotProbe> probe;
  bool complete = false;  // connue entière : lue du cache, ou analysée jusqu'au bout et gardée

  ShotJob() = default;
  ShotJob(const ShotJob&) = delete;
  ShotJob& operator=(const ShotJob&) = delete;
  ShotJob(ShotJob&& other) noexcept = default;
  ShotJob& operator=(ShotJob&& other) noexcept;
  ~ShotJob();
  // Arrête l'analyse et garde au cache ce qu'elle a fait, si la vidéo n'y est pas entière.
  void retire() noexcept;
};

// Ce qui vit d'une vidéo à l'autre (le lecteur) : le cache et la vidéo lue.
struct ShotWork {
  std::filesystem::path cache_dir;  // vide : pas de cache
  ShotJob current;
};

// À l'ouverture d'une chaîne : reprend la vidéo du cache, ou lance l'analyse (depuis la part gardée) ; la
// garde pour le même fichier ; rien sans grade adaptatif (l'analyse en cours
// s'arrête).
void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us);

// Le cache : à appeler à chaque tour, ne fait rien la plupart du temps.
void tend_shots(ShotWork& work);

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
