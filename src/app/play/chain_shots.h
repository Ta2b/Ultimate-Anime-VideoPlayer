// La pré-analyse des plans dans la lecture (media/shot_probe.h) : lancée à l'ouverture d'une vidéo avec le
// grade, elle survit aux réouvertures de la chaîne et aux vidéos suivantes ; le grade (chain_hdr.cpp,
// grade_tick) y lit la lumière du plan de l'image qui vient. Une vidéo analysée entière va au cache
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

class LatestWriter;  // latest_writer.h

// Ce qui vit d'une vidéo à l'autre (le lecteur) : le cache et la vidéo lue.
struct ShotWork {
  std::filesystem::path cache_dir;  // vide : pas de cache
  media::ShotProbeOptions probe;    // les réglages de l'analyse (--probe-nvdec : cuvid_resize)
  ShotJob current;
  // Le thread d'écriture de la lecture (1.11) : le cache des plans complets s'y écrit. Nul : sur place.
  LatestWriter* writer = nullptr;
};

// À l'ouverture d'une chaîne : reprend la vidéo du cache, ou lance l'analyse (depuis la part gardée) ; la
// garde pour le même fichier ; rien sans grade (l'analyse en cours s'arrête).
// Une source HDR (PQ, HLG) sans grade natif HDR n'en a pas besoin : rien ne part.
void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us, bool source_hdr = false);

// Le cache : à appeler à chaque tour, ne fait rien la plupart du temps.
void tend_shots(ShotWork& work);

// Le plan de l'image qui suit la dernière préparée, s'il est analysé : ses valeurs sur l'échelle de
// SceneStats et son numéro (il change à chaque coupe). Cale aussi l'analyse sur la lecture (avance, sauts).
struct ShotScene {
  gpu::KernelOps::SceneStats stats;
  render::SceneLight light;  // pour le grade en trois passes (render/grade_params.h)
  std::uint64_t serial = 0;
  bool fade = false;
};
std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us);
// La scène de cette même image vue à l'ingestion seule (media/live_scenes.h), sans la pré-analyse.
std::optional<ShotScene> next_live(Chain& to);
// Des flashs dangereux autour de l'image qui vient, vus à l'ingestion (media/flash_guard.h, P3.51).
bool next_flashing(Chain& to);
// Les plans connus passés au planificateur du grade v2 de l'écran (8.6), et au livre des coupes sous
// --shot-cuts (8.7), quand leur nombre change.
void feed_plan_shots(Chain& to);

}  // namespace ajp::app::play
