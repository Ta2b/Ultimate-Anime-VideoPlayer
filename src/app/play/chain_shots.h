// La pré-analyse des plans dans la lecture (media/shot_probe.h) : lancée à l'ouverture d'une vidéo avec le
// grade, elle survit aux réouvertures de la chaîne et aux vidéos suivantes ; le grade (chain_hdr.cpp,
// grade_tick) y lit la lumière du plan de l'image qui vient. Une vidéo analysée entière va au cache
// (media/shot_cache.h), seule la vidéo lue s'analyse.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "app/play/settings.h"
#include "media/shot_probe.h"
#include "media/shot_timeline.h"
#include "media/track.h"
#include "render/grade_params.h"

namespace ajp::app::play {

struct Chain;

// Une vidéo et son analyse. Remplacée ou détruite en route, son analyse s'arrête et ce qu'elle a fait va au
// cache, marqué incomplet : la prochaine ouverture de la vidéo la reprend.
struct ShotJob {
  std::filesystem::path cache_dir;  // vide : pas de cache
  std::filesystem::path file;
  // Ce que le thread de l'analyse a lu du cache avant d'analyser (A42-14) : l'empreinte de la vidéo, et si
  // elle y était entière. Lu par le thread de la fenêtre une fois `read` levé.
  struct Cache {
    std::string key;
    std::atomic<bool> read{false};
    std::atomic<bool> complete{false};
    // Gardée entière au cache par keep_if_done : l'arrêt de l'analyse n'y écrit plus sa part (B42-05).
    std::atomic<bool> kept{false};
  };
  std::shared_ptr<Cache> cache;
  // Partagée avec le thread de l'analyse, qui peut lui survivre (B30-01).
  std::shared_ptr<media::ShotTimeline> timeline;
  std::unique_ptr<media::ShotProbe> probe;
  // Le chemin de décodage de l'analyse et le gain de son limiteur de flashs (media/shot_cache.h, B42-07).
  std::uint32_t variant = 0;
  double flash_gain = 1.0;  // le gain du limiteur de flashs de l'analyse (flash_gain_of)
  bool faces = false;       // la peau des visages cherchée (face_skin)
  bool complete = false;    // connue entière : lue du cache, ou analysée jusqu'au bout et gardée

  ShotJob() = default;
  ShotJob(const ShotJob&) = delete;
  ShotJob& operator=(const ShotJob&) = delete;
  ShotJob(ShotJob&& other) noexcept = default;
  ShotJob& operator=(ShotJob&& other) noexcept;
  ~ShotJob();
  // Arrête l'analyse ; son thread garde au cache ce qu'elle a fait, si la vidéo n'y est pas entière
  // (ShotProbeOptions::on_stop), sans retenir le thread de la fenêtre (B42-05).
  void retire() noexcept;
};

class LatestWriter;  // latest_writer.h

// Ce qui vit d'une vidéo à l'autre (le lecteur) : le cache et la vidéo lue.
struct ShotWork {
  std::filesystem::path cache_dir;  // vide : pas de cache
  media::ShotProbeOptions probe;    // les réglages de l'analyse (--probe-nvdec : cuvid_resize)
  // --shot-cuts : l'interpolation lit ses coupes dans les plans, l'analyse part donc aussi sans grade
  // (A42-09).
  bool shot_cuts = false;
  ShotJob current;
  // Le thread d'écriture de la lecture (1.11) : le cache des plans complets s'y écrit. Nul : sur place.
  LatestWriter* writer = nullptr;
};

struct PlayOptions;  // options.h
// Ce que le lecteur garde d'une vidéo à l'autre, d'après ses options. Un essai de calibration (--bench-out)
// ne s'en sert pas (A42-05) : les rafales de décodage de la pré-analyse faussaient la mesure de la machine.
ShotWork shot_work(std::filesystem::path cache_dir, const PlayOptions& options);

// Ce que l'ouverture sait de la sortie : une source HDR (PQ, HLG), une sortie HDR, le pic du grade.
struct ShotOutput {
  bool source_hdr = false;
  bool hdr_output = false;
  double peak_nits = render::grade_limits::kDefaultPeakNits;
};
// À l'ouverture d'une chaîne : reprend la vidéo du cache, ou lance l'analyse (depuis la part gardée) ; la
// garde pour le même fichier ; rien si ni le grade ni l'interpolation n'en lisent les plans (l'analyse en
// cours s'arrête). Le grade d'une source HDR (PQ, HLG) sans grade natif HDR n'en lit pas ; en sortie HDR, le
// grade n'en lit que les flashs dangereux (B42-02).
void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us, const ShotOutput& output = {});
// La même, avant la préparation des moteurs : à la sortie que la chaîne prendra (system_hdr, sauf sdr) et à
// son pic (B42-02 : en sortie HDR, rien sans le limiteur de flashs ; B42-07 : les flashs jugés à ce pic).
void open_shots_early(ShotWork& work, const std::filesystem::path& file, PlaySettings settings,
                      std::int64_t start_us, const media::TrackInfo& video, std::optional<double> peak_seen,
                      bool sdr);

// Le cache : à appeler à chaque tour, ne fait rien la plupart du temps.
void tend_shots(ShotWork& work);

// Le plan de l'image qui suit la dernière préparée, s'il est analysé : sa lumière et son numéro (il change à
// chaque coupe). Cale aussi l'analyse sur la lecture (avance, sauts). C42-08 : les statistiques gamma du plan
// (moyenne, centiles, bleu, chroma) n'avaient aucun lecteur ; seule la lumière sert au grade.
struct ShotScene {
  render::SceneLight light;  // pour le grade en trois passes (render/grade_params.h)
  std::uint64_t serial = 0;
  bool fade = false;
  // Un plan de la pré-analyse : ses bornes, et s'il commence à une vraie coupe. Faux au point de reprise
  // d'une analyse (un saut, une reprise) : le plan d'avant est le même plan réel (B42-01).
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
  bool cut_before = true;
  // Un plan de la pré-analyse, aux valeurs fixes ; faux pour une scène vue à l'ingestion, dont la moyenne
  // bouge à chaque image (A42-04).
  bool planned = false;
  // La peau des visages du plan (face_skin) ; aucune teinte hors de la pré-analyse.
  render::SkinTones skins;
};
// B42-01 : shot suit sans vraie coupe le plan que le grade suivait, fini à previous_end_us (-1 : aucun), à
// 0,1 s près : le même plan réel, coupé au point de reprise d'une analyse. La lumière glisse au lieu de
// sauter.
inline bool continues_shot(const ShotScene& shot, std::int64_t previous_end_us) {
  constexpr std::int64_t kJoinUs = 100'000;
  const std::int64_t gap = shot.start_us - previous_end_us;
  return shot.planned && !shot.cut_before && previous_end_us >= 0 && gap <= kJoinUs && gap >= -kJoinUs;
}
// Ce que le grade HDR anime fait à la lumière d'une source SDR, pour le limiteur de flashs : le blanc de ses
// surfaces (320 nits sous un pic de 400, 690 sous 1 000, 1 304 sous 2 000 : docs/hdr-anime.md) sur celui de
// la source, 203 nits. Continue et croissante (B42-07 : 652 nits à 1 001 contre 690 à 1 000).
inline constexpr double flash_gain_of(double peak_nits) {
  const double white = peak_nits <= 1000.0   ? 320.0 + ((peak_nits - 400.0) * (690.0 - 320.0) / 600.0)
                       : peak_nits <= 2000.0 ? 690.0 + ((peak_nits - 1000.0) * (1304.0 - 690.0) / 1000.0)
                                             : 0.652 * peak_nits;
  return (white > 203.0 ? white : 203.0) / 203.0;
}
// Le numéro d'une scène vue à l'ingestion : jamais celui d'un plan.
inline constexpr std::uint64_t kLiveSerialBit = std::uint64_t{1} << 62U;
std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us);
// La scène de cette même image vue à l'ingestion seule (media/live_scenes.h), sans la pré-analyse.
std::optional<ShotScene> next_live(Chain& to);
// Des flashs dangereux autour de l'image qui vient (media/flash_guard.h, P3.51) : marqués d'avance par la
// pré-analyse, sinon vus à l'ingestion.
bool next_flashing(Chain& to);
// Les plans connus passés au planificateur du grade v2 de l'écran (8.6), et au livre des coupes sous
// --shot-cuts (8.7), quand la chronologie change.
void feed_plan_shots(Chain& to);

}  // namespace ajp::app::play
