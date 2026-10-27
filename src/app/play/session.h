// Session de lecture du thread main : époques, amorce, pause, seek, choix de l'image à
// chaque vsync, sous-titres, retour des slots, fin de lecture.
//
// Horloge : la maître (audio, ou l'horloge hôte sans piste audio) dit le temps média ; la
// session montre l'image dont le pts est atteint et rend les slots des images doublées.
//
// Amorce : à l'ouverture et après chaque seek, l'horloge est en pause. La première image
// de l'époque est montrée dès son arrivée, puis l'horloge part de son pts (sauf pause
// demandée), si bien que le son ne devance jamais l'image après un seek.
//
// Seek : Pause, Flush{e + 1} à tous les threads puis Seek au thread source ; les images et
// sous-titres d'époque < e + 1 sont jetés à leur arrivée.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "app/play/controls.h"
#include "app/play/frame_latency.h"
#include "app/play/frame_queue.h"
#include "app/play/frame_stats.h"
#include "app/play/links.h"
#include "app/play/settings.h"
#include "app/play/subtitle_queue.h"
#include "core/clock.h"
#include "core/messages.h"
#include "core/ring.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"

namespace ajp::app::play {

// Demandes de sous-titres en vol au plus, et réponses gardées : ensemble, render ne tient jamais
// plus de kMaxAtlasesInFlight − 1 atlas (contrat de subs/subtitle_frame.h). Les réponses gardées
// couvrent tout le ring, l'image à l'écran et celles en avance (voir SubtitleQueue) : le ring du
// lecteur n'a pas plus de kMaxKeptSubtitles slots.
inline constexpr std::size_t kMaxSubtitleRequests = 3;
inline constexpr std::size_t kMaxKeptSubtitles = subs::kMaxAtlasesInFlight - kMaxSubtitleRequests - 1;

// Mesures que la session ne tient pas elle-même (compteurs de RIFE, mémoire vidéo), lues à
// chaque relecture de l'OSD ; rien quand la mesure n'existe pas.
class StatsSource {
 public:
  virtual ~StatsSource() = default;
  virtual std::optional<std::string> rife() const = 0;
  virtual std::optional<std::string> memory() const = 0;
  // Période nominale des images de sortie (la source, divisée par le facteur de RIFE) ; 0 :
  // inconnue. Sans elle, la session ne compte pas les images manquées.
  virtual std::int64_t output_period_us() const { return 0; }
  // La même cadence, exacte (images par seconde) : la synchro sur l'écran s'y cale ; 0 : inconnue.
  virtual double output_fps() const { return 0.0; }
  // Images source que gpu a sautées avant leur calcul, déjà passées : la chaîne ne suit pas.
  virtual std::uint64_t skipped() const { return 0; }
};

struct SessionConfig {
  std::int64_t start_us = 0;             // > 0 : seek exact à l'ouverture
  std::int64_t duration_us = 0;          // 0 : inconnue ; borne les seeks relatifs
  std::vector<ChapterMark> chapters;     // du fichier, dans l'ordre
  std::int64_t stop_after_us = 0;        // quitte après ce temps de lecture ; 0 jamais
  std::int64_t fullscreen_after_us = 0;  // bascule en plein écran après ce temps ; 0 jamais
  bool subtitles = false;                // une piste est choisie : demander leurs images
  int video_width = 0;                   // taille de stockage de la vidéo, pour libass
  int video_height = 0;
  double video_sar = 1.0;  // forme du pixel de la vidéo (largeur / hauteur), 1 : carré
  std::optional<subs::VideoMatrix> video_matrix;  // couleurs VSFilter
  float subtitle_scale = 1.0F;                    // taille des sous-titres (subs::RenderTarget::font_scale)
  std::vector<std::string> header;                // la chaîne : modèles, tailles, moteur
  std::vector<std::string> tracks;                // pistes vidéo, audio et sous-titres
  std::string output;                             // sortie de l'écran (HDR10, pic de RTX HDR)
  std::int64_t display_period_us = 0;             // période de rafraîchissement de l'écran ; 0 : mesurée
  double display_hz = 0.0;                        // le même rafraîchissement, exact ; 0 : inconnu
  // Synchro sur l'écran : sur un écran fixe, la lecture accélère ou ralentit d'au plus 0,125 %
  // pour que chaque image tienne un nombre entier de vsyncs (aucune image doublée ni sautée par
  // dérive) ; l'audio suit en se rééchantillonnant. Sans calage possible, l'audio reste maître.
  bool display_sync = false;
  int swapchain_depth = 2;  // images que la file de l'écran garde au plus
  // Délai entre le choix d'une image et son apparition à l'écran ; 0 : estimé (rendu, attente du
  // vsync, file de l'écran). La session choisit l'image dont le pts tombe à son apparition.
  std::int64_t video_latency_us = 0;
  // Cadence fixe : l'image est représentée à chaque période de l'écran même sans changement, pour
  // que le VRR reste à une fréquence constante (une fréquence qui varie fait scintiller les scènes
  // sombres d'un OLED). Sans effet si la période est inconnue.
  bool steady_cadence = false;
  StatsSettings stats;                 // mesures affichées (Tab bascule toutes)
  const StatsSource* extra = nullptr;  // vit plus longtemps que la session
  // Le retard des images, de leur soumission par gpu à leur écriture vue par main : la session le
  // mesure, gpu le lit. Vit plus longtemps que la session ; rien : pas de mesure.
  FrameLatency* latency = nullptr;
};

// Une ligne par présentation, pour --stats.
struct PresentRecord {
  std::int64_t host_us = 0;
  std::int64_t pts_us = 0;
  bool fresh = false;
  std::int64_t render_us = 0;
  std::int64_t interval_us = 0;  // depuis l'image neuve précédente ; 0 pour un redessin
  std::int64_t dropped = 0;      // total des images sautées
  // Ce qui a demandé la présentation : « i » image neuve, « s » sous-titres, « o » OSD, « u »
  // interface, « t » taille de sortie ; plusieurs lettres si plusieurs ont changé.
  std::string why;
  std::int64_t missing = 0;         // total des images manquées (FrameStats::missing)
  std::int64_t lag_us = 0;          // retard de l'image neuve sur l'horloge ; 0 pour un redessin
  std::int64_t uninterpolated = 0;  // total des instants sans image à dessein (FrameStats)
};

class Session {
 public:
  Session(SessionConfig config, Stages& stages, Screen& screen, const MasterClock& clock);

  // À l'ouverture : seek vers start_us s'il y a lieu.
  void begin(std::int64_t host_now);
  void handle(MainInbox::Message&& message);
  // Commande de l'utilisateur (touches, interface) : pause, seek, volume, statistiques, quitter.
  void command(const Command& command);
  void toggle_fullscreen();
  // Sous-titres affichés ou non (piste choisie ou aucune) : demandes et réponses s'arrêtent.
  void set_subtitles(bool on);
  // Taille des sous-titres : les demandes suivantes la portent, et l'image à l'écran est redemandée.
  void set_subtitle_scale(float scale);
  // Mesures affichées, relues au tour suivant.
  void set_stats(const StatsSettings& stats);
  void set_steady_cadence(bool on) { config_.steady_cadence = on; }
  void set_display_sync(bool on) { config_.display_sync = on; }  // au tour suivant
  void set_video_latency(std::int64_t latency_us) { config_.video_latency_us = latency_us; }
  double speed() const { return speed_; }  // vitesse de lecture appliquée (1 : aucun calage)
  const StatsSettings& stats_settings() const { return show_stats_; }
  // Les pistes et la sortie ont changé (piste choisie, pic de RTX HDR) : relues au tour suivant.
  void set_tracks(std::vector<std::string> tracks);
  void set_output(std::string output);
  // Un tour : choisit l'image, puis compose et présente seulement si quelque chose a changé
  // (image neuve, sous-titres, OSD, interface, taille de sortie) ; rien sinon. Rend les slots. Un
  // refus de l'écran est rendu tel quel : la boucle s'arrête.
  std::expected<std::optional<PresentRecord>, Refusal> tick(std::int64_t host_now, const UiFrame& ui = {});
  // Rend tous les slots tenus et arrête les threads.
  void end();

  bool finished() const { return finished_; }
  bool quit_asked() const { return quit_; }  // q ou fenêtre fermée, pas la fin du fichier
  bool paused() const { return paused_; }
  std::uint32_t epoch() const { return epoch_; }
  const FrameStats& stats() const { return stats_; }
  const std::vector<std::string>& refusals() const { return refusals_; }
  std::vector<std::string> osd_lines() const;
  // Ce que la barre de lecture montre : la position (la cible d'un seek en cours, sinon l'image à
  // l'écran), la durée, la pause, le plein écran, les statistiques.
  ControlsView view() const;

 private:
  void seek_to(std::int64_t target_us, bool exact);
  void on_frame(const RingFrame& frame);
  void on_subtitles(subs::SubtitleFrame&& frame);
  void on_refusal(const StageRefusal& refusal);
  void request_subtitles(const RingFrame& frame);
  void retire(const RingFrame& frame);
  void retire_written();  // les images rendues avant d'être écrites, qui le sont maintenant
  void measure_latency(std::int64_t host_now);  // les images écrites depuis le tour d'avant
  void choose(std::int64_t host_now);
  const subs::SubtitleFrame* subtitles_for(std::int64_t pts_us);
  void check_end(std::int64_t host_now);
  void watch_clock(const ClockSnapshot& now, std::int64_t host_now);

  SessionConfig config_;
  Stages& stages_;
  Screen& screen_;
  const MasterClock& clock_;

  std::uint32_t epoch_ = 0;
  bool prerolling_ = true;                        // l'époque n'a pas encore d'image montrée
  std::optional<std::int64_t> preroll_since_us_;  // la première image de l'époque attend ses sous-titres
  bool paused_ = false;                           // pause demandée par l'utilisateur
  bool stream_ended_ = false;
  bool finished_ = false;
  bool quit_ = false;
  std::optional<std::int64_t> pending_seek_us_;  // cible du dernier seek, jusqu'à sa première image
  // Seek dont une boîte a refusé le Flush ou le Seek : rejoué par tick, jusqu'à ce que tout passe.
  struct SeekRetry {
    std::int64_t target_us = 0;
    bool exact = false;
    std::int64_t at_us = 0;
  };
  std::optional<SeekRetry> retry_seek_;
  std::int64_t last_tick_us_ = 0;                       // instant hôte du dernier tick
  std::optional<std::int64_t> clock_stalled_since_us_;  // des images attendent une horloge arrêtée
  std::int64_t audio_stall_grace_us_ = 1'500'000;

  FrameQueue queue_;
  std::optional<RingFrame> current_;   // montrée, slot tenu par main
  std::vector<RingFrame> late_;        // réservé une fois, réutilisé à chaque tour
  std::vector<RingFrame> unwritten_;   // rendues par main avant d'être écrites : parties une fois écrites
  std::vector<RingFrame> unmeasured_;  // reçues, pas encore vues écrites : leur retard reste à mesurer
  SubtitleQueue subtitles_{kMaxKeptSubtitles};
  std::size_t subtitle_requests_ = 0;  // demandes de l'époque sans réponse

  bool fresh_ = false;       // ce tour montre une image neuve
  bool shown_once_ = false;  // une présentation a eu lieu : la suivante seulement si ça change
  // Ce que montraient les sous-titres de la dernière présentation : une réponse plus récente au
  // même contenu (même atlas, mêmes rectangles) ne recompose pas l'écran.
  struct SubtitleLook {
    std::uint64_t generation = 0;  // 0 : aucun sous-titre
    std::vector<subs::SubtitleRect> rects;
    friend bool operator==(const SubtitleLook&, const SubtitleLook&) = default;
  };
  static SubtitleLook look_of(const subs::SubtitleFrame* frame);
  SubtitleLook shown_subtitles_;
  OutputSize shown_size_;                         // taille de sortie de la dernière présentation
  std::int64_t shown_at_us_ = 0;                  // instant hôte de la dernière présentation
  bool shown_ui_ = false;                         // la dernière présentation avait une interface
  StatsSettings show_stats_;                      // mesures affichées
  std::vector<std::string> osd_;                  // relu toutes les kOsdPeriodUs
  std::optional<std::int64_t> osd_at_us_;         // rien : à relire au prochain tour
  std::optional<std::int64_t> playing_since_us_;  // instant hôte de la première image
  std::int64_t last_fresh_host_us_ = 0;
  bool fullscreen_done_ = false;
  FrameStats stats_;
  std::vector<std::string> refusals_;  // derniers refus, pour l'OSD

  // La période attendue entre deux images neuves, 0 si inconnue ; et la mesure d'une image neuve,
  // à l'instant où elle paraît.
  std::int64_t pacing_period_us() const;
  void pace(const RingFrame& frame, std::int64_t visible_us);
  // Synchro sur l'écran : la vitesse qui cale la sortie, demandée quand elle change.
  void follow_display();
  std::int64_t display_period() const;  // la période de l'écran, connue, mesurée ou supposée
  std::int64_t display_delay_us(std::int64_t period) const;

  double speed_ = 1.0;
  double queue_fill_ = 0.0;  // part récente des présentations qui ont attendu une place en file
};

}  // namespace ajp::app::play
