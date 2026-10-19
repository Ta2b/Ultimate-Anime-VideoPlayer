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
#include "app/play/frame_queue.h"
#include "app/play/frame_stats.h"
#include "app/play/links.h"
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

struct SessionConfig {
  std::int64_t start_us = 0;             // > 0 : seek exact à l'ouverture
  std::int64_t duration_us = 0;          // 0 : inconnue ; borne les seeks relatifs
  std::int64_t stop_after_us = 0;        // quitte après ce temps de lecture ; 0 jamais
  std::int64_t fullscreen_after_us = 0;  // bascule en plein écran après ce temps ; 0 jamais
  bool subtitles = false;                // une piste est choisie : demander leurs images
  int video_width = 0;                   // taille de stockage de la vidéo, pour libass
  int video_height = 0;
  std::optional<subs::VideoMatrix> video_matrix;  // couleurs VSFilter
  std::vector<std::string> header;                // premières lignes de l'OSD (modèle, moteur)
  std::int64_t display_period_us = 0;             // période de rafraîchissement de l'écran ; 0 : mesurée
  bool stats = false;                             // lignes de mesure affichées (Tab les bascule)
};

// Une ligne par présentation, pour --stats.
struct PresentRecord {
  std::int64_t host_us = 0;
  std::int64_t pts_us = 0;
  bool fresh = false;
  std::int64_t render_us = 0;
  std::int64_t interval_us = 0;  // depuis l'image neuve précédente ; 0 pour un redessin
  std::int64_t dropped = 0;      // total des images sautées
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
  // Un tour : choisit l'image, puis compose et présente seulement si quelque chose a changé
  // (image neuve, sous-titres, OSD, interface, taille de sortie) ; rien sinon. Rend les slots. Un
  // refus de l'écran est rendu tel quel : la boucle s'arrête.
  std::expected<std::optional<PresentRecord>, Refusal> tick(std::int64_t host_now, const UiFrame& ui = {});
  // Rend tous les slots tenus et arrête les threads.
  void end();

  bool finished() const { return finished_; }
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
  void choose(std::int64_t host_now);
  const subs::SubtitleFrame* subtitles_for(std::int64_t pts_us);
  void check_end(std::int64_t host_now);

  SessionConfig config_;
  Stages& stages_;
  Screen& screen_;
  const MasterClock& clock_;

  std::uint32_t epoch_ = 0;
  bool prerolling_ = true;  // l'époque n'a pas encore d'image montrée
  bool paused_ = false;     // pause demandée par l'utilisateur
  bool stream_ended_ = false;
  bool finished_ = false;
  std::optional<std::int64_t> pending_seek_us_;  // cible du dernier seek, jusqu'à sa première image

  FrameQueue queue_;
  std::optional<RingFrame> current_;  // montrée, slot tenu par main
  std::vector<RingFrame> late_;       // réservé une fois, réutilisé à chaque tour
  SubtitleQueue subtitles_{kMaxKeptSubtitles};
  std::size_t subtitle_requests_ = 0;  // demandes de l'époque sans réponse

  bool fresh_ = false;       // ce tour montre une image neuve
  bool shown_once_ = false;  // une présentation a eu lieu : la suivante seulement si ça change
  std::optional<std::int64_t> shown_subtitles_pts_;  // sous-titres de la dernière présentation
  OutputSize shown_size_;                            // taille de sortie de la dernière présentation
  bool shown_ui_ = false;                            // la dernière présentation avait une interface
  bool show_stats_ = false;                          // lignes de mesure affichées
  std::vector<std::string> osd_;                     // relu toutes les kOsdPeriodUs
  std::optional<std::int64_t> osd_at_us_;            // rien : à relire au prochain tour
  std::optional<std::int64_t> playing_since_us_;     // instant hôte de la première image
  std::int64_t last_fresh_host_us_ = 0;
  bool fullscreen_done_ = false;
  FrameStats stats_;
  std::vector<std::string> refusals_;  // derniers refus, pour l'OSD
};

}  // namespace ajp::app::play
