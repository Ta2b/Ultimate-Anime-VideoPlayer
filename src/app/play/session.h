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
#include <deque>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "app/play/frame_queue.h"
#include "app/play/frame_stats.h"
#include "app/play/links.h"
#include "core/clock.h"
#include "core/messages.h"
#include "core/ring.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"

namespace ajp::app::play {

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
  // Commande de l'utilisateur (touches) : pause, seek, quitter.
  void command(const Command& command);
  void toggle_fullscreen();
  // Un tour : choisit l'image, compose, présente, rend les slots. Un refus de l'écran est
  // rendu tel quel : la boucle s'arrête.
  std::expected<std::optional<PresentRecord>, Refusal> tick(std::int64_t host_now);
  // Rend tous les slots tenus et arrête les threads.
  void end();

  bool finished() const { return finished_; }
  bool paused() const { return paused_; }
  std::uint32_t epoch() const { return epoch_; }
  const FrameStats& stats() const { return stats_; }
  const std::vector<std::string>& refusals() const { return refusals_; }
  std::vector<std::string> osd_lines() const;

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
  std::deque<subs::SubtitleFrame> subtitles_;
  std::size_t subtitle_requests_ = 0;  // demandes de l'époque sans réponse

  bool fresh_ = false;            // ce tour montre une image neuve
  std::vector<std::string> osd_;  // recomposé toutes les kOsdPeriodUs
  std::int64_t osd_at_us_ = 0;
  std::optional<std::int64_t> playing_since_us_;  // instant hôte de la première image
  std::int64_t last_fresh_host_us_ = 0;
  bool fullscreen_done_ = false;
  FrameStats stats_;
  std::vector<std::string> refusals_;  // derniers refus, pour l'OSD
};

}  // namespace ajp::app::play
