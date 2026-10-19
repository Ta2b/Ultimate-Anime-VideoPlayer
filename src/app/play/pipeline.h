// Les threads de la lecture d'un fichier : source (media), gpu (VideoStage), audio et subs,
// reliés entre eux et au thread main (MainInbox), puis arrêtés dans l'ordre inverse.
//
//   source ──images──▶ gpu ──RingFrame──▶ main ──SlotFree──▶ gpu
//      │──paquets audio──▶ audio (horloge maître)
//      └──paquets texte──▶ subs ──SubtitleFrame──▶ main
//
// Le thread gpu vient d'une fabrique : le vrai (Program, backend d'inférence) dans
// player.cpp, une doublure dans les tests.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "app/play/bridges.h"
#include "app/play/links.h"
#include "app/play/session.h"
#include "app/play/tracks.h"
#include "app/play/video_stage.h"
#include "app/play/wiring.h"
#include "audio/audio_thread.h"
#include "core/clock.h"
#include "core/thread.h"
#include "media/hw_api.h"
#include "media/selection.h"
#include "media/source.h"
#include "media/track.h"
#include "subs/subs_messages.h"
#include "subs/subs_thread.h"

namespace ajp::app::play {

struct PipelineConfig {
  std::filesystem::path file;
  media::DecoderOptions decoder;
  media::LanguagePriorities languages{.audio = {"jpn", "ja"}, .subtitles = {"fre", "fr", "eng", "en"}};
  bool audio = true;      // false : piste audio ignorée, horloge hôte
  bool subtitles = true;  // false : aucune piste de sous-titres
  // Pistes choisies avant une réouverture de la chaîne : elles remplacent le choix par langue si
  // elles existent. subtitle_track : rien = choix par langue ; nullopt dedans = aucune.
  std::optional<int> audio_track;
  std::optional<std::optional<int>> subtitle_track;
  subs::RendererOptions subs;
};

// Ce que reçoit la fabrique du thread gpu.
struct VideoStageInit {
  const media::TrackInfo* track = nullptr;  // la piste vidéo choisie
  RingSink* ring = nullptr;                 // boîte de main
  std::function<void()> credit;             // rend une place au thread source
  ThreadReport report;                      // refus et arrêt du thread
  const MasterClock* clock = nullptr;       // horloge maître de la lecture ; vit plus longtemps que le thread
};
using VideoStageFactory = std::function<std::expected<std::unique_ptr<VideoStage>, Refusal>(VideoStageInit)>;

class Pipeline {
 public:
  // Ouvre le fichier, choisit les pistes, ouvre le décodeur et lance les threads. Le
  // thread source part en dernier : tout ce qu'il nourrit existe déjà.
  static std::expected<std::unique_ptr<Pipeline>, Refusal> open(const PipelineConfig& config,
                                                                const VideoStageFactory& make_video);

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  ~Pipeline();  // source d'abord, puis gpu, subs et audio

  MainInbox& inbox() { return inbox_; }
  Stages& stages() { return *stages_; }
  const MasterClock& clock() const { return *clock_; }
  // À chaque tour de main : un Credit pour source si subs a refusé un paquet.
  void pump();

  const media::TrackInfo& video_track() const { return video_track_; }
  // Ce que la session doit savoir du fichier : durée, taille, sous-titres.
  SessionConfig session_config() const;
  std::vector<std::string> describe() const;  // pistes choisies, pour l'OSD et le journal

  // Pistes qu'on peut choisir pendant la lecture.
  const TrackMenu& tracks() const { return menu_; }
  // Change la piste audio : ses paquets valent à partir de l'époque from_epoch, celle du seek que
  // seek() lance (à la position courante) ; audio ouvre ensuite son décodeur. Refus :
  // TrackUnavailable (piste inconnue, pas d'audio), PacketBacklog (boîte audio pleine).
  std::expected<void, Refusal> select_audio(int track, std::uint32_t from_epoch,
                                            const std::function<void()>& seek);
  // Change la piste de sous-titres (rien : aucune). La session relit ensuite la position courante
  // par un seek, pour que subs reçoive les paquets de la nouvelle piste. Refus : TrackUnavailable,
  // SubtitleBacklog.
  std::expected<void, Refusal> select_subtitles(std::optional<int> track);

 private:
  Pipeline() = default;
  const audio::AudioCodecInfo* audio_codec(int track) const;
  const media::TrackInfo* track_info(int track) const;

  // Ordre de déclaration = ordre inverse de destruction : la source s'arrête la première,
  // la boîte de main disparaît la dernière.
  MainInbox inbox_{kMainInboxBounds};
  MainMail mail_{inbox_, nullptr};
  HostClock host_clock_;
  std::unique_ptr<audio::AudioThread> audio_;
  std::unique_ptr<AudioBridge> audio_bridge_;
  std::shared_ptr<subs::SubsInbox> subs_inbox_;
  std::unique_ptr<subs::SubsThread> subs_thread_;
  std::unique_ptr<SubsBridge> subs_bridge_;
  media::SourceInbox source_inbox_{media::kSourceInboxBounds};
  std::unique_ptr<VideoStage> video_;
  std::unique_ptr<ThreadStages> stages_;
  std::shared_ptr<const MasterClock> audio_clock_;
  const MasterClock* clock_ = &host_clock_;
  std::unique_ptr<media::SourceThread> source_;

  media::TrackInfo video_track_;
  std::optional<media::TrackInfo> audio_track_;
  std::optional<media::TrackInfo> subtitle_track_;
  std::int64_t duration_us_ = 0;
  std::string decoder_api_;
  TrackMenu menu_;
  std::vector<std::pair<int, audio::AudioCodecInfo>> audio_codecs_;  // pistes audio décodables
  std::vector<media::TrackInfo> tracks_;                             // audio et sous-titres
};

}  // namespace ajp::app::play
