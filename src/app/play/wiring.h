// Câblage des threads de la lecture vus du thread main : les boîtes à qui la session
// parle (ThreadStages), ce que les threads lui renvoient (MainMail), et l'horloge hôte
// qui remplace l'audio quand le fichier n'a pas de piste audio.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "app/play/bridges.h"
#include "app/play/links.h"
#include "app/play/video_stage.h"
#include "audio/audio_thread.h"
#include "core/clock.h"
#include "core/thread.h"
#include "media/source.h"
#include "subs/subs_messages.h"
#include "subs/subtitle_frame.h"

namespace ajp::app::play {

// Horloge maître sans piste audio : le temps hôte, mis en marche et arrêté par la session.
class HostClock final : public MasterClock {
 public:
  ClockSnapshot snapshot() const override;
  void start(std::uint32_t epoch, std::int64_t media_us, std::int64_t host_us);
  void pause(std::int64_t host_us);
  void flush(std::uint32_t epoch);

 private:
  mutable std::mutex mutex_;
  ClockSnapshot snapshot_;
};

// Ce que les autres threads envoient à main, depuis leur thread, dans sa boîte.
class MainMail final : public RingSink, public subs::SubtitleSink, public media::SourceListener {
 public:
  // audio, s'il existe, reçoit EndOfStream avec la fin du flux.
  MainMail(MainInbox& inbox, audio::AudioThread* audio) : inbox_(inbox), audio_(audio) {}
  // Avant le départ du thread source, seul à appeler end_of_stream.
  void attach_audio(audio::AudioThread* audio) { audio_ = audio; }

  bool try_push(const RingFrame& frame) override { return inbox_.try_push(RingFrame{frame}); }
  bool try_push(subs::SubtitleFrame&& frame) override { return inbox_.try_push(std::move(frame)); }
  void refused(const Refusal& refusal) override { report("subs", refusal, false); }
  void end_of_stream(std::uint32_t epoch) override;
  void failed(const Refusal& refusal) override;

  // Rapporteur d'un thread : ses refus vont à l'OSD, son arrêt arrête la lecture.
  ThreadReport reporter(std::string stage);
  void report(std::string stage, const Refusal& refusal, bool fatal);

 private:
  MainInbox& inbox_;
  audio::AudioThread* audio_;
};

struct StageLinks {
  media::SourceInbox* source = nullptr;
  VideoStage* video = nullptr;
  audio::AudioThread* audio = nullptr;  // nullptr : pas de piste audio, l'horloge hôte mène
  AudioBridge* audio_bridge = nullptr;
  std::shared_ptr<subs::SubsInbox> subs;  // nullptr : pas de sous-titres
  HostClock* host_clock = nullptr;        // exigée sans audio
};

class ThreadStages final : public Stages {
 public:
  explicit ThreadStages(StageLinks links) : links_(std::move(links)) {}

  bool flush(std::uint32_t epoch) override;
  bool seek(const cmd::Seek& seek) override;
  bool play(std::uint32_t epoch, std::int64_t media_us) override;
  bool pause() override;
  bool slot_free(const SlotFree& free) override { return links_.video->post(free); }
  bool request_subtitles(const subs::RenderRequest& request) override;
  void quit() override;

 private:
  StageLinks links_;
  std::uint32_t flushed_epoch_ = 0;
};

}  // namespace ajp::app::play
