// Câblage des threads de la lecture vus du thread main : les boîtes à qui la session
// parle (ThreadStages), ce que les threads lui renvoient (MainMail), l'horloge hôte qui
// remplace l'audio quand le fichier n'a pas de piste audio ou quand l'écran mène, et l'horloge
// que lisent la session et le thread gpu (LeadClock).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

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

// Horloge maître sans piste audio, ou quand l'écran mène : le temps hôte, mis en marche et arrêté
// par la session, à la vitesse qui cale la sortie sur les vsyncs (1 sinon).
class HostClock final : public MasterClock {
 public:
  ClockSnapshot snapshot() const override;
  // Part de media_us ; keep_position : d'où elle s'est arrêtée si c'est la même époque (la reprise
  // quand l'écran mène et que l'audio la suit, comme l'audio maître reprend d'où il s'est arrêté).
  void start(std::uint32_t epoch, std::int64_t media_us, std::int64_t host_us, bool keep_position = false);
  void pause(std::int64_t host_us);
  void flush(std::uint32_t epoch);
  // Vitesse : tout de suite si l'horloge court (depuis sa position à host_us), sinon au départ.
  void set_rate(double rate, std::int64_t host_us);
  // Reprend la position d'une autre horloge à host_us, en marche ou non : le passage de l'audio
  // maître à l'écran maître ne fait ni saut ni retour.
  void adopt(const ClockSnapshot& from, bool running, std::int64_t host_us);

 private:
  mutable std::mutex mutex_;
  ClockSnapshot snapshot_;
  double rate_ = 1.0;
};

// L'horloge que lisent la session et le thread gpu quand le fichier a du son : celle de l'audio,
// ou l'horloge hôte quand l'écran mène (synchro sur l'écran, ThreadStages::set_speed). Le choix
// change en pleine lecture ; les deux horloges vivent aussi longtemps qu'elle.
class LeadClock final : public MasterClock {
 public:
  LeadClock(std::shared_ptr<const MasterClock> audio, std::shared_ptr<const MasterClock> host)
      : audio_(std::move(audio)), host_(std::move(host)) {}
  ClockSnapshot snapshot() const override { return (host_leads() ? host_ : audio_)->snapshot(); }
  bool host_leads() const { return host_leads_.load(std::memory_order_acquire); }
  void lead_with_host(bool on) { host_leads_.store(on, std::memory_order_release); }

 private:
  std::shared_ptr<const MasterClock> audio_;
  std::shared_ptr<const MasterClock> host_;
  std::atomic<bool> host_leads_{false};
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
  // Toujours : maître sans audio ; avec audio, prête à mener quand l'écran se cale (lead).
  std::shared_ptr<HostClock> host_clock;
  LeadClock* lead = nullptr;  // avec audio : qui mène, l'audio ou l'horloge hôte
};

class ThreadStages final : public Stages {
 public:
  explicit ThreadStages(StageLinks links) : links_(std::move(links)) {}

  bool flush(std::uint32_t epoch) override;
  bool seek(const cmd::Seek& seek) override;
  bool play(std::uint32_t epoch, std::int64_t media_us) override;
  bool pause() override;
  bool slot_free(const SlotFree& free) override { return links_.video->post(free); }
  bool idle() const override { return links_.video->idle(); }
  bool request_subtitles(const subs::RenderRequest& request) override;
  bool audio(const Command& command) override;
  bool set_speed(double speed) override;
  bool audio_lost(std::uint32_t epoch, std::int64_t media_us) override;
  void quit() override;

 private:
  StageLinks links_;
  std::uint32_t flushed_epoch_ = 0;
  bool playing_ = false;     // play depuis la dernière pause : l'horloge hôte reprise court
  double speed_ = 1.0;       // dernière vitesse acceptée par set_speed
  bool audio_lost_ = false;  // l'horloge hôte mène faute d'audio, jusqu'au prochain Flush
};

}  // namespace ajp::app::play
