// Thread audio : attend sa boîte Inbox et passe chaque message à AudioPlayer.
//
// Les paquets sont bornés en vol (dans la boîte ou tenus par le lecteur tant
// que la file du périphérique est pleine). Quand push_packet échoue, le
// producteur attend un Credit dans sa propre boîte, rendu par options.credit.
// Le périphérique réveille le thread à chaque demande de trames, par un
// DeviceReading posé dans la boîte : c'est aussi le rythme de l'horloge.
#pragma once

#include <atomic>
#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

#include "audio/output.h"
#include "audio/packet.h"
#include "audio/player.h"
#include "core/clock.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/thread.h"
#include "core/types.h"

namespace ajp::audio {

using AudioInbox = Inbox<AudioPacket, AudioTrack, EndOfStream, Command, Flush, DeviceReading, SyncTo>;

struct AudioThreadOptions {
  std::size_t max_packets = 256;  // paquets en vol ; couvre l'entrelacement des fichiers courants
  PlayerConfig player;
  std::function<void(Credit)> credit;           // rend une place au producteur
  std::function<void(const Refusal&)> refused;  // piste ou réglage refusés, pour l'OSD
};

class AudioThread {
 public:
  // Refuse des options hors bornes, ou un thread refusé par le système (ThreadUnavailable).
  // output doit être ouvert ; le thread le possède ensuite. Une exception qui arrêterait la
  // boucle part, en refus ThreadFailed, au journal et à options.refused.
  static std::expected<std::unique_ptr<AudioThread>, Refusal> start(std::unique_ptr<AudioOutput> output,
                                                                    AudioThreadOptions options);
  AudioThread(const AudioThread&) = delete;
  AudioThread& operator=(const AudioThread&) = delete;
  ~AudioThread();  // demande l'arrêt et attend la fin du thread

  // false : borne atteinte, le paquet n'est pas pris ; attendre un Credit.
  bool push_packet(AudioPacket&& packet);
  // AudioTrack, EndOfStream, Command, Flush ou SyncTo ; false si leur borne est atteinte.
  template <class M>
  bool post(M&& message) {
    static_assert(!std::is_same_v<std::remove_cvref_t<M>, AudioPacket>,
                  "les paquets passent par push_packet");
    static_assert(!std::is_same_v<std::remove_cvref_t<M>, DeviceReading>, "réservé au périphérique");
    return inbox_.try_push(std::forward<M>(message));
  }

  // Horloge audio, maître quand l'écran est VRR.
  std::shared_ptr<const MasterClock> clock() const { return clock_; }

 private:
  AudioThread(std::unique_ptr<AudioOutput> output, AudioThreadOptions options);
  void run(const std::stop_token& stop);
  void release_packet();

  std::size_t max_packets_;
  std::function<void(Credit)> credit_;
  std::atomic<std::size_t> in_flight_{0};
  AudioInbox inbox_;  // vit plus longtemps que la sortie, dont le rappel y écrit
  std::unique_ptr<AudioPlayer> player_;
  std::shared_ptr<const MasterClock> clock_;
  ThreadReport report_;  // journal et options.refused
  std::jthread thread_;  // dernier membre : s'arrête avant que le reste ne soit détruit
};

}  // namespace ajp::audio
