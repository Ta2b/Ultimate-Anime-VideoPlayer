// Logique du thread audio, sans le thread : décodage, file du périphérique,
// horloge et synchronisation. AudioThread la nourrit depuis sa boîte, un
// message à la fois ; les tests l'appellent directement.
//
// Deux horloges possibles :
// - VRR : l'audio est maître. Chaque mesure du périphérique publie la position
//   entendue dans clock(), que present et gpu lisent.
// - Écran fixe : present publie l'horloge maître, reçue par SyncTo. L'audio
//   se cale dessus (silence ajouté ou trames coupées), puis suit sa dérive en
//   changeant sa vitesse de lecture d'au plus 0,125 %.
//
// L'instant présent est celui de la sortie (AudioOutput::reading) : l'horloge
// hôte avec SDL, le temps simulé du faux périphérique dans les tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <optional>

#include "audio/clock.h"
#include "audio/decoder.h"
#include "audio/drift.h"
#include "audio/output.h"
#include "audio/packet.h"
#include "audio/pcm.h"
#include "core/clock.h"
#include "core/messages.h"
#include "core/types.h"

namespace spdlog {
class logger;
}

namespace ajp::audio {

// Choix du maître : sans horloge, l'audio est maître ; sinon il suit celle-ci.
struct SyncTo {
  std::shared_ptr<const MasterClock> master;
};

struct PlayerConfig {
  std::int64_t queue_target_us = 200'000;  // audio gardé d'avance dans la file du périphérique
  std::int64_t resync_us = 50'000;         // écart à l'écran au-delà duquel on recale franchement
  DriftGains drift;
};

// La configuration telle quelle si elle est dans ses bornes, un refus sinon.
std::expected<PlayerConfig, Refusal> checked(const PlayerConfig& config);

struct PlayerEvents {
  std::function<void()> packet_released;        // un paquet a quitté le module : sa place se libère
  std::function<void(const Refusal&)> refused;  // piste ou réglage refusés, pour l'OSD
};

class AudioPlayer {
 public:
  AudioPlayer(std::unique_ptr<AudioOutput> output, const PlayerConfig& config, PlayerEvents events);
  AudioPlayer(const AudioPlayer&) = delete;
  AudioPlayer& operator=(const AudioPlayer&) = delete;
  ~AudioPlayer();

  std::shared_ptr<const MasterClock> clock() const { return clock_; }

  void handle(AudioPacket&& packet);
  void handle(const AudioTrack& track);
  void handle(const EndOfStream& end);
  void handle(const Command& command);
  void handle(const Flush& flush);
  void handle(const DeviceReading& reading);
  void handle(const SyncTo& sync);

  // État, pour les tests et l'OSD.
  const AudioOutput& output() const { return *output_; }
  std::size_t held_packets() const { return held_.size(); }
  bool paused() const { return paused_; }

 private:
  void start_epoch(std::uint32_t epoch);
  void pump();
  void decode_next();
  void finish_stream();
  bool push(PcmBlock& block);
  bool align(PcmBlock& block);
  void publish(std::int64_t heard_us, const DeviceReading& reading);
  void follow(std::int64_t heard_us, const DeviceReading& reading);
  void realign();
  void pause();
  void resume();
  void set_delay(std::int64_t delay_us);
  void apply_ratio(double ratio);
  void release(std::size_t count) const;
  void report(const Refusal& refusal);

  std::unique_ptr<AudioOutput> output_;
  PlayerConfig config_;
  PlayerEvents events_;
  std::shared_ptr<spdlog::logger> log_;
  std::shared_ptr<AudioClock> clock_;
  std::shared_ptr<const MasterClock> master_;  // nul : l'audio est maître
  DriftRegulator regulator_;
  std::optional<AudioDecoder> decoder_;
  std::deque<AudioPacket> held_;  // reçus, pas encore décodés : la file du périphérique est pleine
  std::deque<PcmBlock> ready_;    // décodés, pas encore poussés
  std::optional<std::int64_t> trim_before_us_;
  std::optional<std::int64_t> last_follow_us_;
  std::uint32_t epoch_ = 0;
  bool paused_ = true;
  bool aligning_ = false;     // le prochain bloc se cale sur l'horloge maître
  bool end_pending_ = false;  // EndOfStream reçu, paquets pas encore tous décodés
  bool ended_ = false;        // toutes les trames de la piste sont poussées
};

}  // namespace ajp::audio
