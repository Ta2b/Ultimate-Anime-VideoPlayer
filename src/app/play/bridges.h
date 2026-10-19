// Raccords entre media et les modules qui ne le connaissent pas : audio et subs. Ils
// traduisent les paquets du thread source dans les messages de ces modules (protocole des
// w1-results : audio.AudioPacket, subs.TrackPacket et AttachFont).
#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "audio/audio_thread.h"
#include "audio/packet.h"
#include "core/types.h"
#include "media/packet.h"
#include "media/track.h"
#include "subs/subs_messages.h"

namespace ajp::app::play {

// Paramètres du flux audio recopiés d'AVCodecParameters. Refus AudioCodecUnavailable
// pour un codec sans nom connu de FFmpeg.
std::expected<audio::AudioCodecInfo, Refusal> audio_codec_info(const media::TrackInfo& track);

// Paquets de la piste audio choisie vers le thread audio, appelé depuis le thread source.
// La copie des octets est permise ici : ce sont des paquets audio, pas des images.
//
// Toutes les pistes audio passent par lui ; il ne garde que la piste choisie. Un changement de
// piste vaut à partir d'une époque : les paquets d'avant gardent l'ancienne piste (son décodeur les
// lit encore), ceux d'après prennent la nouvelle (après le Flush, audio ouvre son décodeur).
class AudioBridge final : public media::PacketSink {
 public:
  AudioBridge(audio::AudioThread& audio, int track) : audio_(audio), track_(track), previous_(track) {}

  // Depuis main, avant le seek d'époque from_epoch qui fait entendre la piste track.
  void select(int track, std::uint32_t from_epoch);
  // La piste que garde un paquet de cette époque.
  int track_for(std::uint32_t epoch) const;

  // Depuis main, avant le Flush d'un seek : sous l'époque epoch, les échantillons d'avant
  // target_us sont jetés après décodage (seek exact) ; rien pour un seek rapide.
  void expect_seek(std::uint32_t epoch, std::optional<std::int64_t> target_us);

  bool try_push(media::Packet&& packet) override;

 private:
  audio::AudioThread& audio_;
  mutable std::mutex mutex_;
  std::uint32_t trim_epoch_ = 0;
  std::optional<std::int64_t> trim_us_;
  int track_;
  int previous_;
  std::uint32_t switch_epoch_ = 0;
};

// La piste de sous-titres choisie, en SelectTrack : ASS avec son en-tête, ou texte (SRT).
// Refus TrackUnavailable pour un format que subs ne lit pas (PGS, VobSub).
std::expected<subs::SelectTrack, Refusal> subtitle_track(const media::TrackInfo& track, std::uint32_t id);

// Les polices jointes du fichier, à envoyer avant SelectTrack.
std::vector<subs::AttachFont> attached_fonts(std::span<const media::TrackInfo> tracks);

// Identifiant de piste de subs (SelectTrack, TrackPacket) d'un flux du conteneur : jamais 0.
inline std::uint32_t subs_track_id(int stream_index) {
  return static_cast<std::uint32_t>(stream_index) + 1U;
}

// Paquets de la piste de sous-titres choisie vers le thread subs. Toutes les pistes lisibles
// passent par lui ; les autres sont écartées ici (subs écarterait aussi un paquet d'une autre
// piste que la sienne). Sa boîte ne rend pas de Credit : sur un refus, take_refused() le dit une
// fois à main, qui rend lui-même un Credit à source.
class SubsBridge final : public media::PacketSink {
 public:
  SubsBridge(std::shared_ptr<subs::SubsInbox> inbox, std::optional<int> track)
      : inbox_(std::move(inbox)), track_(track.value_or(kNone)) {}

  // Depuis main : la piste dont les paquets passent ; rien : aucune.
  void select(std::optional<int> track) { track_.store(track.value_or(kNone)); }
  bool try_push(media::Packet&& packet) override;
  bool take_refused() { return refused_.exchange(false); }

 private:
  static constexpr int kNone = -1;
  std::shared_ptr<subs::SubsInbox> inbox_;
  std::atomic<int> track_;
  std::atomic<bool> refused_{false};
};

}  // namespace ajp::app::play
