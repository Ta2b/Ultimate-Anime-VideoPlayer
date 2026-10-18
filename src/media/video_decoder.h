// Décodage vidéo avec libavcodec : matériel par une liste d'essais ordonnée,
// repli logiciel si la liste le permet, refus nommé si rien ne marche.
//
// NVDEC garde sa copie de sécurité : FFmpeg 9.0.2 copie chaque image dans le
// pool CUDA (nvdec.c:568, av_frame_make_writable) sauf si
// AV_HWACCEL_FLAG_UNSAFE_OUTPUT est posé. Ce drapeau rendrait directement les
// surfaces du décodeur, en nombre fixe : une image tenue par gpu bloquerait
// le décodage, ou serait réécrite sous ses pieds. On ne le pose jamais.
#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <span>

#include "core/frame.h"
#include "core/types.h"
#include "media/hw_api.h"
#include "media/packet.h"
#include "media/track.h"

namespace ajp::media {

class VideoDecoder {
 public:
  // Essaie options.attempts dans l'ordre ; les essais écartés restent dans rejected().
  static std::expected<VideoDecoder, Refusal> open(const TrackInfo& track, const DecoderOptions& options);

  VideoDecoder(VideoDecoder&&) noexcept;
  VideoDecoder& operator=(VideoDecoder&&) noexcept;
  ~VideoDecoder();

  // API qui décode vraiment : Software après un repli dans get_format.
  HwApi api() const;
  std::span<const Attempt> rejected() const;
  // Paquets de données invalides sautés depuis l'ouverture (journalisés).
  int skipped_packets() const;

  // Envoie un paquet de la piste ; à appeler quand receive() a rendu std::nullopt.
  std::expected<void, Refusal> send(const Packet& packet);
  // Fin du flux : receive() rend les images restantes, puis drained() devient vrai.
  std::expected<void, Refusal> send_end();
  // Image suivante, sans identité ni époque ; std::nullopt s'il faut un paquet.
  std::expected<std::optional<DecodedFrame>, Refusal> receive();
  bool drained() const;
  // Au seek : vide le décodeur sans le rouvrir.
  void flush();

 private:
  struct State;
  explicit VideoDecoder(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace ajp::media
