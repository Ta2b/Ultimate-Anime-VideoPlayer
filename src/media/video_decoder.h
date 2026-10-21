// Décodage vidéo avec libavcodec : matériel par une liste d'essais ordonnée,
// repli logiciel si la liste le permet, refus nommé si rien ne marche.
//
// NVDEC copie par défaut chaque image dans le pool CUDA (nvdec.c:568,
// av_frame_make_writable). Avec DecoderOptions::direct_surfaces, il pose
// AV_HWACCEL_FLAG_UNSAFE_OUTPUT : l'image est la surface de sortie mappée par
// cuvidMapVideoFrame, rendue au démappage quand sa dernière référence tombe
// (frame->buf[1]). La surface de décodage, elle, n'est plus tenue par l'image
// rendue (decode.c:724), seulement par le DPB du décodeur : il ne s'en trouve
// pas à court, et une image tenue n'est jamais réécrite. Seules les surfaces de sortie sont en
// nombre fixe (le pool, 64 au plus) : l'aval en tient au plus direct_surfaces,
// et extra_hw_frames monte à ce nombre.
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
  // Erreurs de données (AVERROR_INVALIDDATA) passées depuis l'ouverture, journalisées.
  // Un flux abîmé perd des images mais continue, avec ou sans threads d'images :
  // ces erreurs sortent à l'envoi d'un paquet ou à la réception d'une image.
  int data_errors() const;
  // Surfaces réservées par l'hwaccel (taille initiale du pool de son
  // AVHWFramesContext), extra_hw_frames compris ; 0 en logiciel ou avant la
  // première image. Sous NVDEC, c'est ulNumDecodeSurfaces avant sa limite de 32.
  int hw_surfaces() const;
  // Vrai si NVDEC rend ses surfaces sans copie (DecoderOptions::direct_surfaces).
  bool direct_surfaces() const;

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
