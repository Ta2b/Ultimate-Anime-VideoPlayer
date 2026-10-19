// Encodeur vidéo de l'export : libavcodec, choisi par nom, plus la conversion hôte par
// libswscale de RGBA16 PQ BT.2020 vers le format de l'encodeur. Interne à export.
#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "core/types.h"
#include "export/av_util.h"
#include "export/encoder_settings.h"
#include "export/frame_source.h"
#include "export/hdr10.h"

namespace ajp::exporter {

struct EncodeGeometry {
  int width = 0;
  int height = 0;
  AVRational frame_rate{0, 1};  // cadence nominale : durée des paquets et contrôle de débit
};

class VideoEncoder {
 public:
  // Valide les réglages, cherche l'encodeur dans la construction de FFmpeg et l'ouvre.
  // Refus : ceux de validate_settings, EncoderUnavailable (absent de la construction, appareil
  // ou pilote manquant, option refusée), FormatUnsupported (format absent de sa liste).
  // global_header : en-têtes du flux dans extradata, comme Matroska et MP4 l'exigent.
  static std::expected<VideoEncoder, Refusal> open(const EncoderSettings& settings,
                                                   const EncodeGeometry& geometry,
                                                   const Hdr10Mastering& mastering, bool global_header);

  // Convertit l'image (taille déjà vérifiée) et la donne à l'encodeur ; pts en microsecondes
  // depuis le début de l'export. Refus EncodeFailed qui cite l'appel.
  std::expected<void, Refusal> send(const HostFrame& frame, std::int64_t pts_us);

  // Fin du flux : l'encodeur rend ensuite ses derniers paquets.
  std::expected<void, Refusal> flush();

  // Paquet suivant dans packet (base de temps : celle du contexte) ; false quand l'encodeur
  // attend une image de plus ou a tout rendu.
  std::expected<bool, Refusal> receive(AVPacket& packet);

  const AVCodecContext& context() const { return *ctx_; }

 private:
  VideoEncoder() = default;

  std::expected<void, Refusal> prepare_frames(const EncoderSpec& spec);
  std::expected<void, Refusal> renew_converted();
  std::expected<AVFrame*, Refusal> convert(const HostFrame& frame);
  std::uint16_t* rescale_for_yuv(const HostFrame& frame);

  CodecContextPtr ctx_;
  SwsPtr sws_;
  FramePtr source_;                       // vue sur l'image hôte (RGB), ou sur yuv_input_ (YUV)
  std::vector<std::uint16_t> yuv_input_;  // YUV : l'image ramenée à l'échelle 65280 de libswscale
  FramePtr converted_;                    // format logiciel de l'encodeur
  FramePtr device_;                       // encodeurs Vulkan : l'image téléversée
  BufferPtr device_ctx_;
  BufferPtr frames_ctx_;
  std::int64_t frame_duration_us_ = 0;
};

}  // namespace ajp::exporter
