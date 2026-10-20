// L'encodeur vidéo d'un export : le premier candidat d'encoder_choice.h qui s'ouvre, ses images
// reçues en mémoire hôte (chargées sur des surfaces VA-API ou Vulkan quand il le demande), ses
// paquets rendus un par un. Interne au module.
#pragma once

#include <expected>
#include <functional>
#include <string>

#include "core/frame.h"
#include "core/types.h"
#include "export/av_handles.h"
#include "export/encoder_choice.h"

namespace ajp::exporter {

struct EncoderSetup {
  int width = 0;
  int height = 0;
  int fps_num = 0;
  int fps_den = 1;
  AVRational sample_aspect{0, 1};  // celui de la source ; {0, 1} : inconnu
  ColorInfo color;
  Codec codec = Codec::Hevc;
  Quality quality = Quality::VeryHigh;
  GpuVendor vendor = GpuVendor::Other;
};

class VideoEncoder {
 public:
  // Refus : EncoderUnavailable, avec chaque candidat essayé et sa raison.
  static std::expected<VideoEncoder, Refusal> open(const EncoderSetup& setup);

  const AVCodecContext& context() const { return *ctx_; }
  Layout layout() const { return layout_; }
  const std::string& label() const { return label_; }  // « NVENC (hevc_nvenc) »

  using PacketSink = std::function<std::expected<void, Refusal>(AVPacket&)>;
  // Encode une image en mémoire hôte, au format de layout() et datée dans la base de temps de
  // context() ; nullptr vide l'encodeur. Chaque paquet prêt passe par sink. Refus : EncodeFailed.
  std::expected<void, Refusal> encode(AVFrame* frame, const PacketSink& sink);

 private:
  VideoEncoder() = default;
  std::expected<void, Refusal> drain(const PacketSink& sink);

  BufferPtr device_;  // appareil VA-API ou Vulkan de l'encodeur ; vide pour un encodeur hôte
  CodecContextPtr ctx_;
  FramePtr hw_frame_;
  PacketPtr packet_;
  Upload upload_ = Upload::Host;
  Layout layout_ = Layout::P010;
  std::string label_;
};

}  // namespace ajp::exporter
