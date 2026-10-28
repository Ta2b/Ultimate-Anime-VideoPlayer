// Décodage audio par libavcodec, puis conversion au format de la file.
#pragma once

#include <expected>
#include <memory>
#include <vector>

#include "audio/converter.h"
#include "audio/packet.h"
#include "audio/pcm.h"
#include "core/types.h"

struct AVCodecContext;
struct AVFrame;
struct AVPacket;

namespace ajp::audio {

class AudioDecoder {
 public:
  // Refuse un codec sans décodeur (AudioCodecUnavailable) ou une sortie hors bornes.
  static std::expected<AudioDecoder, Refusal> open(const AudioCodecInfo& codec, PcmFormat output);

  AudioDecoder(AudioDecoder&&) noexcept;
  AudioDecoder& operator=(AudioDecoder&&) noexcept;
  ~AudioDecoder();

  // Décode un paquet ; rend zéro, un ou plusieurs blocs au format de sortie.
  std::expected<std::vector<PcmBlock>, Refusal> decode(const AudioPacket& packet);
  // Fin de flux : rend ce que le décodeur et le rééchantillonneur retenaient.
  std::expected<std::vector<PcmBlock>, Refusal> drain();
  // Seek : oublie les paquets en cours et l'état du rééchantillonneur.
  void flush();
  PcmFormat output() const { return converter_.output(); }

 private:
  struct Free {
    void operator()(AVCodecContext* context) const;
    void operator()(AVPacket* packet) const;
    void operator()(AVFrame* frame) const;
  };

  explicit AudioDecoder(PcmFormat output);
  std::expected<std::vector<PcmBlock>, Refusal> receive();

  std::unique_ptr<AVCodecContext, Free> context_;
  std::unique_ptr<AVPacket, Free> packet_;
  std::unique_ptr<AVFrame, Free> frame_;
  PcmConverter converter_;
};

}  // namespace ajp::audio
