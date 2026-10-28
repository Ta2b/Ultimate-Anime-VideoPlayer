// Données de test fabriquées dans le test : un sinus, encodé par un encodeur
// natif de FFmpeg ou rangé dans une AVFrame. Aucun fichier binaire.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>
}

#include "audio/packet.h"
#include "audio/pcm.h"

namespace ajp::audio::test {

struct SineSpec {
  std::string_view encoder = "pcm_s16le";
  int sample_rate = 48'000;
  int channels = 2;
  double hz = 1'000.0;
  double amplitude = 0.5;
  double seconds = 0.5;
  int frame_size = 1'024;     // pour les encodeurs sans taille imposée (PCM)
  std::int64_t start_us = 0;  // temps média du premier échantillon
  std::uint32_t epoch = 0;
};

struct EncodedClip {
  AudioCodecInfo codec;
  std::vector<AudioPacket> packets;
};

EncodedClip encode_sine(const SineSpec& spec);

struct FrameFree {
  void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;

// frames échantillons du sinus à partir de l'échantillon first, même signal sur chaque canal.
FramePtr sine_frame(AVSampleFormat format, int sample_rate, int channels, int frames, double hz,
                    double amplitude, std::int64_t first);

std::vector<float> concat(const std::vector<PcmBlock>& blocks);
std::vector<float> channel(std::span<const float> interleaved, int channels, int index);
// Amplitude de la composante à hz : un sinus d'amplitude a rend environ a.
double tone_amplitude(std::span<const float> mono, int sample_rate, double hz);
double peak(std::span<const float> mono);

}  // namespace ajp::audio::test
