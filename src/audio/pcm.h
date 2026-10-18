// Échantillons décodés : float32 entrelacé, au format de la file du périphérique.
// Plages des réglages audio : une valeur hors plage est refusée, jamais corrigée.
#pragma once

#include <cstdint>
#include <expected>
#include <iterator>
#include <vector>

#include "core/types.h"

namespace ajp::audio {

namespace limits {
// SDL3 place de 1 à 8 canaux, dans son ordre documenté.
inline constexpr int kMinChannels = 1;
inline constexpr int kMaxChannels = 8;
inline constexpr int kMinSampleRate = 8'000;
inline constexpr int kMaxSampleRate = 384'000;
inline constexpr float kMinVolume = 0.0F;
inline constexpr float kMaxVolume = 2.0F;
inline constexpr std::int64_t kMaxDelayUs = 10'000'000;  // délai audio, dans les deux sens
// Correction de dérive quand l'écran est maître : 0,125 %, comme
// video-sync-max-audio-change de mpv. Couvre le rapport 1,001 des cadences NTSC.
inline constexpr double kMaxDriftCorrection = 0.00125;
}  // namespace limits

// Format de la file : float32 entrelacé, canaux dans l'ordre de SDL3.
struct PcmFormat {
  int sample_rate = 48'000;
  int channels = 2;
  friend bool operator==(const PcmFormat&, const PcmFormat&) = default;
};

// Le format tel quel s'il est dans les bornes, un refus FieldOutOfRange sinon.
std::expected<PcmFormat, Refusal> checked(PcmFormat format);

// Conversions entre trames et microsecondes, arrondies au plus proche.
std::int64_t frames_to_us(std::int64_t frames, int sample_rate);
std::int64_t us_to_frames(std::int64_t us, int sample_rate);

// Un bloc d'échantillons et le temps média de sa première trame.
struct PcmBlock {
  std::int64_t pts_us = 0;
  int channels = 2;
  std::vector<float> samples;  // trames × canaux, entrelacé

  std::int64_t frames() const { return channels > 0 ? std::ssize(samples) / channels : 0; }
  // Retire les n premières trames (toutes si n dépasse) ; le temps média avance d'autant.
  void drop_front(std::int64_t n, int sample_rate);
};

}  // namespace ajp::audio
