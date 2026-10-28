// File d'un périphérique de sortie : ce que AudioOutput attend de SDL3.
// L'interface sépare le périphérique du reste ; les tests la simulent pour
// rejouer une consommation exacte, au microseconde près.
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "audio/pcm.h"
#include "core/types.h"

namespace ajp::audio {

// Mesure de la file : trames consommées par le périphérique à l'instant hôte host_us.
struct DeviceReading {
  std::int64_t consumed_frames = 0;
  std::int64_t host_us = 0;
};

// Format voulu pour la file ; 0 garde le format préféré du périphérique.
struct OutputRequest {
  int sample_rate = 0;
  int channels = 0;
  std::string device;  // nom de l'appareil de sortie (devices.h) ; vide ou disparu : celui du système
};

class AudioSink {
 public:
  virtual ~AudioSink() = default;
  // Format d'entrée de la file : float32 entrelacé, canaux dans l'ordre de SDL3.
  virtual PcmFormat format() const = 0;
  virtual std::expected<void, Refusal> put(std::span<const float> samples) = 0;
  // Trames poussées depuis l'ouverture ; ne diminue jamais, même au vidage.
  virtual std::int64_t pushed_frames() const = 0;
  // Trames consommées maintenant : poussées moins encore en file.
  virtual DeviceReading reading() const = 0;
  // Tampon du périphérique, en trames d'entrée : ce qui est consommé mais pas encore entendu.
  virtual std::int64_t latency_frames() const = 0;
  virtual void set_paused(bool paused) = 0;
  virtual void set_gain(float gain) = 0;
  virtual float gain() const = 0;
  virtual void set_ratio(double ratio) = 0;
  virtual double ratio() const = 0;
  // Jette tout ce qui est en file ; les trames jetées comptent comme consommées.
  // Rend les trames réellement consommées par le périphérique à cet instant.
  virtual std::int64_t clear() = 0;
  // Appelé sur le thread du périphérique, chaque fois qu'il demande des trames,
  // avec la mesure exacte de cet instant. Ne doit pas bloquer.
  virtual void on_demand(std::function<void(const DeviceReading&)> callback) = 0;
};

// Ouvre le périphérique de sortie par défaut avec SDL3 (sous-système audio compris).
std::expected<std::unique_ptr<AudioSink>, Refusal> open_sdl_sink(const OutputRequest& request);

}  // namespace ajp::audio
