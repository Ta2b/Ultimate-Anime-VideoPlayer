// Sortie audio : la file du périphérique (SDL3), et le temps média de ce qu'on entend.
//
// Position jouée = trames poussées - trames encore en file - tampon du
// périphérique, traduite en temps média par la Timeline, plus le délai audio.
// La sortie garde une copie de l'audio pas encore consommé : un recalage le
// reprend (take_back) au lieu de le perdre.
// Possédée par le thread audio, qui seul l'appelle ; seul le rappel posé par
// on_demand tourne sur le thread du périphérique.
#pragma once

#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "audio/pcm.h"
#include "audio/sink.h"
#include "audio/timeline.h"
#include "core/types.h"

namespace ajp::audio {

class AudioOutput {
 public:
  explicit AudioOutput(std::unique_ptr<AudioSink> sink);
  // Périphérique de sortie par défaut, ouvert en pause.
  static std::expected<std::unique_ptr<AudioOutput>, Refusal> open_sdl(const OutputRequest& request = {});

  PcmFormat format() const { return format_; }

  // Pousse des trames entrelacées dont la première a le temps média pts_us.
  std::expected<void, Refusal> push(std::int64_t pts_us, std::span<const float> samples);
  // Pousse un silence de calage : il n'a pas de temps média.
  std::expected<void, Refusal> push_silence(std::int64_t frames);
  std::int64_t pushed_frames() const { return sink_->pushed_frames(); }
  std::int64_t queued_us() const;
  std::int64_t latency_us() const;
  DeviceReading reading() const { return sink_->reading(); }
  // Temps média entendu au moment de la mesure, délai compris ; absent si ce
  // qui est entendu précède le dernier vidage ou n'est qu'un silence de calage.
  std::optional<std::int64_t> position(const DeviceReading& reading);

  void pause();
  void resume();
  bool paused() const { return paused_; }
  std::expected<void, Refusal> set_volume(float gain);  // de 0 à 2
  float volume() const { return sink_->gain(); }
  // Délai audio : > 0, le son est entendu plus tard que l'image.
  std::expected<void, Refusal> set_delay(std::int64_t delay_us);
  std::int64_t delay_us() const { return delay_us_; }
  // Vitesse de lecture de la file, à 0,125 % près de 1 (écran maître).
  std::expected<void, Refusal> set_ratio(double ratio);
  double ratio() const { return ratio_; }
  // Vide la file au seek ; ce qui était en file ne sera jamais entendu.
  void clear();
  // Vide la file et rend l'audio qu'elle n'avait pas encore consommé, à repousser.
  std::vector<PcmBlock> take_back();
  void on_demand(std::function<void(const DeviceReading&)> callback);

 private:
  // Copie d'un bloc poussé, de l'indice start de la file.
  struct Unplayed {
    std::int64_t start = 0;
    PcmBlock block;
  };
  void forget_consumed(std::int64_t consumed);

  std::unique_ptr<AudioSink> sink_;
  PcmFormat format_;
  Timeline timeline_;
  std::deque<Unplayed> unplayed_;
  std::int64_t delay_us_ = 0;
  double ratio_ = 1.0;
  bool paused_ = true;
};

}  // namespace ajp::audio
