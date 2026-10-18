// Temps média de chaque trame poussée dans la file du périphérique.
//
// Les indices comptent les trames poussées depuis l'ouverture : ils croissent
// toujours, même après un vidage, si bien qu'une mesure d'avant un vidage
// tombe avant le premier segment et se reconnaît comme périmée.
#pragma once

#include <cstdint>
#include <deque>
#include <optional>

namespace ajp::audio {

class Timeline {
 public:
  explicit Timeline(int sample_rate) : sample_rate_(sample_rate) {}

  // Trames [start, start + frames) : de l'audio qui commence à pts_us, ou,
  // sans pts_us, un silence de calage qui n'a pas de temps média.
  void append(std::int64_t start, std::int64_t frames, std::optional<std::int64_t> pts_us);
  // Temps média de la trame index ; absent si index précède le premier
  // segment connu ou tombe dans un silence.
  std::optional<std::int64_t> pts_at(std::int64_t index) const;
  // Oublie les segments entièrement joués avant index.
  void forget_before(std::int64_t index);
  void clear() { segments_.clear(); }
  bool empty() const { return segments_.empty(); }

 private:
  struct Segment {
    std::int64_t start = 0;
    std::int64_t frames = 0;
    std::optional<std::int64_t> pts_us;
  };

  int sample_rate_;
  std::deque<Segment> segments_;
};

}  // namespace ajp::audio
