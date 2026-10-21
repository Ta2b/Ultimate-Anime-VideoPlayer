// Conversion des trames décodées vers le format de la file, par libswresample.
//
// Le format d'entrée peut changer en cours de flux (fréquence, canaux, type
// d'échantillon) : le convertisseur rend d'abord ce que l'ancien
// rééchantillonneur retenait, puis se reconfigure, sans trou dans le temps média.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "audio/pcm.h"
#include "core/types.h"

struct AVFrame;
struct SwrContext;

namespace ajp::audio {

// Masque AV_CH_* dont l'ordre natif est celui que SDL3 attend pour ce nombre de canaux.
std::uint64_t sdl_channel_mask(int channels);

class PcmConverter {
 public:
  explicit PcmConverter(PcmFormat output);
  PcmConverter(PcmConverter&&) noexcept;
  PcmConverter& operator=(PcmConverter&&) noexcept;
  ~PcmConverter();

  // pts_us : temps média de la première trame de frame ; absent, le bloc suit le précédent.
  std::expected<PcmBlock, Refusal> convert(const AVFrame& frame, std::optional<std::int64_t> pts_us);
  // Rend les trames retenues par le rééchantillonneur, en fin de flux.
  PcmBlock drain();
  // Oublie tout état, au seek.
  void reset();
  PcmFormat output() const { return output_; }

 private:
  struct SwrFree {
    void operator()(SwrContext* swr) const;
  };

  std::expected<void, Refusal> configure(const AVFrame& frame);
  int append(PcmBlock& block, const AVFrame* frame);

  PcmFormat output_;
  std::unique_ptr<SwrContext, SwrFree> swr_;
  std::string input_;  // format, fréquence et disposition de l'entrée en cours
  std::optional<std::int64_t> next_pts_us_;
};

}  // namespace ajp::audio
