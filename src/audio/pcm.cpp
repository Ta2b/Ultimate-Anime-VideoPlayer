#include "audio/pcm.h"

#include <algorithm>
#include <format>
#include <limits>

extern "C" {
#include <libavutil/common.h>
#include <libavutil/mathematics.h>
}

namespace kirari::audio {
namespace {

std::int64_t scale_time(std::int64_t value, std::int64_t num, std::int64_t den) {
  const auto result = av_rescale(value, num, den);
  if (result != std::numeric_limits<std::int64_t>::min() || value < 0) return result;
  return std::numeric_limits<std::int64_t>::max();
}

}  // namespace

std::expected<PcmFormat, Refusal> checked(PcmFormat format) {
  if (format.channels < limits::kMinChannels || format.channels > limits::kMaxChannels) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("audio.channels = {}, attendu de {} à {}", format.channels,
                                               limits::kMinChannels, limits::kMaxChannels)});
  }
  if (format.sample_rate < limits::kMinSampleRate || format.sample_rate > limits::kMaxSampleRate) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("audio.sample_rate = {}, attendu de {} à {}", format.sample_rate,
                            limits::kMinSampleRate, limits::kMaxSampleRate)});
  }
  return format;
}

std::int64_t frames_to_us(std::int64_t frames, int sample_rate) {
  return sample_rate > 0 ? scale_time(frames, 1'000'000, sample_rate) : 0;
}

std::int64_t us_to_frames(std::int64_t us, int sample_rate) {
  return sample_rate > 0 ? scale_time(us, sample_rate, 1'000'000) : 0;
}

void PcmBlock::drop_front(std::int64_t n, int sample_rate) {
  const std::int64_t dropped = std::clamp<std::int64_t>(n, 0, frames());
  samples.erase(samples.begin(), samples.begin() + dropped * channels);
  pts_us = av_sat_add64(pts_us, frames_to_us(dropped, sample_rate));
}

void PcmBlock::drop_before(std::int64_t media_us, int sample_rate) {
  if (media_us <= pts_us) return;
  // Un écart plus long que le bloc le retire en entier, sans convertir l'écart
  // en trames : un horodatage aberrant ne déborde pas.
  const auto distance = av_sat_sub64(media_us, pts_us);
  drop_front(distance >= frames_to_us(frames(), sample_rate) ? frames() : us_to_frames(distance, sample_rate),
             sample_rate);
}

}  // namespace kirari::audio
