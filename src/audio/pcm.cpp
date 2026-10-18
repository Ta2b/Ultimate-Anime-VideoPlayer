#include "audio/pcm.h"

#include <algorithm>
#include <format>

namespace ajp::audio {
namespace {

// Division arrondie au plus proche, symétrique autour de zéro ; den > 0.
std::int64_t divide_rounded(std::int64_t num, std::int64_t den) {
  return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
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
  return divide_rounded(frames * 1'000'000, sample_rate);
}

std::int64_t us_to_frames(std::int64_t us, int sample_rate) {
  return divide_rounded(us * sample_rate, 1'000'000);
}

void PcmBlock::drop_front(std::int64_t n, int sample_rate) {
  const std::int64_t dropped = std::clamp<std::int64_t>(n, 0, frames());
  samples.erase(samples.begin(), samples.begin() + dropped * channels);
  pts_us += frames_to_us(dropped, sample_rate);
}

}  // namespace ajp::audio
