#include "present/display_clock.h"

namespace ajp::present {

std::optional<std::int64_t> audio_offset_us(const ClockSnapshot& audio, const ClockSnapshot& display,
                                            std::int64_t host_us) {
  if (!audio.valid || !display.valid || audio.epoch != display.epoch) return std::nullopt;
  return audio.media_at(host_us) - display.media_at(host_us);
}

}  // namespace ajp::present
