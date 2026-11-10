#include "app/play/sleep_watch.h"

#include <algorithm>
#include <chrono>

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_video.h>

#include "core/log.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <ctime>
#endif

namespace ajp::app::play {

std::int64_t suspend_aware_now_us() {
#if defined(_WIN32)
  return static_cast<std::int64_t>(GetTickCount64()) * 1000;  // compte la veille, au ms près
#elif defined(__APPLE__)
  mach_timebase_info_data_t base{};
  mach_timebase_info(&base);
  const std::uint64_t ticks = mach_continuous_time();
  return static_cast<std::int64_t>(ticks * base.numer / base.denom / 1000);
#else
  timespec now{};
  if (clock_gettime(CLOCK_BOOTTIME, &now) != 0)
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  return std::int64_t{now.tv_sec} * 1'000'000 + now.tv_nsec / 1000;
#endif
}

std::optional<std::int64_t> SleepWatch::turn(std::int64_t steady_us, std::int64_t aware_us) {
  const std::optional<std::int64_t> before = steady_us_;
  const std::int64_t aware_before = aware_us_;
  steady_us_ = steady_us;
  aware_us_ = aware_us;
  if (!before) return std::nullopt;
  const std::int64_t gap = (std::max)(steady_us - *before, aware_us - aware_before);  // windows.h : max
  return gap > kSleepGapUs ? std::optional(gap) : std::nullopt;
}

void ScreenSaverHold::want(bool awake) {
  if (awake_ == awake) return;
  awake_ = awake;
  const bool done = awake ? SDL_DisableScreenSaver() : SDL_EnableScreenSaver();
  if (!done) log_channel("app")->warn("économiseur d'écran : {}", SDL_GetError());
}

}  // namespace ajp::app::play
