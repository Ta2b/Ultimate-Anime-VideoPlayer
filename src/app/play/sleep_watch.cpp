#include "app/play/sleep_watch.h"

#include <algorithm>
#include <atomic>
#include <chrono>

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
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
  if (gap <= kSleepGapUs) return std::nullopt;
  gap_steady_us_ = steady_us - *before;
  gap_aware_us_ = aware_us - aware_before;
  gap_from_aware_ = aware_before;
  return gap;
}

GapKind SleepWatch::last_kind(std::int64_t window_event_aware_us) const {
#if defined(_WIN32)
  constexpr bool kClocksSplit = false;
#else
  constexpr bool kClocksSplit = true;
#endif
  const bool held = window_event_aware_us > gap_from_aware_ + kHeldAfterUs &&
                    window_event_aware_us <= gap_from_aware_ + gap_aware_us_;
  return gap_kind(gap_steady_us_, gap_aware_us_, held, kClocksSplit);
}

namespace {

std::atomic<std::int64_t> g_window_event_us{0};

bool on_window_event(void* /*context*/, SDL_Event* event) {
  switch (event->type) {
    case SDL_EVENT_WINDOW_MOVED:
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_EXPOSED:
      g_window_event_us.store(suspend_aware_now_us(), std::memory_order_relaxed);
      break;
    default:
      break;
  }
  return true;
}

}  // namespace

WindowEvents::WindowEvents() {
  g_window_event_us.store(0, std::memory_order_relaxed);
  (void)SDL_AddEventWatch(&on_window_event, nullptr);
}

WindowEvents::~WindowEvents() {
  SDL_RemoveEventWatch(&on_window_event, nullptr);
}

std::int64_t WindowEvents::last_aware_us() const {
  return g_window_event_us.load(std::memory_order_relaxed);
}

void ScreenSaverHold::want(bool awake) {
  if (awake_ == awake) return;
  awake_ = awake;
  const bool done = awake ? SDL_DisableScreenSaver() : SDL_EnableScreenSaver();
  if (!done) log_channel("app")->warn("économiseur d'écran : {}", SDL_GetError());
}

}  // namespace ajp::app::play
