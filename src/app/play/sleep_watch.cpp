#include "app/play/sleep_watch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <utility>

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "core/clock.h"
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

std::int64_t sleep_unaware_now_us() {
#if defined(_WIN32)
  ULONGLONG ticks = 0;  // en 100 ns
  if (QueryUnbiasedInterruptTime(&ticks) != FALSE) return static_cast<std::int64_t>(ticks / 10);
#endif
  return host_now_us();
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
  constexpr bool kClocksSplit = true;  // sleep_unaware_now_us et suspend_aware_now_us
  const bool held = window_event_aware_us > gap_from_aware_ + kHeldAfterUs &&
                    window_event_aware_us <= gap_from_aware_ + gap_aware_us_;
  return gap_kind(gap_steady_us_, gap_aware_us_, held, kClocksSplit);
}

namespace {

bool on_window_event(void* context, SDL_Event* event) {
  switch (event->type) {
    case SDL_EVENT_WINDOW_MOVED:
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_EXPOSED:
      static_cast<std::atomic<std::int64_t>*>(context)->store(suspend_aware_now_us(),
                                                              std::memory_order_relaxed);
      break;
    default:
      break;
  }
  return true;
}

}  // namespace

WindowEvents::WindowEvents() {
  (void)SDL_AddEventWatch(&on_window_event, &aware_us_);
}

WindowEvents::~WindowEvents() {
  SDL_RemoveEventWatch(&on_window_event, &aware_us_);
}

std::int64_t WindowEvents::last_aware_us() const {
  return aware_us_.load(std::memory_order_relaxed);
}

AwakeWhile::AwakeWhile() {
  if (!SDL_DisableScreenSaver()) log_channel("app")->warn("économiseur d'écran : {}", SDL_GetError());
}

AwakeWhile::~AwakeWhile() {
  if (!SDL_EnableScreenSaver()) log_channel("app")->warn("économiseur d'écran : {}", SDL_GetError());
}

void ScreenSaverHold::want(bool awake) {
  if (awake_ == awake) return;
  // L'état n'est retenu qu'une fois tenu (C11-10) : un échec (bus D-Bus pas prêt juste après la connexion à
  // KDE) se retente au tour suivant, au lieu de laisser l'écran se verrouiller en pleine vidéo. Dit une fois.
  if (const bool done = awake ? SDL_DisableScreenSaver() : SDL_EnableScreenSaver(); done) {
    awake_ = awake;
    failed_ = false;
  } else if (!std::exchange(failed_, true)) {
    log_channel("app")->warn("économiseur d'écran : {} (nouvel essai à chaque tour)", SDL_GetError());
  }
}

}  // namespace ajp::app::play
