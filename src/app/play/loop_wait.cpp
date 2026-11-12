#include "app/play/loop_wait.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include <SDL3/SDL_events.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

#include "core/clock.h"
#include "core/wait_plan.h"

namespace ajp::app::play {
namespace {

// Une fois, sur le thread qui attend (main) : ses minuteurs ne glissent plus de 50 µs par défaut.
void tighten_timer_slack() {
#ifdef __linux__
  thread_local const bool done = [] {
    (void)prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
    return true;
  }();
  (void)done;
#endif
}

// L'attente absolue précise jusqu'à until_us (horloge hôte). Sous Linux, sleep_until donne clock_nanosleep,
// au slack près (1 ns). Sous Windows, sleep_until finit en Sleep à la milliseconde, même avec
// timeBeginPeriod(1) : le réveil glissait de 1 à 2 ms, un tiers de période à 165 Hz (B12-08). Là, un minuteur
// haute résolution (Windows 10 1803 et après) jusqu'à kSpinUs de l'échéance, puis une attente active.
void sleep_until_precise(std::int64_t until_us) {
#ifdef _WIN32
  constexpr std::int64_t kSpinUs = 200;
  // Fermé à la fin du fil (C33-06).
  struct Timer {
    HANDLE handle =
        CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    Timer() = default;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    ~Timer() {
      if (handle != nullptr) CloseHandle(handle);
    }
  };
  thread_local const Timer owned;
  const HANDLE timer = owned.handle;
  const std::int64_t room_us = until_us - host_now_us() - kSpinUs;
  if (room_us > 0 && timer != nullptr) {
    LARGE_INTEGER due{};
    due.QuadPart = -room_us * 10;  // relatif, en 100 ns
    if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0) != 0)
      (void)WaitForSingleObject(timer, INFINITE);
  } else if (room_us > 2000) {  // minuteur refusé (Windows trop ancien) : Sleep jusqu'à 2 ms de l'échéance
    std::this_thread::sleep_for(std::chrono::microseconds(room_us - 2000));
  }
  while (host_now_us() < until_us) YieldProcessor();
#else
  // host_now_us est steady_clock : la même horloge, une échéance absolue.
  std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::microseconds(until_us)));
#endif
}

}  // namespace

void LoopWake::push(void* context) {
  auto* self = static_cast<LoopWake*>(context);
  if (self->event == 0 || self->pending.exchange(true, std::memory_order_acq_rel)) return;
  SDL_Event wake{};
  wake.type = self->event;
  (void)SDL_PushEvent(&wake);
}

std::uint32_t loop_wake_event() {
  static const std::uint32_t registered = SDL_RegisterEvents(1);
  return registered;
}

void wait_loop(std::int64_t wait_us, bool precise) {
  if (!precise) {
    (void)SDL_WaitEventTimeout(nullptr, static_cast<Sint32>(std::max<std::int64_t>(wait_us / 1000, 1)));
    return;
  }
  tighten_timer_slack();
  const std::int64_t now = host_now_us();
#ifdef _WIN32
  constexpr std::int64_t kMarginUs = core::kWaitMarginWindowsUs;
#else
  constexpr std::int64_t kMarginUs = core::kWaitMarginUs;
#endif
  const core::WaitPlan plan = core::wait_plan(now, now + std::max<std::int64_t>(wait_us, 0), kMarginUs);
  if (plan.coarse_ms > 0 && SDL_WaitEventTimeout(nullptr, static_cast<Sint32>(plan.coarse_ms))) return;
  if (SDL_HasEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST)) return;
  sleep_until_precise(plan.precise_until_us);
}

}  // namespace ajp::app::play
