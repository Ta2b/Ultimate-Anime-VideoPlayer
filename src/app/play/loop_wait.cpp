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
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002  // Windows 10 1803 et après
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

#ifdef _WIN32
// B18-05 : sleep_until ne descend pas sous le minuteur du système (1 ms au mieux avec timeBeginPeriod, soit
// 15 % d'une période à 144 Hz). Un minuteur à haute résolution suit l'échéance de bien plus près, sans
// attente active. Faux s'il n'existe pas (Windows plus ancien) : sleep_until reste.
bool wait_high_resolution(std::int64_t until_us) {
  thread_local const HANDLE timer =
      CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (timer == nullptr) return false;
  const std::int64_t left_us = until_us - host_now_us();
  if (left_us <= 0) return true;
  LARGE_INTEGER due{};
  due.QuadPart = -left_us * 10;  // relatif, en centaines de nanosecondes
  if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0) == FALSE) return false;
  return WaitForSingleObject(timer, INFINITE) == WAIT_OBJECT_0;
}
#endif

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
  const core::WaitPlan plan = core::wait_plan(now, now + std::max<std::int64_t>(wait_us, 0));
  if (plan.coarse_ms > 0 && SDL_WaitEventTimeout(nullptr, static_cast<Sint32>(plan.coarse_ms))) return;
  if (SDL_HasEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST)) return;
#ifdef _WIN32
  if (wait_high_resolution(plan.precise_until_us)) return;
#endif
  // host_now_us est steady_clock : la même horloge, une échéance absolue.
  std::this_thread::sleep_until(
      std::chrono::steady_clock::time_point(std::chrono::microseconds(plan.precise_until_us)));
}

}  // namespace ajp::app::play
