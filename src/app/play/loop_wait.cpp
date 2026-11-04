#include "app/play/loop_wait.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include <SDL3/SDL_events.h>

#ifdef __linux__
#include <sys/prctl.h>
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

}  // namespace

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
  // host_now_us est steady_clock : la même horloge, une échéance absolue.
  std::this_thread::sleep_until(
      std::chrono::steady_clock::time_point(std::chrono::microseconds(plan.precise_until_us)));
}

}  // namespace ajp::app::play
