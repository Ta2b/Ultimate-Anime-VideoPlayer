#include "app/play/after_gap.h"

#include <cstdint>
#include <utility>

#include "app/play/frame_stats.h"
#include "app/play/i18n.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {

AfterGap after_gap(Session& session, const ChainEngines& c, PlayerUi& ui, SleepWatch& watch,
                   const WindowEvents& window, bool& announced) {
  const std::optional<std::int64_t> gap_us = watch.turn(sleep_unaware_now_us(), suspend_aware_now_us());
  if (!gap_us) return {};
  std::optional<std::expected<void, Refusal>> health;
  if (c.backend) health = c.backend->kernels()->health();
  return after_gap_seen(session, ui.controls(), *gap_us,
                        watch.last_kind(window.last_aware_us()) == GapKind::Sleep, health, announced,
                        host_now_us());
}

AfterGap after_gap_seen(Session& session, Controls& controls, std::int64_t gap_us, bool sleep_seen,
                        const std::optional<std::expected<void, Refusal>>& health, bool& announced,
                        std::int64_t now_us) {
  const auto log = log_channel("app");
  const std::int64_t at = session.view().position_us;
  AfterGap out{.reopen = std::exchange(announced, false)};
  const bool sleep = out.reopen || sleep_seen;
  log->warn("trou de {:.1f} s ({}) : recalage à {}", static_cast<double>(gap_us) / 1e6,
            out.reopen ? "veille annoncée, pause"
            : sleep    ? "veille, pause"
                       : "gel de la boucle, sans pause",
            format_position(at));
  if (health && !*health) {
    if (health->error().code == RefusalCode::DeviceLost) return {.lost = health->error()};
    log->warn("GPU au réveil : {} ; la chaîne se rouvre", health->error().detail);
    out.reopen = true;
  }
  if (sleep) session.command(cmd::Pause{});
  if (at != kNoPts && !out.reopen) session.command(cmd::Seek{.target_us = at, .exact = true});
  if (sleep) controls.notify(i18n::ui_tr("Paused after sleep"), now_us);
  return out;
}

}  // namespace ajp::app::play
