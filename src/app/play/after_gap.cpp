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
  const std::optional<std::int64_t> gap_us = watch.turn(host_now_us(), suspend_aware_now_us());
  if (!gap_us) return {};
  const auto log = log_channel("app");
  const std::int64_t at = session.view().position_us;
  const bool was_announced = std::exchange(announced, false);
  const bool sleep = was_announced || watch.last_kind(window.last_aware_us()) == GapKind::Sleep;
  // C31-08 : une veille vue au réveil seulement (verrou de logind refusé, signal manqué) rouvre la chaîne
  // comme une veille annoncée : ses tampons perdus ne disent rien non plus.
  AfterGap out{.reopen = sleep, .paused = sleep};
  log->warn("trou de {:.1f} s ({}) : recalage à {}", static_cast<double>(*gap_us) / 1e6,
            was_announced ? "veille annoncée, pause"
            : sleep       ? "veille, pause, chaîne rouverte"
                          : "gel de la boucle, sans pause",
            format_position(at));
  if (c.backend) {
    if (auto healthy = c.backend->kernels()->health(); !healthy) {
      // C31-01 : la relance garde la pause de la veille.
      if (healthy.error().code == RefusalCode::DeviceLost)
        return {.lost = std::move(healthy.error()), .paused = sleep};
      log->warn("GPU au réveil : {} ; la chaîne se rouvre", healthy.error().detail);
      out.reopen = true;
    }
  }
  if (sleep) session.command(cmd::Pause{});
  if (at != kNoPts && !out.reopen) session.command(cmd::Seek{.target_us = at, .exact = true});
  if (sleep) ui.controls().notify(i18n::ui_tr("Paused after sleep"), host_now_us());
  return out;
}

}  // namespace ajp::app::play
