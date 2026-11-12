#include "app/play/player_notices.h"

#include "app/play/first_run.h"

namespace ajp::app::play {

i18n::Translated relaunch_notice(const PlayOptions& options, const std::optional<RelaunchTarget>& reopened,
                                 bool video_reopens, std::string& problem) {
  if (options.relaunched != RelaunchedFor::GpuLost &&
      !(reopened && reopened->reason == RelaunchReason::GpuLost))
    return {};
  i18n::Translated note = i18n::ui_tr("The graphics card stopped responding: the player restarted");
  if (video_reopens) return note;
  problem += problem.empty() ? note.str() : " · " + note.str();
  return {};
}

i18n::Translated fallback_notice(std::optional<RefusalCode> passed_over, core::EngineId wanted,
                                 core::EngineId in_use) {
  if (passed_over == RefusalCode::SdkRejected)
    return i18n::ui_tr_format("{} needs a newer graphics driver: {} is used, slower", engine_label(wanted),
                              engine_label(in_use));
  return i18n::ui_tr_format("{} unavailable: {} is used, slower (Settings > General > Engines)",
                            engine_label(wanted), engine_label(in_use));
}

}  // namespace ajp::app::play
