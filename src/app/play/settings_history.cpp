#include "app/play/settings_history.h"

#include <utility>

namespace ajp::app::play {
namespace {

constexpr std::size_t kMaxUndo = 100;

void take(PlaySettings& s, const ExternalChange& changed, const PlaySettings& from) {
  if (changed.volume) s.volume = from.volume;
  if (changed.audio_delay) s.audio_delay_ms = from.audio_delay_ms;
  if (changed.subtitle_delay) s.subtitle_delay_ms = from.subtitle_delay_ms;
  if (changed.stats) s.stats = from.stats;
}

}  // namespace

ExternalChange external_changes(const PlaySettings& seen, const PlaySettings& running,
                                const PlaySettings& draft) {
  return {.volume = running.volume != seen.volume && running.volume != draft.volume,
          .audio_delay =
              running.audio_delay_ms != seen.audio_delay_ms && running.audio_delay_ms != draft.audio_delay_ms,
          .subtitle_delay = running.subtitle_delay_ms != seen.subtitle_delay_ms &&
                            running.subtitle_delay_ms != draft.subtitle_delay_ms,
          .stats = running.stats != seen.stats && running.stats != draft.stats};
}

void SettingsHistory::reset(const PlaySettings& settled) {
  stable_ = settled;
  undo_.clear();
  redo_.clear();
}

void SettingsHistory::settle(const PlaySettings& draft) {
  if (draft == stable_) return;
  undo_.push_back(stable_);
  if (undo_.size() > kMaxUndo) undo_.erase(undo_.begin());
  redo_.clear();
  stable_ = draft;
}

PlaySettings SettingsHistory::undo(const PlaySettings& draft) {
  if (undo_.empty()) return draft;
  redo_.push_back(draft);
  PlaySettings back = std::move(undo_.back());
  undo_.pop_back();
  stable_ = back;
  return back;
}

PlaySettings SettingsHistory::redo(const PlaySettings& draft) {
  if (redo_.empty()) return draft;
  undo_.push_back(draft);
  PlaySettings forward = std::move(redo_.back());
  redo_.pop_back();
  stable_ = forward;
  return forward;
}

void SettingsHistory::adopt(const ExternalChange& changed, const PlaySettings& running) {
  if (!changed.any()) return;
  take(stable_, changed, running);
  for (PlaySettings& s : undo_) take(s, changed, running);
  for (PlaySettings& s : redo_) take(s, changed, running);
}

}  // namespace ajp::app::play
