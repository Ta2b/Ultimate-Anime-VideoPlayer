// Règles reprises de the-database-mpv player/loadfile.c:561-604 (compare_track)
// et 653-733 (select_default_track), réduites aux pistes internes d'un fichier
// et aux options par défaut de mpv (subs-fallback=default, subs-fallback-forced=yes).
// Comportement fixé par tests/unit/media/selection_test.cpp.
#include "media/selection.h"

#include <tuple>

#include "media/language.h"
#include "media/track_language.h"

namespace ajp::media {
namespace {

// Clé d'ordre : plus grande = préférée. L'index le plus bas gagne les égalités.
using Rank = std::tuple<int, bool, bool, int>;

template <class RankOf>
std::optional<int> best_track(std::span<const TrackInfo> tracks, TrackKind kind, RankOf rank_of) {
  std::optional<int> best;
  std::optional<Rank> best_rank;
  for (const TrackInfo& track : tracks) {
    if (track.kind != kind) continue;
    const std::optional<Rank> rank = rank_of(track);
    if (!rank) continue;
    if (!best_rank || *rank > *best_rank) {
      best = track.index;
      best_rank = rank;
    }
  }
  return best;
}

const TrackInfo* find_track(std::span<const TrackInfo> tracks, std::optional<int> index) {
  if (!index) return nullptr;
  for (const TrackInfo& track : tracks) {
    if (track.index == *index) return &track;
  }
  return nullptr;
}

}  // namespace

TrackChoice choose_tracks(std::span<const TrackInfo> tracks, const LanguagePriorities& priorities) {
  TrackChoice choice;
  choice.video = best_track(tracks, TrackKind::Video, [](const TrackInfo& t) -> std::optional<Rank> {
    if (t.attached_picture) return std::nullopt;
    return Rank{0, t.is_default, false, -t.index};
  });

  choice.audio = best_track(tracks, TrackKind::Audio, [&](const TrackInfo& t) -> std::optional<Rank> {
    return Rank{language_score(priorities.audio, track_language(t)), t.is_default, false, -t.index};
  });

  choice.subtitle = best_track(tracks, TrackKind::Subtitle, [&](const TrackInfo& t) -> std::optional<Rank> {
    const int score = language_score(priorities.subtitles, track_language(t));
    if (score > 0) return Rank{score, !t.forced, t.is_default, -t.index};
    return std::nullopt;
  });
  if (choice.subtitle) return choice;

  // Aucune langue voulue : les panneaux forcés dans la langue de l'audio, sinon le défaut.
  const TrackInfo* audio = find_track(tracks, choice.audio);
  const std::string audio_language = audio != nullptr ? track_language(*audio) : std::string{};
  const std::vector<std::string> audio_only{audio_language};
  choice.subtitle = best_track(tracks, TrackKind::Subtitle, [&](const TrackInfo& t) -> std::optional<Rank> {
    const bool forced_match =
        t.forced && !audio_language.empty() && language_score(audio_only, track_language(t)) > 0;
    if (!forced_match && !t.is_default) return std::nullopt;
    return Rank{0, forced_match, t.is_default, -t.index};
  });
  return choice;
}

}  // namespace ajp::media
