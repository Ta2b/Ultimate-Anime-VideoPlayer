// Règles reprises de the-database-mpv player/loadfile.c:561-604 (compare_track)
// et 653-733 (select_default_track), réduites aux pistes internes d'un fichier
// et aux options par défaut de mpv (subs-fallback=default, subs-fallback-forced=yes).
// Comportement fixé par tests/unit/media/selection_test.cpp.
#include "media/selection.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>

#include "media/language.h"
#include "media/track_language.h"

namespace ajp::media {
namespace {

// Clé d'ordre : plus grande = préférée. L'index le plus bas gagne les égalités.
// (langue, piste complète, critère propre au type, drapeau défaut, -index)
using Rank = std::tuple<int, bool, bool, bool, int>;

// Une piste de panneaux, de chansons seules, de commentaire ou pour malentendants n'est pas la
// piste de dialogue : elle ne passe qu'après une piste complète de la même langue.
bool is_full_dialogue(const TrackInfo& track) {
  std::string word;
  const auto special = [&] {
    static constexpr std::string_view kWords[] = {"sign",        "signs", "song", "songs",  "commentary",
                                                  "commentaire", "sdh",   "cc",   "hearing"};
    return std::ranges::find(kWords, word) != std::end(kWords);
  };
  bool found = false;
  for (const char c : track.title + ' ') {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    } else {
      found = found || (!word.empty() && special());
      word.clear();
    }
  }
  return !found;
}

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

bool subtitle_readable(std::string_view codec) {
  // Codec inconnu (vide) : lisible, le refus éventuel vient plus loin. Les formats image (PGS,
  // VobSub, DVB) et mov_text/WebVTT ne sont pas lus par le module de sous-titres.
  return codec.empty() || codec == "ass" || codec == "ssa" || codec == "subrip" || codec == "srt" ||
         codec == "text";
}

namespace {

// La meilleure piste de sous-titres lisible dans `language` (dialogue complet d'abord), sinon rien.
std::optional<int> subtitle_in(std::span<const TrackInfo> tracks, const std::string& language) {
  const std::vector<std::string> wanted{language};
  return best_track(tracks, TrackKind::Subtitle, [&](const TrackInfo& t) -> std::optional<Rank> {
    if (!subtitle_readable(t.codec)) return std::nullopt;
    const int score = language_score(wanted, track_language(t));
    if (score > 0) return Rank{score, is_full_dialogue(t), !t.forced, t.is_default, -t.index};
    return std::nullopt;
  });
}

std::optional<int> audio_in(std::span<const TrackInfo> tracks, const std::string& language) {
  const std::vector<std::string> wanted{language};
  return best_track(tracks, TrackKind::Audio, [&](const TrackInfo& t) -> std::optional<Rank> {
    const int score = language_score(wanted, track_language(t));
    if (score <= 0) return std::nullopt;
    return Rank{score, true, t.is_default, false, -t.index};
  });
}

}  // namespace

TrackChoice choose_tracks(std::span<const TrackInfo> tracks, const LanguagePriorities& priorities_in) {
  LanguagePriorities priorities = priorities_in;
  TrackChoice choice;
  choice.video = best_track(tracks, TrackKind::Video, [](const TrackInfo& t) -> std::optional<Rank> {
    if (t.attached_picture) return std::nullopt;
    return Rank{0, true, t.is_default, false, -t.index};
  });
  for (const LanguagePreset& preset : priorities.presets) {
    // Un préréglage sans langue d'audio prend la piste que le fichier préfère.
    const std::optional<int> audio = preset.audio.empty()
                                         ? best_track(tracks, TrackKind::Audio,
                                                      [](const TrackInfo& t) -> std::optional<Rank> {
                                                        return Rank{0, true, t.is_default, false, -t.index};
                                                      })
                                         : audio_in(tracks, preset.audio);
    if (!audio) continue;
    std::optional<int> subtitle;
    if (!preset.subtitles.empty()) {
      subtitle = subtitle_in(tracks, preset.subtitles);
      if (!subtitle) continue;
    }
    choice.audio = audio;
    choice.subtitle = subtitle;
    return choice;
  }
  // Aucun préréglage entier : leurs langues, de rang en rang, rejoignent les listes.
  for (const LanguagePreset& preset : priorities.presets) {
    if (!preset.audio.empty()) priorities.audio.push_back(preset.audio);
    if (!preset.subtitles.empty()) priorities.subtitles.push_back(preset.subtitles);
  }
  choice.audio = best_track(tracks, TrackKind::Audio, [&](const TrackInfo& t) -> std::optional<Rank> {
    return Rank{language_score(priorities.audio, track_language(t)), true, t.is_default, false, -t.index};
  });

  choice.subtitle = best_track(tracks, TrackKind::Subtitle, [&](const TrackInfo& t) -> std::optional<Rank> {
    if (!subtitle_readable(t.codec)) return std::nullopt;
    const int score = language_score(priorities.subtitles, track_language(t));
    if (score > 0) return Rank{score, is_full_dialogue(t), !t.forced, t.is_default, -t.index};
    return std::nullopt;
  });
  if (choice.subtitle) return choice;

  // Aucune langue voulue : les panneaux forcés dans la langue de l'audio, sinon le défaut.
  const TrackInfo* audio = find_track(tracks, choice.audio);
  const std::string audio_language = audio != nullptr ? track_language(*audio) : std::string{};
  const std::vector<std::string> audio_only{audio_language};
  choice.subtitle = best_track(tracks, TrackKind::Subtitle, [&](const TrackInfo& t) -> std::optional<Rank> {
    if (!subtitle_readable(t.codec)) return std::nullopt;
    const bool forced_match =
        t.forced && !audio_language.empty() && language_score(audio_only, track_language(t)) > 0;
    if (!forced_match && !t.is_default) return std::nullopt;
    return Rank{0, is_full_dialogue(t), forced_match, t.is_default, -t.index};
  });
  return choice;
}

}  // namespace ajp::media
