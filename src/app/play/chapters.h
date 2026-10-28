// Chapitres du fichier (Matroska, MP4) : où l'on est, où mènent les boutons « chapitre
// précédent » et « chapitre suivant » de la barre (et Page précédente / Page suivante, comme
// dans mpv), et ce que le message bref en dit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace ajp::app::play {

struct ChapterMark {
  std::int64_t start_us = 0;
  std::string title;  // vide : sans titre
};

// En deçà de ce temps depuis le début du chapitre courant, « précédent » mène au chapitre
// d'avant ; au-delà, au début du chapitre courant (comme une piste de lecteur audio).
inline constexpr std::int64_t kChapterRestartUs = 3'000'000;

// Le chapitre où tombe position_us (le dernier commencé), ou rien avant le premier.
std::optional<std::size_t> chapter_at(std::span<const ChapterMark> chapters, std::int64_t position_us);

// Le chapitre où mène un pas (+1 suivant, −1 précédent) depuis position_us, ou rien : pas de
// chapitre suivant après le dernier.
std::optional<std::size_t> chapter_step(std::span<const ChapterMark> chapters, std::int64_t position_us,
                                        int direction);

// Le chapitre est-il un générique (opening, ending, intro, outro, crédits) d'après son titre ? « OP »,
// « ED 2 », « Opening », « Générique de fin »… Les titres de Matroska n'ont pas de norme ; sans
// titre, jamais.
bool is_credits_chapter(const ChapterMark& chapter);

// Où mène « passer le générique » depuis position_us : le début du chapitre suivant si l'on est dans
// un générique, la fin de la vidéo si c'est le dernier chapitre ; rien hors d'un générique.
std::optional<std::int64_t> credits_skip_target(std::span<const ChapterMark> chapters,
                                                std::int64_t position_us, std::int64_t duration_us);

// Le saut de repli hors d'un générique reconnu : la durée classique d'un opening d'anime.
inline constexpr std::int64_t kSkipFallbackUs = 85'000'000;

// « Chapitre 3/12 · Opening », ou « Chapitre 3/12 » sans titre.
std::string chapter_label(std::span<const ChapterMark> chapters, std::size_t index);

}  // namespace ajp::app::play
