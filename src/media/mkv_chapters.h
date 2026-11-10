// Les chapitres de Matroska que FFmpeg aplatit sans le dire (A14-03) : une édition ordonnée
// (EditionFlagOrdered, l'ordre de lecture n'est pas celui du fichier) ou des chapitres liés à d'autres
// segments (ChapterSegmentUID : générique dans un fichier à part, sorties BD de fansub 2008 à 2014). Le
// lecteur joue les parties d'une édition ordonnée qui sont dans le fichier ; il reconnaît les autres pour le
// dire.
#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

namespace ajp::media {

// Une partie d'une édition ordonnée, en µs sur la ligne de temps du segment (pas encore celle du Demuxer).
struct ChapterPart {
  static constexpr std::int64_t kToEnd = std::numeric_limits<std::int64_t>::max();
  std::int64_t start_us = 0;
  std::int64_t end_us = kToEnd;  // sans ChapterTimeEnd : jusqu'à la fin du fichier
};

struct ChapterLayout {
  bool ordered = false;  // une édition à EditionFlagOrdered = 1
  bool linked = false;   // un chapitre qui renvoie à un autre segment
  // L'édition jouée (celle à EditionFlagDefault, sinon la première), si elle est ordonnée : ses chapitres
  // activés de ce fichier, dans l'ordre de l'édition. Ceux d'un autre segment n'y sont pas.
  std::vector<ChapterPart> parts;
  bool plain() const { return !ordered && !linked; }
};

// Lit l'élément Chapters du fichier (par le SeekHead, ou en parcourant le segment jusqu'au premier Cluster).
// Un fichier qui n'est pas du Matroska, illisible ou sans chapitres : une disposition simple.
ChapterLayout read_chapter_layout(const std::filesystem::path& path);

}  // namespace ajp::media
