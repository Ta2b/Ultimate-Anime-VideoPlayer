// Les chapitres de Matroska que FFmpeg aplatit sans le dire (A14-03) : une édition ordonnée
// (EditionFlagOrdered, l'ordre de lecture n'est pas celui du fichier) ou des chapitres liés à d'autres
// segments (ChapterSegmentUID : générique dans un fichier à part, sorties BD de fansub 2008 à 2014). Le
// lecteur ne les joue pas ; il les reconnaît pour le dire.
#pragma once

#include <cstdint>
#include <filesystem>

namespace ajp::media {

struct ChapterLayout {
  bool ordered = false;  // une édition à EditionFlagOrdered = 1
  bool linked = false;   // un chapitre qui renvoie à un autre segment
  bool plain() const { return !ordered && !linked; }
};

// Lit l'élément Chapters du fichier (par le SeekHead, ou en parcourant le segment jusqu'au premier Cluster).
// Un fichier qui n'est pas du Matroska, illisible ou sans chapitres : une disposition simple.
ChapterLayout read_chapter_layout(const std::filesystem::path& path);

}  // namespace ajp::media
