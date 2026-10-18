// Clips partagés par les tests de media.
#pragma once

#include <filesystem>

#include "clip_writer.h"

namespace ajp::media::test {

inline constexpr std::int64_t kFrameUs = 40'000;  // 25 im/s
inline constexpr char kFontBytes[] = "OTTO police factice";

// FFV1 sans perte, 30 images 64x48 à 25 im/s ; audio jpn et eng (défaut) ;
// sous-titres fre complets (défaut) et fre forcés ; une police ; 3 chapitres.
ClipSpec full_clip();
// MPEG-4, un point clé toutes les 10 images, luma uniforme par image ; un audio
// et un sous-titre, pour le seek exact et le thread source.
ClipSpec seek_clip();
// FFV1 10 bits, BT.2020, PQ, siting haut-gauche, HDR10.
ClipSpec hdr_clip();
// MP4, MPEG-4 avec images B, 24000/1001 im/s.
ClipSpec bframes_clip();

// Écrit le clip et échoue le test s'il n'a pas pu l'être.
std::filesystem::path require_clip(const ClipSpec& spec);

}  // namespace ajp::media::test
