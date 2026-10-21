// Couleurs des scripts ASS comme VSFilter les montrait. VSFilter dessinait sur
// l'image YCbCr en convertissant les couleurs du script par BT.601 : sur une
// vidéo BT.709, le lecteur les relit par BT.709, et les typesettings ont été
// réglés à l'œil sur ce résultat. Comportement de référence, réécrit et non
// repris : the-database-mpv sub/sd_ass.c mangle_colors, en mode « basic »
// (sub-ass-vsfilter-color-compat=basic, défaut de mpv et du lecteur actuel).
// Le mode basic ne corrige que ce cas : script TV.601, ou sans en-tête
// « YCbCr Matrix », sur une vidéo BT.709 en plage limitée.
#pragma once

#include <cstdint>
#include <optional>

#include "core/frame.h"
#include "subs/subtitle_frame.h"

namespace ajp::subs {

// En-tête « YCbCr Matrix » du script, tel que libass le lit.
enum class ScriptMatrix : std::uint8_t {
  Absent,   // pas d'en-tête : VSFilter supposait TV.601
  None,     // « None » : couleurs telles quelles (pistes texte converties)
  Bt601Tv,  // « TV.601 »
  Other,    // TV.709, PC.601, illisible… : rien à corriger en mode basic
};

// Matrice et plage de la vidéo source, sur laquelle les sous-titres se posent.
struct VideoMatrix {
  ColorMatrix matrix = ColorMatrix::Bt709;
  ColorRange range = ColorRange::Limited;
  friend bool operator==(const VideoMatrix&, const VideoMatrix&) = default;
};

// true : ce script, sur cette vidéo, était montré par VSFilter avec des couleurs
// converties. Une vidéo inconnue (nullopt) laisse les couleurs du script.
bool vsfilter_converts(ScriptMatrix script, const std::optional<VideoMatrix>& video);

// RGB 8 bits en YCbCr BT.601 limité sur 8 bits, relu en RGB par BT.709 limité,
// chaque étape arrondie au plus proche et bornée à [0, 255]. L'alpha ne change pas.
Rgba8 vsfilter_bt601_on_bt709(Rgba8 color);

}  // namespace ajp::subs
