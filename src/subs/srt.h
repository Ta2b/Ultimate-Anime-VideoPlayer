// Sous-titres texte (fichiers SRT, paquets texte UTF-8 de Matroska) convertis
// en ASS, pour que libass reste le seul moteur de rendu. Fonctions pures.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::subs {

struct SrtCue {
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
  std::string text;  // texte SRT d'origine, lignes séparées par '\n'
};

// Heures au plus dans un horodatage SRT.
inline constexpr int kSrtMaxHours = 99;

// En-tête ASS des pistes texte : un style « Default » sur 384×288, les valeurs
// par défaut de FFmpeg pour les sous-titres convertis.
std::string_view text_track_header();

// Découpe un fichier SRT en UTF-8 (BOM et fins de ligne CRLF acceptés).
// Refus SubtitleFormat : ligne inattendue, horodatage mal formé ou hors plage,
// fin avant début ; le détail donne le numéro de ligne.
std::expected<std::vector<SrtCue>, Refusal> parse_srt(std::string_view file);

// Texte SRT en texte d'événement ASS : <i> <b> <u> <s> et <font color face size>
// deviennent des balises ASS, '\n' devient \N. Les blocs {…} ASS, fréquents
// ({\an8}), passent tels quels ; une balise inconnue reste du texte.
std::string srt_text_to_ass(std::string_view text);

}  // namespace ajp::subs
