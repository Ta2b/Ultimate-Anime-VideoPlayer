#pragma once

// Le texte d'un code AVERROR, un seul pour tous les modules (P5.13). En-tête seul : core ne lie pas FFmpeg,
// les modules qui l'incluent le lient déjà. av_err2str est une macro C99 (littéral composé), inutilisable en
// C++.

#include <array>
#include <cerrno>
#include <format>
#include <string>

extern "C" {
#include <libavutil/error.h>
}

namespace ajp::core {

inline std::string av_error_text(int error) {
  // Les refus qu'on voit le plus à l'ouverture d'un fichier, en français (le reste garde le texte de FFmpeg).
  switch (error) {
    case AVERROR_INVALIDDATA:
      return "format non reconnu ou fichier abîmé (Invalid data found when processing input)";
    case AVERROR(ENOENT):
      return "fichier introuvable (No such file or directory)";
    case AVERROR(EACCES):
      return "accès refusé (Permission denied)";
    case AVERROR(EIO):
      return "fichier tronqué ou illisible (Input/output error)";
    default:
      break;
  }
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  if (av_strerror(error, text.data(), text.size()) < 0) return std::format("erreur FFmpeg {}", error);
  return text.data();
}

}  // namespace ajp::core
