// Texte des codes d'erreur FFmpeg. En-tête interne, inclus par les .cpp seulement.
#pragma once

#include <array>
#include <string>

extern "C" {
#include <libavutil/error.h>
}

namespace ajp::audio {

// av_err2str est une macro C99 (littéral composé), inutilisable en C++.
inline std::string av_error_text(int error) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  av_strerror(error, text.data(), text.size());
  return text.data();
}

}  // namespace ajp::audio
