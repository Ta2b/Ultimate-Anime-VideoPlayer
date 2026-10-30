// Les réponses de tête (HRIR) livrées avec le lecteur : data/hrir/*.wav, de la collection HeSuVi (licence
// MIT, data/hrir/LICENSE-hesuvi.txt), embarquées octet pour octet. Une réponse se choisit par sa clé («
// bundled:atmos » dans hrir_path), comme un fichier de l'utilisateur.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "audio/hrir.h"

namespace ajp::app::play {

struct BundledHrir {
  std::string_view key;    // atmos, dtshx...
  const char* label;       // clé de traduction, pour le menu
  std::string_view bytes;  // le WAV
};

extern const std::string_view kHrirAtmos;
extern const std::string_view kHrirDtshx;
extern const std::string_view kHrirSbx100;
extern const std::string_view kHrirSbx33;
extern const std::string_view kHrirSonic;

// Toutes les réponses livrées, dans l'ordre du menu.
std::span<const BundledHrir> bundled_hrirs();
// La réponse livrée pour un hrir_path « bundled:<clé> » ; nullptr si le chemin n'en est pas une.
const BundledHrir* find_bundled_hrir(std::string_view hrir_path);

}  // namespace ajp::app::play
