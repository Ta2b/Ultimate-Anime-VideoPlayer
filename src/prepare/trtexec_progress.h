// Progression d'une construction lue dans la sortie de trtexec, pour l'OSD.
//
// trtexec n'imprime pas de pourcentage : chaque ligne repère connue fixe une
// fraction, qui ne recule jamais. Les repères viennent des journaux de trtexec
// 10 et 11 (à vérifier sur trtexec 11.3) ; une ligne inconnue ne change rien.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ajp::prepare {

struct BuildStage {
  float fraction = 0.0F;  // 0 à 1
  std::string stage;      // en français, affiché tel quel
};

class TrtexecProgress {
 public:
  // Rend une étape quand la ligne fait avancer la construction.
  std::optional<BuildStage> feed(std::string_view line);

  bool passed() const { return passed_; }  // « &&&& PASSED » vu
  bool failed() const { return failed_; }  // « &&&& FAILED » vu
  // Dernière ligne d'erreur « [E] », pour le détail d'un refus ; vide sinon.
  const std::string& last_error() const { return last_error_; }

 private:
  float fraction_ = 0.0F;
  bool passed_ = false;
  bool failed_ = false;
  std::string last_error_;
};

}  // namespace ajp::prepare
