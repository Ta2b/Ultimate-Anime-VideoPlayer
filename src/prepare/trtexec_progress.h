// Progression d'une construction lue dans la sortie de trtexec, pour l'OSD.
//
// trtexec n'imprime pas de pourcentage : chaque ligne repère connue fixe une
// fraction, qui ne recule jamais. Les repères viennent des journaux de trtexec
// 10 et 11 (à vérifier sur trtexec 11.3) ; une ligne inconnue ne change rien. Entre « Finished parsing » et
// « Total Activation Memory », trtexec n'écrit rien (il n'est pas lancé en --verbose) : la barre de l'OSD
// avance alors avec le temps attendu (app/play/build_progress.h), pas avec ces repères (C05-08).
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ajp::prepare {

struct BuildStage {
  float fraction = 0.0F;  // 0 à 1
  std::string stage;      // clé anglaise, traduite par l'interface (build_stage_text)
};

class TrtexecProgress {
 public:
  // Rend une étape quand la ligne fait avancer la construction.
  std::optional<BuildStage> feed(std::string_view line);

  bool passed() const { return passed_; }  // « &&&& PASSED » vu
  bool failed() const { return failed_; }  // « &&&& FAILED » vu
  // « v110300 » de « &&&& RUNNING … [TensorRT v110300] » : la version de TensorRT que trtexec a chargée ;
  // vide pour toute autre ligne.
  static std::string reported_version(std::string_view line);
  // Dernière ligne d'erreur « [E] », pour le détail d'un refus ; vide sinon.
  const std::string& last_error() const { return last_error_; }

 private:
  float fraction_ = 0.0F;
  bool passed_ = false;
  bool failed_ = false;
  std::string last_error_;
};

}  // namespace ajp::prepare
