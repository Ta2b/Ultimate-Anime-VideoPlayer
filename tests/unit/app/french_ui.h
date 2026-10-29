#pragma once

#include "app/play/i18n.h"

namespace ajp::app::play {

// Un test qui vérifie les textes français met l'interface en français le temps de sa portée (l'anglais est le
// défaut), puis la rend à l'anglais.
struct FrenchUi {
  FrenchUi() { i18n::set_language(i18n::Language::French); }
  ~FrenchUi() { i18n::set_language(i18n::Language::English); }
  FrenchUi(const FrenchUi&) = delete;
  FrenchUi& operator=(const FrenchUi&) = delete;
};

}  // namespace ajp::app::play
