// VRR relu pendant la lecture : KDE arme ou désarme la fréquence variable selon la fenêtre active (politique
// « Automatique »). Une lecture isolée peut tomber pendant une bascule : un changement n'est retenu qu'après
// deux lectures égales de suite.
#pragma once

#include <optional>

namespace ajp::app::play {

class VrrWatch {
 public:
  explicit VrrWatch(bool armed) : armed_(armed) {}

  // Une lecture (rien : le système ne la dit pas, ignorée). Rend le nouvel état quand il change.
  std::optional<bool> observe(std::optional<bool> reading) {
    if (!reading) return std::nullopt;
    if (*reading == armed_) {
      candidate_.reset();
      return std::nullopt;
    }
    if (candidate_ != reading) {
      candidate_ = reading;  // première lecture différente : on attend la confirmation
      return std::nullopt;
    }
    armed_ = *reading;
    candidate_.reset();
    return armed_;
  }

  bool armed() const { return armed_; }

 private:
  bool armed_;
  std::optional<bool> candidate_;
};

}  // namespace ajp::app::play
