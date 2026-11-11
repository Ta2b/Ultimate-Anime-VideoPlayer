// VRR relu pendant la lecture : KDE arme ou désarme la fréquence variable selon la fenêtre active (politique
// « Automatique »). Une lecture isolée peut tomber pendant une bascule : un changement n'est retenu qu'après
// deux lectures égales de suite.
#pragma once

#include <cstdint>
#include <optional>

namespace ajp::app::play {

// NvAPI_Disp_GetVRRInfo (Windows, B11-02) : bit 0 bIsVRREnabled (G-SYNC permis sur cet écran), bit 4
// bIsDisplayInVRRMode (engagé en ce moment) dans les SDK récents. Un pilote qui ne pose jamais le bit 4
// ferait lire « non armé » pour toujours : il n'est exigé qu'une fois vu posé dans ce processus. Pur.
inline constexpr std::uint32_t kNvVrrEnabled = 1U << 0U;
inline constexpr std::uint32_t kNvVrrInMode = 1U << 4U;
constexpr bool nv_vrr_armed(std::uint32_t flags, bool in_mode_seen) {
  if ((flags & kNvVrrEnabled) == 0) return false;
  return (flags & kNvVrrInMode) != 0 || !in_mode_seen;
}

class VrrWatch {
 public:
  explicit VrrWatch(bool armed) : armed_(armed) {}

  // Une lecture (rien : le système ne la dit pas, ignorée). Rend le nouvel état quand il change.
  std::optional<bool> observe(std::optional<bool> reading) {
    if (!reading) return std::nullopt;
    if (*reading == armed_) {
      pending_ = false;
      return std::nullopt;
    }
    if (!pending_) {
      pending_ = true;  // première lecture différente : on attend la confirmation
      return std::nullopt;
    }
    armed_ = *reading;
    pending_ = false;
    return armed_;
  }

  // Une lecture sûre (present timing, 6.10) : retenue tout de suite, sans confirmation ; rien, ignorée.
  std::optional<bool> observe_sure(std::optional<bool> reading) {
    pending_ = false;
    if (!reading || *reading == armed_) return std::nullopt;
    armed_ = *reading;
    return armed_;
  }
  // La lecture de present timing si elle est là (sûre), sinon celle du système (à confirmer).
  std::optional<bool> observe(std::optional<bool> sure, std::optional<bool> reading) {
    return sure ? observe_sure(sure) : observe(reading);
  }

  bool armed() const { return armed_; }

 private:
  bool armed_;
  // Une lecture différente d'armed_ attend sa confirmation. Un booléen suffit : la seule valeur différente
  // d'armed_ est !armed_. Un optional<bool> faisait signaler à gcc 13 une lecture non initialisée (Release).
  bool pending_ = false;
};

}  // namespace ajp::app::play
