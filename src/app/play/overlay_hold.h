// L'interface et l'OSD ne présentent pas seuls juste avant une image neuve (A24-02) : une présentation
// partie à un instant quelconque prend, sur un écran fixe, le vblank où l'image neuve était attendue, qui
// sort alors un vblank plus tard. À moins de kOverlayGuardPeriods périodes d'écran de l'image suivante, le
// changement est gardé et part avec elle (la composition prend toujours l'interface du moment) ; plus tôt,
// il part seul, sur un vblank que l'image n'attendait pas. En interpolation à la fréquence de l'écran,
// l'interface suit donc les images ; à 23,976 i/s sur 144 Hz, elle garde ses 60 Hz entre deux images. En
// pause, ou quand plus aucune image neuve n'arrive (seek, tampon vide, fin), il part tout de suite. Pur.
#pragma once

#include <cstdint>
#include <utility>

namespace ajp::app::play {

// Sans image neuve depuis ce temps, la lecture ne coule plus : l'interface présente seule. Deux images à
// 23,976 i/s tiennent dedans ; une source sous 10 i/s garde l'interface d'avant.
inline constexpr std::int64_t kOverlayHoldUs = 100'000;

class OverlayHold {
 public:
  struct Due {
    bool osd = false;
    bool ui = false;
  };
  // osd_new, ui_new : ce qui a changé à ce tour. hold : une image neuve est attendue sous peu
  // (overlay_holds). fresh : elle part à ce tour. Rend ce qui doit présenter maintenant ; rien tant que le
  // changement attend.
  Due note(bool osd_new, bool ui_new, bool hold, bool fresh) {
    osd_ = osd_ || osd_new;
    ui_ = ui_ || ui_new;
    if (hold && !fresh) return {};
    return {.osd = std::exchange(osd_, false), .ui = std::exchange(ui_, false)};
  }
  bool pending() const { return osd_ || ui_; }

 private:
  bool osd_ = false;
  bool ui_ = false;
};

// La lecture coule : pas en pause, et une image neuve présentée il y a moins de kOverlayHoldUs.
constexpr bool overlay_flowing(bool paused, std::int64_t last_fresh_us, std::int64_t now_us) {
  return !paused && last_fresh_us != 0 && now_us - last_fresh_us < kOverlayHoldUs;
}

// Une présentation de l'interface seule doit partir au moins ce nombre de périodes d'écran avant l'image
// suivante : une pour son vblank, une de marge sur l'instant de l'image (à-coups du rendu, file FIFO).
inline constexpr std::int64_t kOverlayGuardPeriods = 2;

// Le changement attend : la lecture coule, et l'image suivante (la dernière neuve plus la période de sortie)
// est à moins de la garde, ou en retard. Sans période connue, il attend tant que la lecture coule.
constexpr bool overlay_holds(bool paused, std::int64_t last_fresh_us, std::int64_t now_us,
                             std::int64_t output_period_us, std::int64_t display_period_us) {
  if (!overlay_flowing(paused, last_fresh_us, now_us)) return false;
  if (output_period_us <= 0 || display_period_us <= 0) return true;
  return now_us > last_fresh_us + output_period_us - kOverlayGuardPeriods * display_period_us;
}

}  // namespace ajp::app::play
