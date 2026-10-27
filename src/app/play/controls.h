// État de l'interface de lecture, séparé de son dessin (controls_draw.cpp) et testé seul :
// visibilité et fondu selon l'activité de la souris, glissement sur la barre de progression,
// volume et muet, messages brefs, et le choix de recomposer l'interface ou non.
//
// Comme dans mpv, VLC ou YouTube : la barre paraît au moindre mouvement de la souris, reste tant
// que la lecture est en pause, qu'un menu est ouvert ou que la souris est dessus, et s'efface
// après kHideAfterUs d'inactivité. Sur OLED (set_oled_care), une pause sans activité efface aussi
// la barre après kPausedHideUs : une image fixe et une barre immobile marqueraient la dalle.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "app/play/chapters.h"
#include "core/frame.h"
#include "core/messages.h"
#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::int64_t kHideAfterUs = 2'000'000;    // inactivité avant l'effacement
inline constexpr std::int64_t kPausedHideUs = 20'000'000;  // OLED : en pause, inactivité avant l'effacement
inline constexpr std::int64_t kFadeUs = 200'000;           // durée du fondu
inline constexpr std::int64_t kToastUs = 1'600'000;        // durée d'un message bref
inline constexpr std::int64_t kDragSeekUs = 120'000;       // seeks au plus à ce rythme pendant le glissement
inline constexpr std::int64_t kUiFrameUs = 16'000;         // interface recomposée au plus à ~60 Hz
inline constexpr std::int64_t kPositionStepUs = 100'000;   // la position affichée avance par pas de 0,1 s
inline constexpr float kMaxVolume = 1.5F;                  // 150 % ; cmd::SetVolume accepte jusqu'à 2
inline constexpr float kVolumeStep = 0.05F;
inline constexpr std::int64_t kDoubleClickUs = 300'000;  // simple clic sur l'image : attend ce délai

// Ce que la session montre, lu par le dessin à chaque image de l'interface.
struct ControlsView {
  std::int64_t position_us = kNoPts;
  std::int64_t duration_us = 0;  // 0 : inconnue, la barre ne permet pas de seek
  bool paused = false;
  bool fullscreen = false;
  bool stats = false;                     // lignes de mesure (OSD, Tab) affichées
  std::span<const ChapterMark> chapters;  // dans l'ordre ; vide : boutons de chapitre absents
};

// Temps affiché : « 4:05 », « 12:34 », ou « 1:02:03 » quand la durée dépasse l'heure (même
// largeur pour la position et la durée). « --:-- » pour un temps inconnu.
std::string format_clock(std::int64_t us, std::int64_t duration_us);

// Décalage vers le haut des sous-titres du bas (bas du texte à text_bottom) pour qu'ils restent
// au-dessus d'une barre de hauteur reserved, sur une sortie de hauteur height ; 0 s'ils y sont déjà.
float subtitle_lift(float text_bottom, float height, float reserved);

// Fraction de la barre ↔ temps, bornés à [0, 1] et [0, durée].
double fraction_of(std::int64_t position_us, std::int64_t duration_us);
std::int64_t time_at(double fraction, std::int64_t duration_us);

class Controls {
 public:
  // Mouvement, clic ou molette : la barre paraît (fondu) et le délai d'effacement repart.
  void activity(std::int64_t now_us);
  // La souris a quitté la fenêtre : la barre s'efface sans attendre (sauf pause ou menu).
  void pointer_left(std::int64_t now_us);
  // Souris sur la barre, menu ouvert ou glissement : la barre reste.
  void set_held(bool held) { held_ = held; }
  // Écran OLED : la pause ne retient plus la barre au-delà de kPausedHideUs sans activité.
  void set_oled_care(bool on) { oled_care_ = on; }

  // Opacité de la barre : 1 visible, 0 effacée, entre les deux pendant un fondu.
  float opacity(std::int64_t now_us, bool paused) const;
  bool shown(std::int64_t now_us, bool paused) const { return opacity(now_us, paused) > 0.0F; }
  // Le curseur se cache avec la barre, sauf hors de la fenêtre.
  bool cursor_hidden(std::int64_t now_us, bool paused) const;

  // Volume (gain 0 à kMaxVolume) et muet ; chaque changement rend la commande pour audio.
  float volume() const { return volume_; }
  bool muted() const { return muted_; }
  cmd::SetVolume set_volume(float volume);  // borne, et quitte le muet
  cmd::SetVolume step_volume(float delta);
  cmd::SetVolume toggle_mute();
  cmd::SetVolume volume_command() const { return {.gain = muted_ ? 0.0F : volume_}; }

  // Clic sur l'image, hors de la barre : le double clic bascule le plein écran, le simple clic
  // met en pause (ou reprend) après kDoubleClickUs sans second clic (pending_click le rend).
  enum class VideoClick : std::uint8_t { TogglePause, ToggleFullscreen };
  std::optional<VideoClick> click(std::int64_t now_us, int clicks);
  std::optional<VideoClick> pending_click(std::int64_t now_us);

  // Barre de progression. Clic : la tête suit la souris (press) ; glissement : seeks au point
  // clé, au plus toutes les kDragSeekUs et seulement si la cible a changé ; au lâcher, un seek
  // exact.
  void press(double fraction);
  std::optional<cmd::Seek> drag_to(double fraction, std::int64_t duration_us, std::int64_t now_us);
  cmd::Seek release(double fraction, std::int64_t duration_us);
  bool dragging() const { return dragging_; }
  // Fraction affichée pendant le glissement (la tête suit la souris, pas l'image).
  std::optional<double> drag_fraction() const {
    return dragging_ ? std::optional(drag_fraction_) : std::nullopt;
  }

  // Message bref (volume, piste, réglage), en haut de l'écran.
  void notify(std::string text, std::int64_t now_us);
  // Message d'avant la lecture (installation, repli) : mis de côté, montré à la première image.
  void defer(std::string text);
  void show_deferred(std::int64_t now_us);
  std::optional<std::string_view> toast(std::int64_t now_us) const;
  float toast_opacity(std::int64_t now_us) const;

  // Faut-il recomposer l'interface à ce tour ? Oui après une entrée (input_pending), pendant un
  // fondu ou un message, quand la position affichée change d'un pas, et une fois après que la
  // barre s'est effacée (pour l'ôter de l'écran). Jamais plus souvent que kUiFrameUs.
  bool needs_frame(std::int64_t now_us, const ControlsView& view, bool input_pending) const;
  // L'interface vient d'être recomposée avec view ; drawn : quelque chose était visible.
  void framed(std::int64_t now_us, const ControlsView& view, bool drawn);
  bool drawn() const { return drawn_; }

 private:
  bool animating(std::int64_t now_us, bool paused) const;
  bool fading_in(std::int64_t now_us) const;

  std::optional<std::int64_t> active_at_;  // dernière activité ; rien : jamais montrée
  std::int64_t shown_since_ = 0;           // début du fondu d'apparition
  bool left_ = false;                      // souris hors de la fenêtre
  std::int64_t left_at_ = 0;
  bool held_ = false;
  bool oled_care_ = false;
  float volume_ = 1.0F;
  bool muted_ = false;
  bool dragging_ = false;
  double drag_fraction_ = 0.0;
  std::optional<std::int64_t> click_at_;  // simple clic en attente d'un éventuel second
  std::optional<std::int64_t> drag_seek_at_;
  std::int64_t drag_target_us_ = -1;
  std::string toast_;
  std::string deferred_;
  std::int64_t toast_at_us_ = 0;
  std::optional<std::int64_t> framed_at_;
  std::int64_t framed_position_step_ = -1;
  bool framed_paused_ = false;
  bool drawn_ = false;
};

}  // namespace ajp::app::play
