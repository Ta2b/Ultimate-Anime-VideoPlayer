// Le niveau de retour de présentation (chantier 6.2, docs/adr/0011-presentation-en-boucle-fermee.md) : ce que
// l'appareil, la surface et le compositeur permettent de savoir sur l'instant d'affichage de chaque image, du
// plus pauvre au plus riche. Pur : placebo_open.cpp lit les capacités, ce fichier choisit.
//   Open            : aucun retour, la présentation d'aujourd'hui (sondage, VsyncPhase).
//   Wayland         : wp_presentation (6.4), l'instant du compositeur.
//   Wait            : VK_KHR_present_id2 + present_wait2, la fin de présentation de chaque image.
//   Timing          : VK_EXT_present_timing, l'instant du premier pixel relu après coup.
//   TimingAbsolute  : present timing avec une date visée (presentAtAbsoluteTime).
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace ajp::app::play {

enum class PresentLevel : std::uint8_t { Open, Wayland, Wait, Timing, TimingAbsolute };

// --present=… : auto choisit le meilleur niveau disponible ; un niveau nommé n'est jamais dépassé, et
// retombe sur le meilleur en dessous. wait est le défaut (options.h) ; open est la présentation d'avant.
enum class PresentWanted : std::uint8_t { Open, Auto, Wayland, Wait, Timing };

// Ce que l'appareil (fonctionnalités activées) et la surface (capacités) offrent.
struct PresentCaps {
  bool present_id2 = false;           // fonctionnalité et capacité de la surface
  bool present_wait2 = false;         // idem
  bool timing = false;                // VkPhysicalDevicePresentTimingFeaturesEXT::presentTiming et la surface
  bool timing_absolute = false;       // presentAtAbsoluteTime, de l'appareil et de la surface
  std::uint32_t present_stages = 0;   // VkPresentStageFlagsEXT que la surface sait dater
  bool calibrated = false;            // VK_KHR_calibrated_timestamps (horloge du GPU recalée sur l'hôte)
  bool wayland_presentation = false;  // wp_presentation offert par le compositeur (6.4)
};

constexpr std::optional<PresentWanted> parse_present(std::string_view text) {
  if (text == "open") return PresentWanted::Open;
  if (text == "auto") return PresentWanted::Auto;
  if (text == "wayland") return PresentWanted::Wayland;
  if (text == "wait") return PresentWanted::Wait;
  if (text == "timing") return PresentWanted::Timing;
  return std::nullopt;
}

constexpr std::string_view name(PresentLevel level) {
  switch (level) {
    case PresentLevel::Open:
      return "open";
    case PresentLevel::Wayland:
      return "wayland";
    case PresentLevel::Wait:
      return "wait";
    case PresentLevel::Timing:
      return "timing";
    case PresentLevel::TimingAbsolute:
      return "timing-absolute";
  }
  return "open";
}

constexpr bool available(PresentLevel level, const PresentCaps& caps) {
  switch (level) {
    case PresentLevel::Open:
      return true;
    case PresentLevel::Wayland:
      return caps.wayland_presentation;
    case PresentLevel::Wait:
      return caps.present_id2 && caps.present_wait2;
    case PresentLevel::Timing:
      // present timing date les images par leur identifiant : present_id2 est requis.
      return caps.timing && caps.present_id2;
    case PresentLevel::TimingAbsolute:
      return caps.timing && caps.timing_absolute && caps.present_id2;
  }
  return false;
}

// wp_presentation s'ouvre pour toute demande autre qu'open : c'est le niveau daté où retombent wait (le
// défaut), auto et timing quand present timing manque ou ne rend rien (A10-02).
constexpr bool wants_wayland(PresentWanted wanted) {
  return wanted != PresentWanted::Open;
}

// Le niveau retenu : jamais au-dessus de ce qui est demandé ni de ce qui est offert. Open reste Open. Un
// niveau qui date l'affichage passe devant celui qui ne fait que sonder : Wayland devant Wait, y compris à
// --present=wait, où Wait ne reste que sans wp_presentation (A10-02 : sans date, pas de grille des vblanks).
constexpr PresentLevel choose_present_level(const PresentCaps& caps, PresentWanted wanted) {
  using enum PresentLevel;
  if (wanted == PresentWanted::Open) return Open;
  if (wanted == PresentWanted::Wayland) return available(Wayland, caps) ? Wayland : Open;
  if (wanted != PresentWanted::Wait) {
    if (available(TimingAbsolute, caps)) return TimingAbsolute;
    if (available(Timing, caps)) return Timing;
  }
  if (available(Wayland, caps)) return Wayland;
  return available(Wait, caps) ? Wait : Open;
}

constexpr bool timing_level(PresentLevel level) {
  return level == PresentLevel::Timing || level == PresentLevel::TimingAbsolute;
}

// Present timing reste muet (A10-06) : kSilentPresents présentations parties, aucune date revenue. Le niveau
// retombe alors sur fallback_level : wp_presentation s'il est ouvert, sinon le sondage.
inline constexpr std::uint64_t kSilentPresents = 120;
constexpr bool feedback_silent(std::uint64_t issued, std::uint64_t returned) {
  return returned == 0 && issued >= kSilentPresents;
}
constexpr PresentLevel fallback_level(bool wayland_open) {
  return wayland_open ? PresentLevel::Wayland : PresentLevel::Wait;
}

// Swapchain hors Open : 4 images, 2 en vol (une image visée d'avance, une à l'écran) ; Open garde celle
// d'aujourd'hui (kSwapchainDepth, libplacebo choisit le nombre d'images).
struct SwapchainShape {
  int min_images = 0;  // 0 : libplacebo choisit
  int depth = 0;       // 0 : kSwapchainDepth
};
constexpr SwapchainShape swapchain_shape(PresentLevel level) {
  if (level == PresentLevel::Open || level == PresentLevel::Wayland) return {};
  return {.min_images = 4, .depth = 2};
}

// L'état de la fréquence lu par present timing (6.10) : VkSwapchainTimingPropertiesEXT dit VRR par un
// intervalle de UINT64_MAX, une fréquence fixe par un intervalle égal à la durée d'un rafraîchissement ; rien
// tant que la durée n'est pas connue.
enum class RefreshState { Unknown, Fixed, Variable };
constexpr RefreshState vrr_state(std::uint64_t duration_ns, std::uint64_t interval_ns) {
  if (duration_ns == 0) return RefreshState::Unknown;
  if (interval_ns == UINT64_MAX) return RefreshState::Variable;
  if (interval_ns == duration_ns) return RefreshState::Fixed;
  return RefreshState::Unknown;
}

}  // namespace ajp::app::play
