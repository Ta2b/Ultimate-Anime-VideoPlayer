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
// retombe sur le meilleur en dessous, sauf wait (le défaut, options.h), qui prend un niveau qui date quand il
// y en a un (choose_present_level). open est la présentation d'avant.
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
// niveau qui date l'affichage passe devant celui qui ne fait que sonder (A10-02 : sans date, pas de grille
// des vblanks). À --present=wait, le défaut : wp_presentation d'abord (CachyOS), sinon present timing relu
// après coup (Timing, jamais de date visée), sinon le sondage. Sans cela, Windows et X11 n'avaient jamais de
// grille des vblanks, quand CachyOS l'a (B10-01). Present timing muet retombe sur le sondage
// (feedback_silent).
constexpr PresentLevel choose_present_level(const PresentCaps& caps, PresentWanted wanted) {
  using enum PresentLevel;
  if (wanted == PresentWanted::Open) return Open;
  if (wanted == PresentWanted::Wayland) return available(Wayland, caps) ? Wayland : Open;
  if (wanted != PresentWanted::Wait) {
    if (available(TimingAbsolute, caps)) return TimingAbsolute;
    if (available(Timing, caps)) return Timing;
  }
  if (available(Wayland, caps)) return Wayland;
  if (wanted == PresentWanted::Wait && available(Timing, caps)) return Timing;
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

// Un relevé de l'horloge hôte rendu par Vulkan, en nanosecondes (A10-07). Sous Linux le domaine
// CLOCK_MONOTONIC est déjà en ns (per_second = 1e9). Sous Windows ce domaine n'existe pas : l'horloge de
// host_now_us y est le compteur de performance (VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER), rendu en tics.
// Sans débordement : la partie entière et le reste sont convertis à part.
constexpr std::uint64_t host_ticks_to_ns(std::uint64_t ticks, std::uint64_t per_second) {
  if (per_second == 0 || per_second == 1'000'000'000ULL) return ticks;
  return (ticks / per_second) * 1'000'000'000ULL + (ticks % per_second) * 1'000'000'000ULL / per_second;
}

// L'inverse : des ns de l'hôte en tics du compteur, sans débordement.
constexpr std::uint64_t host_ns_to_ticks(std::uint64_t ns, std::uint64_t per_second) {
  if (per_second == 0 || per_second == 1'000'000'000ULL) return ns;
  return (ns / 1'000'000'000ULL) * per_second + (ns % 1'000'000'000ULL) * per_second / 1'000'000'000ULL;
}

// Une date de present timing dans le domaine du compteur de performance de Windows (B10-02) : la
// spécification donne les dates en ns, un pilote peut les rendre en tics du compteur. Une présentation
// revient moins d'une seconde après son départ : la lecture la plus proche de l'heure hôte now_us est la
// bonne. Vrai pour des tics. À 10 MHz, les deux lectures diffèrent d'un facteur 100 : aucune confusion
// possible après le démarrage.
constexpr bool host_stamp_in_ticks(std::uint64_t value, std::uint64_t per_second, std::int64_t now_us) {
  if (per_second == 0 || per_second == 1'000'000'000ULL) return false;
  const auto as_ns = static_cast<std::int64_t>(value / 1000);
  const auto as_ticks = static_cast<std::int64_t>(host_ticks_to_ns(value, per_second) / 1000);
  const auto distance = [now_us](std::int64_t us) { return us > now_us ? us - now_us : now_us - us; };
  return distance(as_ticks) < distance(as_ns);
}

// Les acquisitions d'image de la swapchain qui échouent de suite sur une fenêtre visible, avec un appareil
// sain (B31-15) : surface perdue (VK_ERROR_SURFACE_LOST_KHR, écran débranché que le compositeur a détruit).
// Rien ne la rend dans ce processus : au-delà de kLostSurfaceUs, l'écran le dit et la lecture se relance
// (comme pour un GPU perdu), au lieu d'une image figée sans fin sous le son qui continue.
inline constexpr std::int64_t kLostSurfaceUs = 3'000'000;
class AcquireStreak {
 public:
  // Un échec à now_us ; vrai quand les échecs durent depuis kLostSurfaceUs sans une réussite.
  constexpr bool failed(std::int64_t now_us) {
    if (since_us_ == 0) since_us_ = now_us;
    return now_us - since_us_ >= kLostSurfaceUs;
  }
  // Une image acquise, ou une fenêtre cachée (rien n'y est demandé) : la série s'arrête.
  constexpr void reset() { since_us_ = 0; }

 private:
  std::int64_t since_us_ = 0;
};

// L'instantané garde l'espace de couleur de la fenêtre (A10-18) : toujours pour « .pq.ppm » sur une sortie
// PQ, et pour toute fenêtre SDR. Une fenêtre HDR photographiée en SDR passe par sRGB.
constexpr bool snapshot_keeps_target(bool pq_file, bool target_hdr) {
  return pq_file || !target_hdr;
}

}  // namespace ajp::app::play
