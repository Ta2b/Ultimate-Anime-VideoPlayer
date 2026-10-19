// resolve() : profil, capacités, source et écran deviennent un Plan immuable,
// ou un refus nommé. Fonction pure, sans GPU ni état global.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "core/plan.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

// Bloqueur TrueHDR tant que app n'a pas consulté vendor_hdr : la valeur par défaut refuse.
inline constexpr std::string_view kTrueHdrUnchecked = "TrueHDR : environnement non vérifié";
// Bloqueur quand la route TrueHDR n'est pas installée, l'environnement le permettant.
inline constexpr std::string_view kTrueHdrNotInstalled = "TrueHDR n'est pas installé";

// Ce que le système sait faire maintenant : modèles installés, routes HDR.
struct Capabilities {
  std::vector<ModelId> models;
  // Premier bloqueur TrueHDR, en phrase affichable telle quelle ; vide : TrueHDR peut être
  // tenté. app le tire de vendor_hdr::first_blocker_reason() : core ne dépend pas de vendor_hdr.
  std::string true_hdr_blocker = std::string(kTrueHdrUnchecked);
  bool true_hdr_available() const { return true_hdr_blocker.empty(); }
};

struct SourceInfo {
  int width = 0;
  int height = 0;
  double fps = 0.0;
};

// D'où vient un champ de DisplayInfo, pour l'OSD et le journal : la valeur par défaut de
// ce fichier, un réglage du profil, une mesure du système, ou --display.
enum class Provenance : std::uint8_t { Default, Profile, System, CommandLine };

// Espace de couleur de la sortie présentée.
enum class OutputSpace : std::uint8_t {
  Srgb,   // SDR : sRGB, primaires BT.709
  Pq,     // HDR10 : PQ (ST 2084), primaires BT.2020
  ScRgb,  // linéaire étendu, primaires BT.709 : le compositeur HDR de Windows
};

namespace display_defaults {
// Blanc de référence BT.2408, celui de libplacebo (PL_COLOR_SDR_WHITE) ; un écran SDR
// inconnu vaut ce blanc en crête, avec un contraste de 1000:1.
inline constexpr double kSdrWhiteNits = 203.0;
inline constexpr double kPeakNits = 203.0;
inline constexpr double kMinNits = 0.203;
}  // namespace display_defaults

namespace limits {
inline constexpr Range<double> kPeakNits{1.0, 10000.0};
inline constexpr Range<double> kSdrWhiteNits{1.0, 10000.0};
}  // namespace limits

struct DisplayProvenance {
  Provenance size = Provenance::Default;
  Provenance refresh = Provenance::Default;
  Provenance hdr = Provenance::Default;
  Provenance vrr = Provenance::Default;
  Provenance peak = Provenance::Default;
  Provenance min = Provenance::Default;
  Provenance sdr_white = Provenance::Default;
  Provenance space = Provenance::Default;
  friend bool operator==(const DisplayProvenance&, const DisplayProvenance&) = default;
};

struct DisplayInfo {
  int width = 0;
  int height = 0;
  // Fixe : le rafraîchissement ; VRR : le plafond du panneau ; 0 sans VRR : écran inconnu,
  // et resolve() ne cale aucune cadence.
  double refresh_hz = 0.0;
  bool hdr = false;  // HDR actif : la sortie part en HDR maintenant
  bool vrr = false;  // rafraîchissement variable actif
  double peak_nits = display_defaults::kPeakNits;
  double min_nits = display_defaults::kMinNits;  // dans [0, peak_nits[
  double sdr_white_nits = display_defaults::kSdrWhiteNits;
  OutputSpace space = OutputSpace::Srgb;
  DisplayProvenance provenance;
};

// La luminance de l'écran tient dans ses bornes ; un NaN est refusé.
std::expected<void, Refusal> check_display(const DisplayInfo& display);

// Première règle dont les bornes, incluses, contiennent la source ; nullptr sinon.
const Rule* match_rule(const Profile& profile, const SourceInfo& source);

// Écran connu (VRR, ou rafraîchissement non nul) : la règle choisie passe par pace_rule(),
// et le Plan porte sa cadence et son facteur d'interpolation retouché.
std::expected<Plan, Refusal> resolve(const Profile& profile, const Capabilities& caps,
                                     const SourceInfo& source, const DisplayInfo& display);

}  // namespace ajp::core
