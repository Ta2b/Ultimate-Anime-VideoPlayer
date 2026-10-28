// L'écran tel que son EDID le décrit (Linux : /sys/class/drm/<sortie>/edid ; Windows : le registre
// des écrans présents, par SetupAPI ; les deux lisibles sans droits) :
// nom, bloc HDR statique de CTA-861 (PQ, luminances), et s'il est OLED, dans l'ordre du plan :
//   1. DisplayID 2.0, bloc 0x21 (paramètres de l'écran), octet 0x1E, bits 6:4 = 010b : définitif ;
//   2. table des modèles connus, par fabricant et product id EDID ou par nom ;
//   3. heuristique EDID : noir ≤ 0,001 nit ou code ≤ 2, et plein écran ≤ 0,5 × pic (un mini-LED
//      peut annoncer un noir quasi nul : le choix du profil tranche toujours) ;
//   4. sinon, inconnu.
// Sans dépendance (libdisplay-info n'est pas dans le préfixe de la tranche verticale).
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ajp::app::play {

enum class OledSource : std::uint8_t { DisplayId, KnownModel, Heuristic, Unknown };

struct DisplayFacts {
  std::string connector;                     // « card1-DP-1 »
  std::string name;                          // descripteur 0xFC : « MAG 272U E16 »
  std::string manufacturer;                  // identifiant PNP : « CND »
  std::uint16_t product = 0;                 // product id EDID : 0x8CD7
  std::string panel;                         // technologie connue : « QD-OLED », vide sinon
  bool pq = false;                           // EOTF ST 2084 annoncée
  std::optional<double> peak_nits;           // contenu max désiré (CTA) ou pic de DisplayID
  std::optional<double> frame_average_nits;  // moyenne d'image max (CTA)
  std::optional<double> min_nits;            // noir annoncé
  std::optional<int> min_code;               // octet du noir dans le bloc CTA (0 : noir parfait)
  std::optional<bool> displayid_oled;        // bloc 0x21 lu : AMOLED ou non
  bool oled = false;
  // Fréquence variable (VRR) utilisable : propriété DRM vrr_capable sous Linux (le compositeur
  // l'arme à son gré, KDE en plein écran) ; capacité du système à présenter sans attendre le vsync
  // sous Windows. Rien : inconnu.
  std::optional<bool> vrr;
  // VRR armé maintenant par le compositeur (VRR_ENABLED du CRTC sous Linux). Rien : inconnu ou, sous
  // Windows, identique à vrr.
  std::optional<bool> vrr_armed;
  OledSource source = OledSource::Unknown;
};

// Ce que dit un EDID (128 octets par bloc, extensions comprises) ; un bloc illisible est ignoré.
DisplayFacts parse_edid(std::span<const std::uint8_t> edid);
// « MAG 272U E16 · HDR10, pic 1037 nits, moyenne 265, noir 0,00064 nit · OLED : non (EDID) »
std::string describe(const DisplayFacts& display);
// L'écran branché (Linux : /sys/class/drm/*-*/status = connected ; Windows : présent) ; s'il y en a
// plusieurs, celui dont le nom se retrouve dans wanted (nom de l'écran selon SDL). Rien si aucun
// EDID ne se lit, et toujours rien sur macOS.
std::optional<DisplayFacts> read_display(std::string_view wanted);

}  // namespace ajp::app::play
