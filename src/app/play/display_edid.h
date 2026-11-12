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
#include <vector>

#include "render/display_model.h"

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
  // Ce que chaque bloc annonce, avant le choix ci-dessus (A11-11) : le journal les montre côte à côte quand
  // les deux blocs parlent, pour décider d'une seule source par grandeur sur un écran réel.
  std::optional<double> cta_peak_nits;
  std::optional<double> cta_min_nits;
  std::optional<double> displayid_peak_nits;
  std::optional<double> displayid_min_nits;
  std::optional<bool> displayid_oled;  // bloc 0x21 lu : AMOLED ou non
  bool oled = false;
  // Fréquence variable (VRR) utilisable : propriété DRM vrr_capable sous Linux (le compositeur
  // l'arme à son gré, KDE en plein écran) ; sous Windows, la plage VRR de l'EDID (vrr_range), jamais la
  // présentation sans vsync (DXGI_FEATURE_PRESENT_ALLOW_TEARING), que tout Windows 10 offre même sur un écran
  // 60 Hz fixe. Rien : inconnu.
  std::optional<bool> vrr;
  // VRR armé maintenant par le compositeur (VRR_ENABLED du CRTC sous Linux) ou le pilote (NVAPI sous Windows,
  // rempli par window_display de host_facts.h). Rien : inconnu (AMD ou Intel sous Windows : le retour de
  // présentation le dit, 6.10).
  std::optional<bool> vrr_armed;
  // Fréquence verticale minimale du descripteur de plage 0xFD de l'EDID (Hz), qu'il annonce ou non une
  // fréquence variable : le plancher du VRR quand vrr_range manque.
  std::optional<int> vrr_min_hz;
  // La plage de fréquence variable que l'écran annonce (12.2), dans l'ordre : DisplayID 2.0 bloc 0x2B
  // (Adaptive-Sync), bloc AMD FreeSync de CTA-861, HF-VSDB HDMI 2.1 (VRRmin, VRRmax), puis le descripteur
  // 0xFD d'un EDID 1.4 qui se dit à fréquence continue (octet 0x18, bit 0). Une plage de 10 Hz ou moins
  // n'en est pas une. Rien : l'écran n'annonce pas de VRR.
  struct VrrRange {
    int min_hz = 0;
    int max_hz = 0;
    friend bool operator==(const VrrRange&, const VrrRange&) = default;
  };
  std::optional<VrrRange> vrr_range;
  // Le plancher du VRR (Hz) : celui de vrr_range, sinon vrr_min_hz, sinon 0.
  int vrr_floor_hz() const { return vrr_range ? vrr_range->min_hz : vrr_min_hz.value_or(0); }
  // Primaires et blanc des octets 0x19 à 0x22 du bloc de base (10 bits : (fort << 2 | faible) / 1024),
  // gardés seulement si le blanc est à ±0,01 de D65.
  std::optional<render::Chromaticities> primaries;
  OledSource source = OledSource::Unknown;
};

// Ce que dit un EDID (128 octets par bloc, extensions comprises) ; un bloc illisible est ignoré.
DisplayFacts parse_edid(std::span<const std::uint8_t> edid);
// « MAG 272U E16 · HDR10, pic 1037 nits, moyenne 265, noir 0,00064 nit · OLED : non (EDID) »
std::string describe(const DisplayFacts& display);
// Pour le journal, à la suite de describe : vide tant qu'un seul bloc de l'EDID annonce le pic ou le noir ;
// sinon ce que dit chacun, ce qui est retenu aujourd'hui (pic du CTA, noir de DisplayID s'il vient après), et
// ce qu'une seule source par grandeur donnerait (DisplayID d'abord, comme pour l'OLED).
std::string luminance_sources(const DisplayFacts& display);
// La ligne du journal : describe, puis luminance_sources.
std::string describe_for_log(const DisplayFacts& display);
// L'écran branché (Linux : /sys/class/drm/*-*/status = connected ; Windows : présent) ; s'il y en a
// plusieurs, celui dont le nom se retrouve dans wanted (nom de l'écran selon SDL). Rien si aucun
// EDID ne se lit, et toujours rien sur macOS.
// Le VRR armé à cet instant pour ce connecteur (« card1-DP-1 »), sans relire l'EDID : connector_id dans
// sysfs, puis /dev/dri/cardN, GETCONNECTOR, GETENCODER et la propriété VRR_ENABLED du CRTC
// (B34-14). Rien hors Linux ou si le système ne le dit pas.
std::optional<bool> vrr_armed_now(std::string_view connector);

// L'état armé lu dans DRM (B11-01), pur. Sans CRTC (sortie éteinte), rien n'est armé ; VRR_ENABLED absente
// (descripteur sans DRM_CLIENT_CAP_ATOMIC, pilote sans VRR) est un état inconnu, jamais « non armé ».
constexpr std::optional<bool> vrr_armed_from(std::uint32_t crtc, std::optional<std::uint64_t> vrr_enabled) {
  if (crtc == 0) return false;
  if (!vrr_enabled) return std::nullopt;
  return *vrr_enabled != 0;
}

// L'écran de `found` que nomme `wanted` (le nom que SDL donne à l'écran de la fenêtre). Pur (A11-03).
// Dans l'ordre : le connecteur nommé en toutes lettres (« DP-2 » pour « card1-DP-2 »), puis le nom de
// l'EDID. Deux écrans du même modèle portent le même nom : avant, le premier gagnait et son connecteur
// passait pour celui de la fenêtre (VRR armé et réglages KWin de l'autre écran). L'EDID commun est rendu,
// sans connecteur ni état propre à une sortie : inconnu plutôt que faux. Sans aucun nom reconnu, le premier.
//
// instance (Windows, B34-02) : le chemin d'interface du moniteur de la fenêtre (EnumDisplayDevices,
// EDD_GET_DEVICE_INTERFACE_NAME), ou son identifiant d'instance. L'écran dont le connecteur (l'identifiant
// d'instance de SetupAPI) lui correspond passe devant les noms : deux écrans du même modèle, un nom SDL
// générique (« Generic PnP Monitor ») ou un EDID sans nom ne font plus lire l'EDID d'un autre écran.
std::optional<DisplayFacts> pick_display(const std::vector<DisplayFacts>& found, std::string_view wanted,
                                         std::string_view instance = {});
std::optional<DisplayFacts> read_display(std::string_view wanted, std::string_view instance = {});

// « \\?\DISPLAY#GSM5B08#5&2a0b3e4c&0&UID4352#{e6f07b5f-…} » (chemin d'interface) ou
// « DISPLAY\GSM5B08\5&2A0B3E4C&0&UID4352 » (identifiant d'instance) : les deux rendent le second, en
// capitales, pour les comparer. Pur.
std::string monitor_instance_from_interface(std::string_view path);

}  // namespace ajp::app::play
