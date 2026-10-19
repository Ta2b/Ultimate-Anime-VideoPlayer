// Détection de l'écran : ce que le système annonce (SystemDisplay), ce que le profil force
// (DisplayOverrides) et ce que --display remplace deviennent un DisplayInfo dont chaque champ
// garde sa provenance, pour l'OSD et le journal.
//
// Priorité, du plus faible au plus fort : valeur par défaut, système, profil, --display.
// Le blanc SDR du système n'est jamais appliqué : il est proposé (proposed_sdr_white_nits),
// l'utilisateur le reprend dans son profil s'il le veut.
//
// Ce que chaque système sait dire :
//   - SDL3, partout : taille et rafraîchissement exact du mode courant, HDR actif, marge HDR
//     de la fenêtre. Sous Windows la marge multiplie le blanc SDR du système ; sous Wayland elle
//     est relative au blanc de référence du compositeur, que SDL3 tait : la crête vaut alors
//     marge × 203 nits, le blanc d'une sortie PQ (comme mpv), et aucun blanc n'est proposé ;
//   - Windows, en plus : DXGI_OUTPUT_DESC1 (espace, luminances min et max) et le blanc SDR de
//     DisplayConfigGetDeviceInfo (GET_SDR_WHITE_LEVEL), comme mpv ;
//   - Linux : pas de libdisplay-info (hors des dépendances), donc ni EDID ni DisplayID 0x21 :
//     ni noir de l'écran, ni crête du panneau hors compositeur, ni indice OLED ;
//   - VRR : aucun système ne le dit par SDL, il vient des réglages.
#pragma once

#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "core/profile.h"
#include "core/resolve.h"
#include "core/types.h"

namespace ajp::present {

// Lecture brute du système ; un champ absent n'est pas connu de ce système.
struct SystemDisplay {
  std::string name;
  int width = 0;  // pixels du mode courant ; 0 : inconnu
  int height = 0;
  std::optional<core::Ratio> refresh;  // rationnel exact du mode, en Hz
  std::optional<bool> hdr;             // sortie HDR active
  std::optional<double> sdr_white_nits;
  std::optional<double> peak_nits;
  std::optional<double> min_nits;
  std::vector<std::string> unknown;  // ce que ce système ne sait pas dire, en phrases
};

// Chaque champ de DisplayInfo, forçable par le profil.
struct DisplayOverrides {
  std::optional<int> width;
  std::optional<int> height;
  std::optional<double> refresh_hz;
  std::optional<bool> hdr;
  std::optional<bool> vrr;
  std::optional<double> peak_nits;
  std::optional<double> min_nits;
  std::optional<double> sdr_white_nits;
  std::optional<core::OutputSpace> space;
};

struct DisplayDetection {
  core::DisplayInfo info;
  std::optional<core::Ratio> refresh;             // le rafraîchissement retenu, exact
  std::optional<double> proposed_sdr_white_nits;  // blanc du système, proposé, jamais appliqué
  std::vector<std::string> unknown;
};

// Assemble l'écran. Le rafraîchissement du système est ramené à sa valeur nominale
// (nominal_refresh) ; une crête n'est lue du système qu'en HDR (en SDR, la marge vaut 1 et ne
// dit rien du panneau). Refuse toute valeur hors de ses bornes, d'où qu'elle vienne, en
// nommant le champ et sa provenance.
std::expected<DisplayDetection, Refusal> detect_display(const SystemDisplay& system,
                                                        const DisplayOverrides& profile,
                                                        const std::optional<core::DisplayInfo>& command_line);

// Nom affichable d'une provenance, pour l'OSD.
std::string_view provenance_name(core::Provenance provenance);

}  // namespace ajp::present
