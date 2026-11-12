// Ce que le poste et l'écran de la fenêtre disent d'eux-mêmes : rafraîchissement et marge HDR
// selon SDL, nom de l'écran (pour son EDID), GPU selon Vulkan, moteurs présents pour l'écran de
// bienvenue.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <SDL3/SDL_video.h>

#include "app/play/display_edid.h"
#include "app/play/engine_install.h"
#include "app/play/first_run.h"
#include "app/play/options.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"

namespace ajp::app::play {

// Période de rafraîchissement de l'écran de la fenêtre ; 0 si inconnue (la session la mesure
// alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window);
// Le même rafraîchissement, exact quand SDL le donne en fraction (59,94 Hz = 60000/1001) ; 0 si
// inconnu.
double display_refresh_hz(SDL_Window* window);
// La hauteur en pixels du mode de l'écran de la fenêtre ; 0 si inconnue.
int display_height(SDL_Window* window);
// Sa largeur, dans les mêmes unités ; 0 : inconnue.
int display_width(SDL_Window* window);
// Crête vue par le contenu : la marge HDR de SDL3 × 203 nits ; rien si l'écran n'est pas en HDR.
std::optional<double> content_peak_nits(SDL_Window* window);
// Ce que seuls les appels vidéo de SDL disent (le thread main seul les fait) : le mode et le nom de l'écran
// de la fenêtre. Les deux relevés qui suivent, avec lui, ne font plus d'appel vidéo de SDL (les propriétés
// de la fenêtre se lisent de tout thread) : sous Windows, QueryDisplayConfig, DXGI, le registre, qui
// coûtent des millisecondes, se font hors du thread qui présente (B11-05, B11-06, display_probe.h).
struct DisplaySdl {
  double hz = 0.0;   // le mode de SDL ; 0 : inconnu
  std::string name;  // le nom de l'écran selon SDL, pour son EDID
};
DisplaySdl display_sdl(SDL_Window* window);
double display_refresh_hz(SDL_Window* window, const DisplaySdl& sdl);
std::optional<double> content_peak_nits(SDL_Window* window, const DisplaySdl& sdl);
// Ce que le système dit de l'écran, et la crête qui s'en déduit : la règle de content_peak_nits, sans
// fenêtre ni système, pour qu'un test la tienne (A11-12).
struct PeakFacts {
  bool windows = false;
  bool sdl_hdr = false;           // SDL dit le HDR actif, avec une marge au-dessus de 1
  double headroom = 1.0;          // la marge HDR de SDL
  double sdr_white_nits = 203.0;  // Windows : le curseur « luminosité du contenu SDR »
  // Le système dit le HDR actif, ce que SDL ne voit pas toujours : « Utiliser le HDR » sous Windows
  // (advancedColorEnabled), « highDynamicRange » de la sortie dans le fichier de KWin sous Linux (C34-09).
  bool system_hdr_active = false;
  std::optional<double> edid_peak;  // la crête que l'EDID annonce
  // Le pic que le système a calibré (system_peak.h : DXGI MaxLuminance, KWin), en unités du contenu ; lu
  // dès que SDL ou le système dit le HDR actif (C34-02).
  std::optional<double> system_peak;
};
// La crête quand le système n'a pas de pic calibré.
std::optional<double> uncalibrated_peak_nits(const PeakFacts& facts);
// La crête vue par le contenu : rien si ni SDL ni le système ne disent le HDR actif ; sinon le pic calibré
// par le système, sinon uncalibrated_peak_nits. Pur.
std::optional<double> detected_peak_nits(const PeakFacts& facts);
// Ce que relit le fil de l'écran (display_probe.h) : la crête, et d'où elle vient. Le contrat d'écran ne
// range en « pic du système » que la part calibrée (C34-02) ; le reste est une déduction.
struct PeakReading {
  std::optional<double> nits;        // detected_peak_nits
  std::optional<double> calibrated;  // PeakFacts::system_peak, quand c'est lui qui donne nits
};
PeakReading peak_reading(const PeakFacts& facts);
// Les faits de l'écran de la fenêtre (EDID, SDL, système), relus hors du thread qui présente.
PeakFacts peak_facts(SDL_Window* window, const DisplaySdl& sdl);
// La crête « automatique » du grade : celle de l'écran bornée à 200-4000 nits (un écran HDR de 310 nits ne
// reçoit pas une cible de 400 ; un téléviseur de 3000 nits est exploité, B37-09) ; 1000 si inconnue.
int auto_peak_nits(std::optional<double> detected);
// Le HDR suit le système, sans réglage : la sortie est en HDR10 quand l'écran en reçoit (Windows ou le
// compositeur l'a activé) et que sa crête atteint kMinHdrPeakNits. En dessous, un écran qui se dit HDR mais
// n'a pas la lumière pour les éclats (300 nits) : l'image reste en SDR, gradée. HDR coupé dans le système sur
// un écran HDR : SDR aussi. --sdr force le SDR.
inline constexpr double kMinHdrPeakNits = 350.0;
bool system_hdr(std::optional<double> peak_nits);

// Un portable débranché (SDL_GetPowerInfo) : la carte y est bridée (P3.12), le banc le note. L'Auto n'y est
// pas plafonné : son plafond reste le maximum de l'écran (décision 11).
bool on_battery();
// Ce que SDL (et, sous Windows, la configuration d'affichage) dit du HDR de l'écran de la fenêtre, pour le
// journal : de quoi comprendre pourquoi un écran HDR est vu en SDR.
std::string describe_hdr(SDL_Window* window);
// Le nom de l'écran de la fenêtre selon SDL, pour choisir son EDID.
std::string display_name(SDL_Window* window);
// L'écran de la fenêtre selon son EDID, avec son VRR armé : DRM sous Linux, NVAPI sous Windows sur une carte
// NVIDIA (window_vrr_armed). Rien si aucun EDID ne se lit.
std::optional<DisplayFacts> window_display(SDL_Window* window);
// La même, le nom de SDL déjà lu (display_sdl) : sans appel vidéo de SDL, de tout thread (display_probe.h).
std::optional<DisplayFacts> window_display(SDL_Window* window, std::string_view sdl_name);
// Le VRR armé à cet instant sur l'écran de la fenêtre : le CRTC DRM du connecteur sous Linux ; sous Windows,
// NvAPI_Disp_GetVRRInfo (G-SYNC ou G-SYNC Compatible engagé sur cet écran), relu au plus 4 fois par seconde.
// Rien : le système ne le dit pas (carte AMD ou Intel sous Windows, macOS), le retour de présentation reste.
std::optional<bool> window_vrr_armed(SDL_Window* window, std::string_view connector);
GpuFacts gpu_facts(const PlaceboScreen& screen);
// Le backend TensorRT est-il là : téléchargé (engines.lock) ou à côté du lecteur ?
bool tensorrt_backend_present(const EngineCatalog& catalog, const PlayerPaths& paths);
// Les backends présents ou téléchargeables (catalog).
EnginesHere engines_here(const PlayOptions& options, const PlayerPaths& paths, const EngineCatalog& catalog,
                         std::string_view gpu_name);

}  // namespace ajp::app::play
