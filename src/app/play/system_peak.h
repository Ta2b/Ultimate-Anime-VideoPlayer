// Le pic que le système a calibré pour l'écran, lu là où le système le garde : KDE Plasma
// (~/.config/kwinoutputconfig.json, réglages d'écran > HDR : « luminosité de crête » et « luminosité SDR ») ;
// Windows (DXGI_OUTPUT_DESC1::MaxLuminance, que l'application « Étalonnage HDR » de Windows règle). La marge
// HDR de SDL3 est relative au blanc de référence du compositeur, pas à 203 nits : sans ce blanc, le pic qui
// s'en déduit est faux dès que la luminosité SDR n'est pas 203.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ajp::app::play {

// Ce que KWin garde pour une sortie : la crête réglée à la main (« maxPeakBrightnessOverride ») et le blanc
// de référence (« sdrBrightness », ou « referenceLuminance » selon les versions), en nits.
struct KwinOutput {
  std::optional<double> peak_override;
  std::optional<double> reference;
  bool hdr = false;  // « highDynamicRange » : KWin envoie du HDR à cette sortie (C34-09)
};
// Le pic vu par le contenu sur l'écran de la fenêtre (content_peak, host_facts.h) et d'où il vient (C11-09) :
// Compositor seulement pour le pic que le système a calibré (DXGI, KDE), Headroom pour la marge de SDL
// (× blanc), Edid pour la crête de l'EDID, Default pour la valeur prudente de Windows sans rien d'autre.
struct ContentPeak {
  enum class Source : std::uint8_t { Default, Compositor, Headroom, Edid };
  double nits = 0.0;
  Source source = Source::Default;
  std::optional<double> full_frame_nits;  // Windows : MaxFullFrameLuminance de DXGI, avec le pic calibré
  friend bool operator==(const ContentPeak&, const ContentPeak&) = default;
};

// connector : « DP-1 » (le préfixe « card1- » de DRM est ôté). Rien si la sortie n'y est pas.
// edid : le début de « edidIdentifier » de l'écran branché (kwin_edid_key), vide si inconnu. KWin garde une
// entrée par écran déjà vu sur ce port : celle de cet écran l'emporte (A11-04) ; sans elle, celle où le HDR
// est actif, sinon la première.
std::optional<KwinOutput> parse_kwin_output(std::string_view json, std::string_view connector,
                                            std::string_view edid = {});
// « MSI 15066 » : le fabricant PNP et le product id en décimal, comme KWin commence « edidIdentifier ». Vide
// sans fabricant.
std::string kwin_edid_key(std::string_view manufacturer, std::uint16_t product);
// La sortie de ce connecteur DRM (« card1-DP-1 ») lue dans ~/.config/kwinoutputconfig.json ; rien hors Linux,
// sans le fichier ou sans la sortie.
std::optional<KwinOutput> read_kwin_output(std::string_view connector, std::string_view edid = {});
// La date de modification de ce fichier (ns), pour le relire quand KDE le réécrit ; rien sans lui.
std::optional<std::int64_t> kwin_config_stamp();

// Le pic calibré par le système pour l'écran de ce connecteur DRM (« card1-DP-1 »), en unités du contenu PQ
// (render/display_model.h) ; headroom : la marge HDR de SDL3. Linux : la crête de KWin, sinon marge × blanc
// de référence de KWin, ramenées au blanc de 203 du contenu ; Windows : MaxLuminance de la sortie DXGI en
// HDR. Rien si le système ne le dit pas. monitor : sous Windows, le HMONITOR de la fenêtre ; seule sa sortie
// compte (P3.57). Sans lui, la plus lumineuse des sorties en HDR. edid : sous Linux, kwin_edid_key de l'écran
// branché, comme le contrat d'écran : l'entrée KWin d'un ancien écran du même port ne compte pas (B11-04).
// Hors KDE (GNOME, Hyprland, Sway), rien : le journal le dit une fois, et le pic reste marge de SDL × 203.
// Le fichier de KDE se cherche sous ~/.config : XDG_CONFIG_HOME n'est pas lu (aucune variable
// d'environnement dans src/, B11-12).
std::optional<double> system_peak_nits(std::string_view connector, double headroom,
                                       const void* monitor = nullptr, std::string_view edid = {},
                                       double* full_frame_nits = nullptr);

}  // namespace ajp::app::play
