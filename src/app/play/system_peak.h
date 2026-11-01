// Le pic que le système a calibré pour l'écran, lu là où le système le garde : KDE Plasma
// (~/.config/kwinoutputconfig.json, réglages d'écran > HDR : « luminosité de crête » et « luminosité SDR ») ;
// Windows (DXGI_OUTPUT_DESC1::MaxLuminance, que l'application « Étalonnage HDR » de Windows règle). La marge
// HDR de SDL3 est relative au blanc de référence du compositeur, pas à 203 nits : sans ce blanc, le pic qui
// s'en déduit est faux dès que la luminosité SDR n'est pas 203.
#pragma once

#include <optional>
#include <string_view>

namespace ajp::app::play {

// Ce que KWin garde pour une sortie : la crête réglée à la main (« maxPeakBrightnessOverride ») et le blanc
// de référence (« sdrBrightness », ou « referenceLuminance » selon les versions), en nits.
struct KwinOutput {
  std::optional<double> peak_override;
  std::optional<double> reference;
};
// connector : « DP-1 » (le préfixe « card1- » de DRM est ôté). Rien si la sortie n'y est pas.
std::optional<KwinOutput> parse_kwin_output(std::string_view json, std::string_view connector);

// Le pic calibré par le système pour l'écran de ce connecteur DRM (« card1-DP-1 ») ; headroom : la marge HDR
// de SDL3. Linux : la crête de KWin, sinon marge × blanc de référence de KWin ; Windows : MaxLuminance de
// la sortie DXGI en HDR. Rien si le système ne le dit pas.
std::optional<double> system_peak_nits(std::string_view connector, double headroom);

}  // namespace ajp::app::play
