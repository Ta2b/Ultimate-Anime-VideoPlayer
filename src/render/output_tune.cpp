#include "render/output_tune.h"

#include <algorithm>
#include <cmath>

#include "render/display_model.h"

namespace ajp::render {
namespace {

double smoothstep(double lo, double hi, double x) {
  const double t = std::clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Le niveau d'un code 8 bits au gamma 2,4, le blanc à 1.
double code_level(double code) {
  return std::pow(code / 255.0, 2.4);
}

// La compression de KWin sans métadonnées : de 10 000 nits de contenu vers le sommet de la courbe.
constexpr double kContentTop = 10000.0;

}  // namespace

double pq::to_nits(double code) {
  const double p = std::pow(std::clamp(code, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

double pq::from_nits(double nits) {
  const double q = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
  return std::pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), 78.84375);
}

double surface_white_target(double panel_peak) {
  return std::max(kSurfaceWhiteShare * panel_peak,
                  std::min(kSurfaceWhiteFloorNits, kSurfaceWhiteMaxShare * panel_peak));
}

LookToe look_toe(double white) {
  return {.black = code_level(4.0) * white,
          .full = code_level(8.0) * white,
          .mid = code_level(32.0) * white,
          .join = code_level(48.0) * white};
}

double toe_level(const LookToe& toe, double m) {
  if (!(m > 0.0) || !(toe.join > 0.0)) return 0.0;
  return toe.join * std::pow(m / toe.join, kToeExponent) * smoothstep(toe.black, toe.full, m);
}

double toe_pile_share(const LookToe& toe, double m) {
  if (!(toe.join > 0.0) || m >= toe.join) return 1.0;
  if (!(m > 0.0)) return 0.0;
  return smoothstep(std::log(toe.mid), std::log(toe.join), std::log(m));
}

double look_neutral(const CubeLut& stack, const LookToe& toe, double nits) {
  const auto code = static_cast<float>(pq::from_nits(nits));
  const std::array<float, 3> out = sample_lut(stack, {code, code, code});
  // La luminance BT.2020 de la sortie : la pile teinte à peine les gris.
  const double pile =
      0.2627 * pq::to_nits(out[0]) + 0.6780 * pq::to_nits(out[1]) + 0.0593 * pq::to_nits(out[2]);
  const double share = toe_pile_share(toe, nits);
  return std::lerp(toe_level(toe, nits), pile, share);
}

double compress_neutral(double content_nits, double reference, double curve_peak) {
  const double k = reference / kContentReferenceNits;
  const double rel = std::max(content_nits * k / reference, 0.0);
  const double in_range = kContentTop * k / reference;
  const double out_range = curve_peak / reference;
  const double v = (out_range * (1.0 + in_range) - in_range) / (in_range * in_range);
  return rel * (1.0 + rel * v) / (1.0 + rel) * reference;
}

double curve_peak_for(double content_nits, double target, double reference) {
  const double k = reference / kContentReferenceNits;
  const double in_range = kContentTop * k / reference;
  const double rel = content_nits * k / reference;
  // rel × (1 + rel × v) / (1 + rel) = cible / référence, résolu en v ; v = 0 : Reinhard seul, v = 1 :
  // l'identité.
  const double v =
      rel > 0.0 ? std::clamp((target / reference * (1.0 + rel) / rel - 1.0) / rel, 0.0, 1.0) : 1.0;
  return reference * (v * in_range * in_range + in_range) / (1.0 + in_range);
}

double emitter_level(double m, double gain, double top) {
  const double v = m * gain;
  if (gain <= 1.0) return v;
  if (m >= top) return m;
  const double knee = std::max(0.65 * top, m);
  if (v <= knee) return v;
  const double room = std::max(top - knee, 1e-3);
  return knee + room * (1.0 - std::exp(-(v - knee) / room));
}

OutputTune output_tune(const GradeParams& params, const CubeLut* stack, double reference, double panel_peak) {
  // Le blanc d'une surface avant la pile : le blanc de la source au relèvement fixe (pas celui du plan : la
  // cible ne bouge pas à chaque coupe).
  const double source_white = params.adapt > 0.5 ? params.sdr_white : params.paper;
  const double lifted = source_white * kAnimeLift;
  const bool pile = stack != nullptr && params.look_pq > 0.5;
  const double pile_white = pile ? look_neutral(*stack, look_toe(lifted), lifted) : lifted;
  OutputTune tune;
  tune.curve_peak = curve_peak_for(pile_white, surface_white_target(panel_peak), reference);
  tune.white = std::min(compress_neutral(pile_white, reference, tune.curve_peak), panel_peak);
  // Le limiteur de la scène (le même que la passe anime) : le pic en scène sombre, la moitié en scène très
  // claire, sans jamais retirer aux éclats toute leur marge sur les surfaces.
  const double key =
      std::clamp((params.fs_apl >= 0.0 ? params.fs_apl : 0.15 * source_white) / params.sdr_white, 0.0, 1.0);
  const double limited = panel_peak * std::lerp(1.0, 0.5, smoothstep(0.30, 0.70, key));
  tune.top = std::min(std::max(limited, kEmitterMinRoom * tune.white), panel_peak);
  return tune;
}

}  // namespace ajp::render
