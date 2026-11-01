// La mire de calibration du HDR (hdr_pattern.h) : deux rectangles pleins écrits par pl_dispatch dans la
// swapchain, au PQ exact de leurs nits. Rien d'autre ne passe entre la valeur et l'écran : ni grade, ni
// adaptation de tons, ni mélange.
#include <array>
#include <string>

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>

#include "app/play/hdr_pattern.h"
#include "app/play/placebo_impl.h"

namespace ajp::app::play {

std::expected<void, Refusal> PlaceboScreen::Impl::draw_pattern(const HdrPattern& pattern,
                                                               const pl_frame& target) {
  const pl_tex tex = target.planes[0].texture;
  if (target.color.transfer != PL_COLOR_TRC_PQ)
    return std::unexpected(Refusal{RefusalCode::HdrUnavailable, "mire HDR : la sortie n'est pas en PQ"});
  if (pattern_dispatch == nullptr) pattern_dispatch = pl_dispatch_create(log, vk->gpu);
  pl_dispatch_reset_frame(pattern_dispatch);
  const PatternLayout layout = pattern_layout(OutputSize{tex->params.w, tex->params.h}, pattern);
  const auto fill = [&](const PatternRect& r, int nits) -> std::expected<void, Refusal> {
    const auto level = static_cast<float>(pq_of_nits(nits));
    const pl_shader_var var{.var = pl_var_float("ajp_level"), .data = &level, .dynamic = true};
    pl_custom_shader custom{};
    custom.description = "mire HDR";
    custom.body = "color = vec4(vec3(ajp_level), 1.0);";
    custom.input = PL_SHADER_SIG_NONE;
    custom.output = PL_SHADER_SIG_COLOR;
    custom.variables = &var;
    custom.num_variables = 1;
    pl_shader sh = pl_dispatch_begin(pattern_dispatch);
    if (!pl_shader_custom(sh, &custom)) {
      pl_dispatch_abort(pattern_dispatch, &sh);
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_shader_custom (mire HDR)"});
    }
    pl_color_repr repr = target.repr;
    pl_shader_encode_color(sh, &repr);
    pl_rect2d rect{r.x0, r.y0, r.x1, r.y1};
    if (target.planes[0].flipped) {
      rect.y0 = tex->params.h - r.y1;
      rect.y1 = tex->params.h - r.y0;
    }
    pl_dispatch_params params{};
    params.shader = &sh;
    params.target = tex;
    params.rect = rect;
    if (!pl_dispatch_finish(pattern_dispatch, &params))
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_dispatch_finish (mire HDR)"});
    return {};
  };
  if (auto drawn = fill(layout.frame, pattern.frame_nits); !drawn) return drawn;
  return fill(layout.square, pattern.nits);
}

}  // namespace ajp::app::play
