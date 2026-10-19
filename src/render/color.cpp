#include "render/color.h"

#include <libplacebo/colorspace.h>

namespace ajp::render {
namespace {

pl_color_primaries to_placebo(Primaries primaries) {
  switch (primaries) {
    case Primaries::Bt709:
      return PL_COLOR_PRIM_BT_709;
    case Primaries::Bt601_625:
      return PL_COLOR_PRIM_BT_601_625;
    case Primaries::Bt601_525:
      return PL_COLOR_PRIM_BT_601_525;
    case Primaries::Bt2020:
      return PL_COLOR_PRIM_BT_2020;
    case Primaries::DciP3:
      return PL_COLOR_PRIM_DCI_P3;  // AVCOL_PRI_SMPTE431, blanc DCI
  }
  return PL_COLOR_PRIM_UNKNOWN;
}

pl_color_transfer to_placebo(Transfer transfer) {
  switch (transfer) {
    case Transfer::Bt1886:
      return PL_COLOR_TRC_BT_1886;
    case Transfer::Srgb:
      return PL_COLOR_TRC_SRGB;
    case Transfer::Linear:
      return PL_COLOR_TRC_LINEAR;
    case Transfer::Pq:
      return PL_COLOR_TRC_PQ;
    case Transfer::Hlg:
      return PL_COLOR_TRC_HLG;
  }
  return PL_COLOR_TRC_UNKNOWN;
}

pl_hdr_metadata to_placebo(const HdrMetadata& hdr) {
  pl_hdr_metadata out{};
  out.prim.red = {.x = hdr.primaries_xy[0], .y = hdr.primaries_xy[1]};
  out.prim.green = {.x = hdr.primaries_xy[2], .y = hdr.primaries_xy[3]};
  out.prim.blue = {.x = hdr.primaries_xy[4], .y = hdr.primaries_xy[5]};
  out.prim.white = {.x = hdr.white_xy[0], .y = hdr.white_xy[1]};
  out.min_luma = hdr.min_luminance;
  out.max_luma = hdr.max_luminance;
  out.max_cll = hdr.max_cll;
  out.max_fall = hdr.max_fall;
  return out;
}

}  // namespace

SlotColor slot_color(const ColorInfo& source, core::HdrMode route, double display_peak_nits) {
  if (route == core::HdrMode::TrueHdr && !is_hdr(source.transfer)) {
    HdrMetadata tagged{};
    tagged.max_luminance = static_cast<float>(display_peak_nits);
    return SlotColor{.primaries = Primaries::Bt2020, .transfer = Transfer::Pq, .hdr = tagged};
  }
  return SlotColor{.primaries = source.primaries, .transfer = source.transfer, .hdr = source.hdr};
}

namespace detail {

pl_color_space to_placebo(const SlotColor& color, double sdr_white_nits) {
  pl_color_space out{};
  out.primaries = render::to_placebo(color.primaries);
  out.transfer = render::to_placebo(color.transfer);
  if (color.hdr) out.hdr = render::to_placebo(*color.hdr);
  if (!is_hdr(color.transfer) && sdr_white_nits > 0.0) out.hdr.max_luma = static_cast<float>(sdr_white_nits);
  return out;
}

pl_color_space to_placebo(const core::DisplayInfo& display) {
  pl_color_space out = pl_color_space_srgb;
  if (display.hdr && display.space == core::OutputSpace::Pq) {
    out = pl_color_space{.primaries = PL_COLOR_PRIM_BT_2020, .transfer = PL_COLOR_TRC_PQ};
  } else if (display.hdr && display.space == core::OutputSpace::ScRgb) {
    out = pl_color_space{.primaries = PL_COLOR_PRIM_BT_709, .transfer = PL_COLOR_TRC_LINEAR};
  } else {
    return out;
  }
  out.hdr.max_luma = static_cast<float>(display.peak_nits);
  out.hdr.min_luma = static_cast<float>(display.min_nits);
  return out;
}

}  // namespace detail

}  // namespace ajp::render
