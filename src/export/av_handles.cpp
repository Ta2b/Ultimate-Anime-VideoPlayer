#include "export/av_handles.h"

#include <array>
#include <format>
#include <utility>

extern "C" {
#include <libavutil/error.h>
}

namespace ajp::exporter {

std::string utf8(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::string av_error_text(int error) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  av_strerror(error, text.data(), text.size());
  return text.data();
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error) {
  return refuse(code, std::format("{} : {}", where, av_error_text(error)));
}

AVPixelFormat pixel_format(Layout layout) {
  switch (layout) {
    case Layout::P010:
      return AV_PIX_FMT_P010LE;
    case Layout::Nv12:
      return AV_PIX_FMT_NV12;
    case Layout::Yuv420p10:
      return AV_PIX_FMT_YUV420P10LE;
  }
  return AV_PIX_FMT_NONE;
}

AvColor av_color(const ColorInfo& color) {
  AvColor out;
  switch (color.primaries) {
    case Primaries::Bt709:
      out.primaries = AVCOL_PRI_BT709;
      break;
    case Primaries::Bt601_625:
      out.primaries = AVCOL_PRI_BT470BG;
      break;
    case Primaries::Bt601_525:
      out.primaries = AVCOL_PRI_SMPTE170M;
      break;
    case Primaries::Bt2020:
      out.primaries = AVCOL_PRI_BT2020;
      break;
    case Primaries::DciP3:
      out.primaries = AVCOL_PRI_SMPTE431;
      break;
  }
  switch (color.transfer) {
    case Transfer::Bt1886:
      out.transfer = AVCOL_TRC_BT709;
      break;
    case Transfer::Srgb:
      out.transfer = AVCOL_TRC_IEC61966_2_1;
      break;
    case Transfer::Linear:
      out.transfer = AVCOL_TRC_LINEAR;
      break;
    case Transfer::Pq:
      out.transfer = AVCOL_TRC_SMPTE2084;
      break;
    case Transfer::Hlg:
      out.transfer = AVCOL_TRC_ARIB_STD_B67;
      break;
  }
  switch (color.matrix) {
    case ColorMatrix::Bt601:
      out.matrix = AVCOL_SPC_BT470BG;
      break;
    case ColorMatrix::Bt2020Ncl:
      out.matrix = AVCOL_SPC_BT2020_NCL;
      break;
    case ColorMatrix::Bt709:
    case ColorMatrix::Rgb:  // les images converties sont toujours en YUV
      out.matrix = AVCOL_SPC_BT709;
      break;
  }
  out.range = color.range == ColorRange::Full ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
  switch (color.siting) {
    case ChromaSiting::Left:
      out.siting = AVCHROMA_LOC_LEFT;
      break;
    case ChromaSiting::Center:
      out.siting = AVCHROMA_LOC_CENTER;
      break;
    case ChromaSiting::TopLeft:
      out.siting = AVCHROMA_LOC_TOPLEFT;
      break;
  }
  return out;
}

ColorInfo color_of(const AVCodecParameters& video) {
  ColorInfo out;  // niveaux vidéo, BT.709, BT.1886, chroma à gauche
  const bool sd = video.width < 1280 && video.height <= 576;
  switch (video.color_space) {
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      out.matrix = ColorMatrix::Bt601;
      break;
    case AVCOL_SPC_BT2020_NCL:
      out.matrix = ColorMatrix::Bt2020Ncl;
      break;
    case AVCOL_SPC_BT709:
    case AVCOL_SPC_RGB:
      break;
    default:
      if (sd) out.matrix = ColorMatrix::Bt601;
      break;
  }
  switch (video.color_primaries) {
    case AVCOL_PRI_BT709:
      break;
    case AVCOL_PRI_BT470BG:
      out.primaries = Primaries::Bt601_625;
      break;
    case AVCOL_PRI_SMPTE170M:
    case AVCOL_PRI_SMPTE240M:
      out.primaries = Primaries::Bt601_525;
      break;
    case AVCOL_PRI_BT2020:
      out.primaries = Primaries::Bt2020;
      break;
    case AVCOL_PRI_SMPTE431:
    case AVCOL_PRI_SMPTE432:
      out.primaries = Primaries::DciP3;
      break;
    default:
      if (out.matrix == ColorMatrix::Bt2020Ncl) {
        out.primaries = Primaries::Bt2020;
      } else if (out.matrix == ColorMatrix::Bt601) {
        out.primaries = video.height == 576 ? Primaries::Bt601_625 : Primaries::Bt601_525;
      }
      break;
  }
  switch (video.color_trc) {
    case AVCOL_TRC_SMPTE2084:
      out.transfer = Transfer::Pq;
      break;
    case AVCOL_TRC_ARIB_STD_B67:
      out.transfer = Transfer::Hlg;
      break;
    case AVCOL_TRC_IEC61966_2_1:
      out.transfer = Transfer::Srgb;
      break;
    case AVCOL_TRC_LINEAR:
      out.transfer = Transfer::Linear;
      break;
    default:
      break;
  }
  switch (video.chroma_location) {
    case AVCHROMA_LOC_CENTER:
      out.siting = ChromaSiting::Center;
      break;
    case AVCHROMA_LOC_TOPLEFT:
      out.siting = ChromaSiting::TopLeft;
      break;
    default:
      break;
  }
  return out;
}

}  // namespace ajp::exporter
