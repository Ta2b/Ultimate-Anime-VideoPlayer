// Les règles de devinette des champs non renseignés sont reprises de
// the-database-mpv video/mp_image.c:998-1019 (matrice, plage, primaires,
// transfert), 1054-1058 (HDR10 ignoré en SDR), 1060-1069 (siting de chroma) et
// video/csputils.c:192-214 (mp_csp_guess_colorspace, mp_csp_guess_primaries).
// Comportement fixé par tests/unit/media/color_test.cpp.
#include "media/color.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <utility>

extern "C" {
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

#include "media/av_util.h"
#include "media/log.h"
#include "media/track.h"

namespace ajp::media {
namespace {

std::string name_or_number(const char* name, int value) {
  return name != nullptr ? std::string(name) : std::to_string(value);
}

bool is_rgb(AVPixelFormat format) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  return desc != nullptr && (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0;
}

bool is_subsampled(AVPixelFormat format) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  return desc != nullptr && (desc->log2_chroma_w > 0 || desc->log2_chroma_h > 0);
}

// Une matrice voisine d'une du vocabulaire la remplace, avec un avertissement par matrice et par
// processus (la couleur est lue à chaque image) : un léger écart de teinte vaut mieux qu'un
// fichier refusé. Les autres (ICtCp et IPT de Dolby Vision, YCgCo) donneraient des couleurs
// fausses : refusées.
ColorMatrix stand_in(AVColorSpace space, ColorMatrix used) {
  static std::atomic<std::uint64_t> warned{0};
  const std::uint64_t bit = std::uint64_t{1} << (static_cast<unsigned>(space) & 63U);
  if ((warned.fetch_or(bit) & bit) == 0) {
    make_logger()->warn("matrice {} lue comme {} : couleurs approchées",
                        name_or_number(av_color_space_name(space), static_cast<int>(space)),
                        used == ColorMatrix::Bt2020Ncl ? "BT.2020 NCL"
                        : used == ColorMatrix::Bt709   ? "BT.709"
                                                       : "BT.601");
  }
  return used;
}

// Un champ voisin d'une valeur du vocabulaire la remplace, avertie une fois par genre et par valeur
// et par processus : primaires, transfert et siting ne valent pas un fichier refusé.
void approximated(std::atomic<std::uint64_t>& warned, int value, const char* kind, const std::string& name,
                  const char* used) {
  const std::uint64_t bit = std::uint64_t{1} << (static_cast<unsigned>(value) & 63U);
  if ((warned.fetch_or(bit) & bit) == 0)
    make_logger()->warn("{} {} lu comme {} : couleurs approchées", kind, name, used);
}

std::expected<ColorMatrix, Refusal> map_matrix(AVColorSpace space, AVColorPrimaries primaries, int width,
                                               int height) {
  const ColorMatrix guessed = width >= 1280 || height > 576 ? ColorMatrix::Bt709 : ColorMatrix::Bt601;
  switch (space) {
    case AVCOL_SPC_BT709:
      return ColorMatrix::Bt709;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      return ColorMatrix::Bt601;
    case AVCOL_SPC_BT2020_NCL:
      return ColorMatrix::Bt2020Ncl;
    case AVCOL_SPC_RGB:
      return ColorMatrix::Rgb;
    case AVCOL_SPC_UNSPECIFIED:
      return guessed;
    case AVCOL_SPC_BT2020_CL:  // mêmes coefficients, appliqués à la lumière linéaire
      return stand_in(space, ColorMatrix::Bt2020Ncl);
    case AVCOL_SPC_SMPTE240M:  // Kr 0,212 et Kb 0,087 contre 0,2126 et 0,0722
      return stand_in(space, ColorMatrix::Bt709);
    case AVCOL_SPC_FCC:  // Kr 0,30 et Kb 0,11 contre 0,299 et 0,114
      return stand_in(space, ColorMatrix::Bt601);
    case AVCOL_SPC_CHROMA_DERIVED_NCL:  // dérivée des primaires : la matrice qui les accompagne
    case AVCOL_SPC_CHROMA_DERIVED_CL:
      return stand_in(space, primaries == AVCOL_PRI_BT2020  ? ColorMatrix::Bt2020Ncl
                             : primaries == AVCOL_PRI_BT709 ? ColorMatrix::Bt709
                                                            : guessed);
    default:
      return refuse(RefusalCode::FormatUnsupported,
                    "matrice " + name_or_number(av_color_space_name(space), static_cast<int>(space)));
  }
}

std::expected<Primaries, Refusal> map_primaries(AVColorPrimaries primaries, ColorMatrix matrix, int width,
                                                int height) {
  switch (primaries) {
    case AVCOL_PRI_BT709:
      return Primaries::Bt709;
    case AVCOL_PRI_BT470BG:
      return Primaries::Bt601_625;
    case AVCOL_PRI_SMPTE170M:
    case AVCOL_PRI_SMPTE240M:  // mêmes primaires, voir pixfmt.h
      return Primaries::Bt601_525;
    case AVCOL_PRI_BT2020:
      return Primaries::Bt2020;
    case AVCOL_PRI_SMPTE431:
      return Primaries::DciP3;
    case AVCOL_PRI_UNSPECIFIED:
      if (matrix == ColorMatrix::Bt2020Ncl) return Primaries::Bt2020;
      if (matrix != ColorMatrix::Bt601 || width >= 1280 || height > 576) return Primaries::Bt709;
      if (height == 576) return Primaries::Bt601_625;
      if (height == 480 || height == 486) return Primaries::Bt601_525;
      return Primaries::Bt709;
    default: {  // BT.470M, FILM, SMPTE428, SMPTE432 (P3 D65), EBU3213 : les plus proches
      static std::atomic<std::uint64_t> warned{0};
      const bool p3 = primaries == AVCOL_PRI_SMPTE428 || primaries == AVCOL_PRI_SMPTE432;
      const Primaries used = p3                               ? Primaries::DciP3
                             : primaries == AVCOL_PRI_EBU3213 ? Primaries::Bt601_625
                             : primaries == AVCOL_PRI_BT470M  ? Primaries::Bt601_525
                                                              : Primaries::Bt709;
      approximated(warned, static_cast<int>(primaries), "primaires",
                   name_or_number(av_color_primaries_name(primaries), static_cast<int>(primaries)),
                   p3                             ? "DCI-P3"
                   : used == Primaries::Bt601_625 ? "BT.601 625"
                   : used == Primaries::Bt601_525 ? "BT.601 525"
                                                  : "BT.709");
      return used;
    }
  }
}

std::expected<Transfer, Refusal> map_transfer(AVColorTransferCharacteristic trc, ColorRange range) {
  switch (trc) {
    // Courbes SDR de prise de vue : toutes affichées en BT.1886, comme le fork.
    case AVCOL_TRC_BT709:
    case AVCOL_TRC_SMPTE170M:
    case AVCOL_TRC_GAMMA22:
    case AVCOL_TRC_GAMMA28:
    case AVCOL_TRC_BT2020_10:
    case AVCOL_TRC_BT2020_12:
      return Transfer::Bt1886;
    case AVCOL_TRC_IEC61966_2_1:
      return Transfer::Srgb;
    case AVCOL_TRC_LINEAR:
      return Transfer::Linear;
    case AVCOL_TRC_SMPTE2084:
      return Transfer::Pq;
    case AVCOL_TRC_ARIB_STD_B67:
      return Transfer::Hlg;
    case AVCOL_TRC_UNSPECIFIED:
      return range == ColorRange::Limited ? Transfer::Bt1886 : Transfer::Srgb;
    case AVCOL_TRC_SMPTE240M:  // proches de BT.709 : l'écart se lit à peine
    case AVCOL_TRC_IEC61966_2_4:
    case AVCOL_TRC_BT1361_ECG: {
      static std::atomic<std::uint64_t> warned{0};
      approximated(warned, static_cast<int>(trc), "transfert",
                   name_or_number(av_color_transfer_name(trc), static_cast<int>(trc)), "BT.1886");
      return Transfer::Bt1886;
    }
    default:  // LOG, LOG_SQRT, SMPTE428 : une courbe fausse changerait toute l'image
      return refuse(RefusalCode::FormatUnsupported,
                    "transfert " + name_or_number(av_color_transfer_name(trc), static_cast<int>(trc)));
  }
}

std::expected<ChromaSiting, Refusal> map_siting(AVChromaLocation location, ColorRange range) {
  switch (location) {
    case AVCHROMA_LOC_UNSPECIFIED:  // MPEG-2/4 et H.264 en plage limitée, JPEG en plage pleine
      return range == ColorRange::Full ? ChromaSiting::Center : ChromaSiting::Left;
    case AVCHROMA_LOC_LEFT:
      return ChromaSiting::Left;
    case AVCHROMA_LOC_CENTER:
      return ChromaSiting::Center;
    case AVCHROMA_LOC_TOPLEFT:
      return ChromaSiting::TopLeft;
    default: {  // TOP, BOTTOM, BOTTOMLEFT : un demi-pixel de chroma d'écart, au plus
      static std::atomic<std::uint64_t> warned{0};
      const bool left = location == AVCHROMA_LOC_BOTTOMLEFT;
      approximated(warned, static_cast<int>(location), "siting",
                   name_or_number(av_chroma_location_name(location), static_cast<int>(location)),
                   left ? "gauche" : "centre");
      return left ? ChromaSiting::Left : ChromaSiting::Center;
    }
  }
}

float to_float(AVRational value) {
  return static_cast<float>(av_q2d(value));
}

bool within(float value, float lo, float hi) {
  return std::isfinite(value) && lo <= value && value <= hi;
}

}  // namespace

std::expected<PixelFormat, Refusal> map_pixel_format(AVPixelFormat format) {
  switch (format) {
    case AV_PIX_FMT_NV12:
      return PixelFormat::Nv12;
    case AV_PIX_FMT_P010LE:
      return PixelFormat::P010;
    case AV_PIX_FMT_P016LE:
      return PixelFormat::P016;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
      return PixelFormat::Yuv420p;
    case AV_PIX_FMT_YUV420P10LE:
      return PixelFormat::Yuv420p10;
    case AV_PIX_FMT_YUV444P16LE:
      return PixelFormat::Yuv444p16;
    case AV_PIX_FMT_RGBA64LE:
      return PixelFormat::Rgba16Unorm;
    case AV_PIX_FMT_RGBA:
      return PixelFormat::Rgba8Unorm;
    default:
      return refuse(
          RefusalCode::FormatUnsupported,
          "format de pixel " + name_or_number(av_get_pix_fmt_name(format), static_cast<int>(format)));
  }
}

std::expected<ColorInfo, Refusal> color_info(const AVFrame& frame, AVPixelFormat sw_format) {
  ColorInfo color;
  const bool rgb = is_rgb(sw_format);
  if (frame.color_range == AVCOL_RANGE_JPEG || sw_format == AV_PIX_FMT_YUVJ420P || rgb) {
    color.range = ColorRange::Full;
  }
  if (rgb) {
    color.matrix = ColorMatrix::Rgb;  // des pixels RGB n'ont pas de matrice YUV
  } else {
    auto matrix = map_matrix(frame.colorspace, frame.color_primaries, frame.width, frame.height);
    if (!matrix) return std::unexpected(matrix.error());
    color.matrix = *matrix;
  }
  auto primaries = map_primaries(frame.color_primaries, color.matrix, frame.width, frame.height);
  if (!primaries) return std::unexpected(primaries.error());
  color.primaries = *primaries;
  auto transfer = map_transfer(frame.color_trc, color.range);
  if (!transfer) return std::unexpected(transfer.error());
  color.transfer = *transfer;
  if (is_subsampled(sw_format)) {
    auto siting = map_siting(frame.chroma_location, color.range);
    if (!siting) return std::unexpected(siting.error());
    color.siting = *siting;
  } else {
    color.siting = ChromaSiting::Center;  // sans sous-échantillonnage, chroma et luma coïncident
  }
  // Une source SDR garde parfois les métadonnées HDR10 de son master : elles ne
  // décrivent pas ses pixels et sont ignorées, comme mpv (mp_image.c:1054-1058).
  if (color.transfer != Transfer::Pq && color.transfer != Transfer::Hlg) return color;
  auto hdr = hdr_metadata(frame);
  if (!hdr) {  // métadonnées indicatives aberrantes : l'image reste lisible sans elles
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) make_logger()->warn("métadonnées HDR10 ignorées : {}", hdr.error().detail);
    return color;
  }
  color.hdr = *hdr;
  return color;
}

std::expected<std::optional<HdrMetadata>, Refusal> hdr_metadata(const AVFrame& frame) {
  const AVFrameSideData* mastering = av_frame_get_side_data(&frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
  const AVFrameSideData* light = av_frame_get_side_data(&frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
  if (mastering == nullptr && light == nullptr) return std::nullopt;

  HdrMetadata hdr;
  if (mastering != nullptr) {
    const auto& display = *reinterpret_cast<const AVMasteringDisplayMetadata*>(mastering->data);
    if (display.has_primaries != 0) {
      for (std::size_t i = 0; i < 3; ++i) {
        hdr.primaries_xy[2 * i] = to_float(display.display_primaries[i][0]);
        hdr.primaries_xy[2 * i + 1] = to_float(display.display_primaries[i][1]);
      }
      hdr.white_xy = {to_float(display.white_point[0]), to_float(display.white_point[1])};
    }
    if (display.has_luminance != 0) {
      hdr.max_luminance = to_float(display.max_luminance);
      hdr.min_luminance = to_float(display.min_luminance);
    }
  }
  if (light != nullptr) {
    const auto& level = *reinterpret_cast<const AVContentLightMetadata*>(light->data);
    hdr.max_cll = static_cast<float>(level.MaxCLL);
    hdr.max_fall = static_cast<float>(level.MaxFALL);
  }

  for (const float xy : hdr.primaries_xy) {
    if (!within(xy, 0.0F, 1.0F))
      return refuse(RefusalCode::FormatUnsupported, std::format("HDR10 : primaire {}", xy));
  }
  for (const float xy : hdr.white_xy) {
    if (!within(xy, 0.0F, 1.0F))
      return refuse(RefusalCode::FormatUnsupported, std::format("HDR10 : blanc {}", xy));
  }
  constexpr float kMaxNits = 10000.0F;  // plafond de PQ
  if (!within(hdr.max_luminance, 0.0F, kMaxNits) || !within(hdr.min_luminance, 0.0F, kMaxNits) ||
      (hdr.max_luminance > 0.0F && hdr.min_luminance >= hdr.max_luminance)) {
    return refuse(RefusalCode::FormatUnsupported, std::format("HDR10 : luminance de mastering [{}, {}] nits",
                                                              hdr.min_luminance, hdr.max_luminance));
  }
  if (!within(hdr.max_cll, 0.0F, kMaxNits) || !within(hdr.max_fall, 0.0F, kMaxNits)) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("HDR10 : MaxCLL {} et MaxFALL {} nits", hdr.max_cll, hdr.max_fall));
  }
  return hdr;
}

std::optional<ColorInfo> track_color(const TrackInfo& track) {
  if (track.kind != TrackKind::Video || !track.parameters) return std::nullopt;
  const AVCodecParameters& par = track.parameters->av();
  FramePtr frame(av_frame_alloc());
  if (!frame) return std::nullopt;
  frame->width = par.width;
  frame->height = par.height;
  frame->colorspace = par.color_space;
  frame->color_range = par.color_range;
  frame->color_primaries = par.color_primaries;
  frame->color_trc = par.color_trc;
  frame->chroma_location = par.chroma_location;
  // Les métadonnées HDR10 du flux, quand le conteneur les donne : les mêmes que celles que le décodeur
  // posera sur les images.
  static constexpr std::pair<AVPacketSideDataType, AVFrameSideDataType> kHdrData[] = {
      {AV_PKT_DATA_MASTERING_DISPLAY_METADATA, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA},
      {AV_PKT_DATA_CONTENT_LIGHT_LEVEL, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL}};
  for (const auto& [packet_type, frame_type] : kHdrData) {
    const AVPacketSideData* data =
        av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data, packet_type);
    if (data == nullptr) continue;
    if (AVFrameSideData* copy = av_frame_new_side_data(frame.get(), frame_type, data->size))
      std::memcpy(copy->data, data->data, data->size);
  }
  auto color = color_info(*frame, static_cast<AVPixelFormat>(par.format));
  if (!color) return std::nullopt;
  return *color;
}

}  // namespace ajp::media
