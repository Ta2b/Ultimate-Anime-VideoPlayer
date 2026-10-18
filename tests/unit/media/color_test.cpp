// Fixe les règles de couleur reprises de the-database-mpv video/mp_image.c:998-1019
// et video/csputils.c:192-214, et les refus de ce que core/frame.h ne sait pas dire.
#include "media/color.h"

#include <cstring>
#include <memory>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/mastering_display_metadata.h>
}

#include "media/av_util.h"

using namespace ajp::media;
using ajp::ChromaSiting;
using ajp::ColorMatrix;
using ajp::ColorRange;
using ajp::PixelFormat;
using ajp::Primaries;
using ajp::RefusalCode;
using ajp::Transfer;

namespace {

FramePtr frame(int width, int height, AVPixelFormat format = AV_PIX_FMT_YUV420P) {
  FramePtr f(av_frame_alloc());
  f->width = width;
  f->height = height;
  f->format = format;
  return f;
}

std::optional<RefusalCode> refusal_of(const AVFrame& f) {
  auto color = color_info(f, static_cast<AVPixelFormat>(f.format));
  return color ? std::nullopt : std::optional(color.error().code);
}

AVMasteringDisplayMetadata* add_mastering(AVFrame& f) {
  return av_mastering_display_metadata_create_side_data(&f);
}

}  // namespace

TEST_CASE("color : formats de pixel du vocabulaire, les autres refusés", "[media][color]") {
  CHECK(map_pixel_format(AV_PIX_FMT_NV12) == PixelFormat::Nv12);
  CHECK(map_pixel_format(AV_PIX_FMT_P010LE) == PixelFormat::P010);
  CHECK(map_pixel_format(AV_PIX_FMT_P016LE) == PixelFormat::P016);
  CHECK(map_pixel_format(AV_PIX_FMT_YUV420P) == PixelFormat::Yuv420p);
  CHECK(map_pixel_format(AV_PIX_FMT_YUVJ420P) == PixelFormat::Yuv420p);
  CHECK(map_pixel_format(AV_PIX_FMT_YUV420P10LE) == PixelFormat::Yuv420p10);
  CHECK(map_pixel_format(AV_PIX_FMT_YUV444P16LE) == PixelFormat::Yuv444p16);
  CHECK(map_pixel_format(AV_PIX_FMT_RGBA64LE) == PixelFormat::Rgba16Unorm);
  CHECK(map_pixel_format(AV_PIX_FMT_RGBA) == PixelFormat::Rgba8Unorm);
  auto refused = map_pixel_format(AV_PIX_FMT_YUV422P);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FormatUnsupported);
  CHECK(refused.error().detail.find("yuv422p") != std::string::npos);
}

TEST_CASE("color : une source non étiquetée est devinée par sa taille, comme mpv", "[media][color]") {
  const auto guess = [](int w, int h) { return *color_info(*frame(w, h), AV_PIX_FMT_YUV420P); };
  CHECK(guess(1920, 1080).matrix == ColorMatrix::Bt709);
  CHECK(guess(1920, 1080).primaries == Primaries::Bt709);
  CHECK(guess(1280, 400).matrix == ColorMatrix::Bt709);  // largeur HD
  CHECK(guess(720, 576).matrix == ColorMatrix::Bt601);
  CHECK(guess(720, 576).primaries == Primaries::Bt601_625);
  CHECK(guess(720, 480).primaries == Primaries::Bt601_525);
  CHECK(guess(720, 486).primaries == Primaries::Bt601_525);
  CHECK(guess(640, 360).primaries == Primaries::Bt709);  // ni PAL ni NTSC
  CHECK(guess(720, 480).transfer == Transfer::Bt1886);
  CHECK(guess(720, 480).range == ColorRange::Limited);
  CHECK(guess(720, 480).siting == ChromaSiting::Left);
}

TEST_CASE("color : primaires devinées d'après la matrice", "[media][color]") {
  FramePtr f = frame(1920, 1080);
  f->colorspace = AVCOL_SPC_BT2020_NCL;
  CHECK(color_info(*f, AV_PIX_FMT_YUV420P)->primaries == Primaries::Bt2020);
  f->colorspace = AVCOL_SPC_SMPTE170M;
  f->height = 576;
  f->width = 720;
  CHECK(color_info(*f, AV_PIX_FMT_YUV420P)->primaries == Primaries::Bt601_625);
}

TEST_CASE("color : plage pleine pour YUVJ et RGB, RGB en sRGB sans matrice", "[media][color]") {
  CHECK(color_info(*frame(64, 48, AV_PIX_FMT_YUVJ420P), AV_PIX_FMT_YUVJ420P)->range == ColorRange::Full);
  FramePtr jpeg = frame(64, 48);
  jpeg->color_range = AVCOL_RANGE_JPEG;
  const auto full = *color_info(*jpeg, AV_PIX_FMT_YUV420P);
  CHECK(full.range == ColorRange::Full);
  CHECK(full.transfer == Transfer::Srgb);  // transfert inconnu en plage pleine

  FramePtr rgb = frame(64, 48, AV_PIX_FMT_RGBA);
  rgb->colorspace = AVCOL_SPC_BT709;  // une étiquette YUV sur des pixels RGB est sans objet
  const auto color = *color_info(*rgb, AV_PIX_FMT_RGBA);
  CHECK(color.matrix == ColorMatrix::Rgb);
  CHECK(color.range == ColorRange::Full);
  CHECK(color.transfer == Transfer::Srgb);
}

TEST_CASE("color : étiquettes reconnues", "[media][color]") {
  FramePtr f = frame(3840, 2160);
  f->colorspace = AVCOL_SPC_BT2020_NCL;
  f->color_primaries = AVCOL_PRI_BT2020;
  f->color_trc = AVCOL_TRC_ARIB_STD_B67;
  f->chroma_location = AVCHROMA_LOC_CENTER;
  const auto hlg = *color_info(*f, AV_PIX_FMT_YUV420P10LE);
  CHECK(hlg.transfer == Transfer::Hlg);
  CHECK(hlg.siting == ChromaSiting::Center);
  f->color_trc = AVCOL_TRC_GAMMA28;  // BT.470 BG, courant sur les DVD PAL
  CHECK(color_info(*f, AV_PIX_FMT_YUV420P10LE)->transfer == Transfer::Bt1886);
  f->color_trc = AVCOL_TRC_LINEAR;
  CHECK(color_info(*f, AV_PIX_FMT_YUV420P10LE)->transfer == Transfer::Linear);
  f->color_primaries = AVCOL_PRI_SMPTE431;
  CHECK(color_info(*f, AV_PIX_FMT_YUV420P10LE)->primaries == Primaries::DciP3);
}

TEST_CASE("color : ce que core/frame.h ne sait pas dire est refusé", "[media][color]") {
  FramePtr f = frame(1920, 1080);
  f->colorspace = AVCOL_SPC_SMPTE240M;
  CHECK(refusal_of(*f) == RefusalCode::FormatUnsupported);
  f->colorspace = AVCOL_SPC_BT709;
  f->color_trc = AVCOL_TRC_LOG;
  CHECK(refusal_of(*f) == RefusalCode::FormatUnsupported);
  f->color_trc = AVCOL_TRC_BT709;
  f->color_primaries = AVCOL_PRI_SMPTE432;  // P3 D65 : pas DCI-P3
  CHECK(refusal_of(*f) == RefusalCode::FormatUnsupported);
  f->color_primaries = AVCOL_PRI_BT709;
  f->chroma_location = AVCHROMA_LOC_BOTTOM;
  CHECK(refusal_of(*f) == RefusalCode::FormatUnsupported);
  // Sans sous-échantillonnage, le siting est sans objet.
  FramePtr full = frame(1920, 1080, AV_PIX_FMT_YUV444P16LE);
  full->chroma_location = AVCHROMA_LOC_BOTTOM;
  CHECK(refusal_of(*full) == std::nullopt);
}

TEST_CASE("color : métadonnées HDR10 lues, valeurs hors bornes physiques refusées", "[media][color]") {
  FramePtr f = frame(3840, 2160);
  auto none = hdr_metadata(*f);
  REQUIRE(none.has_value());
  CHECK_FALSE(none->has_value());

  AVContentLightMetadata* light = av_content_light_metadata_create_side_data(f.get());
  light->MaxCLL = 800;
  light->MaxFALL = 200;
  auto only_light = hdr_metadata(*f);
  REQUIRE(only_light.has_value());
  REQUIRE(only_light->has_value());
  CHECK((*only_light)->max_cll == 800.0F);
  CHECK((*only_light)->max_luminance == 0.0F);  // inconnu

  AVMasteringDisplayMetadata* mastering = add_mastering(*f);
  mastering->has_luminance = 1;
  mastering->max_luminance = av_make_q(4000, 1);
  mastering->min_luminance = av_make_q(5, 1000);
  CHECK((*hdr_metadata(*f))->max_luminance == 4000.0F);

  mastering->min_luminance = av_make_q(5000, 1);  // au-dessus du maximum
  CHECK_FALSE(hdr_metadata(*f));
  mastering->min_luminance = av_make_q(0, 1);
  mastering->has_primaries = 1;
  mastering->display_primaries[0][0] = av_make_q(3, 2);  // x > 1
  auto refused = hdr_metadata(*f);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FormatUnsupported);
  mastering->display_primaries[0][0] = av_make_q(708, 1000);
  light->MaxCLL = 20000;
  CHECK_FALSE(hdr_metadata(*f));
}
