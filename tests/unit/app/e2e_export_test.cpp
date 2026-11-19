// L'export de bout en bout sous Vulkan logiciel : 2 s exportées par le vrai lecteur (FFV1 sans perte, grade
// HDR anime en HDR10), puis la teinte de chaque plan du fichier, décodé selon ses étiquettes, comparée à celle
// de la source. Les passes du grade, posées par libplacebo après l'encodage YUV de la cible, gradaient Y, Cb et
// Cr comme R, V et B : un fond gris-bleu sortait brun-orangé, des cheveux lavande magenta (novembre 2026).
// Caché ([.e2e]) comme les autres cas du lecteur.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
}

#include "clip_writer.h"
#include "e2e_util.h"
#include "prepare/utf8.h"

namespace {

using namespace kirari::e2e;  // NOLINT(google-build-using-namespace)

// Les codes Y, Cb et Cr moyens d'un carré de 8 × 8 au centre d'une image, à leur profondeur, et ses étiquettes.
struct Sample {
  std::array<double, 3> ycbcr{};
  int depth = 8;
  AVColorSpace matrix = AVCOL_SPC_UNSPECIFIED;
  AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;
  AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
};

Sample center_of(const AVFrame& frame) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
  Sample out{.depth = desc->comp[0].depth,
             .matrix = frame.colorspace,
             .transfer = frame.color_trc,
             .primaries = frame.color_primaries};
  const int bytes = out.depth > 8 ? 2 : 1;
  const auto code = [&](int plane, int x, int y) -> int {
    const std::uint8_t* row = frame.data[plane] + static_cast<std::ptrdiff_t>(y) * frame.linesize[plane];
    if (bytes == 1) return row[x];
    std::uint16_t value = 0;
    std::memcpy(&value, row + static_cast<std::ptrdiff_t>(x) * 2, 2);
    return value;
  };
  constexpr int kSide = 8;
  const int x0 = frame.width / 2 - kSide / 2;
  const int y0 = frame.height / 2 - kSide / 2;
  for (int y = y0; y < y0 + kSide; ++y) {
    for (int x = x0; x < x0 + kSide; ++x) {
      out.ycbcr[0] += code(0, x, y);
      out.ycbcr[1] += code(1, x / 2, y / 2);
      out.ycbcr[2] += code(2, x / 2, y / 2);
    }
  }
  for (double& v : out.ycbcr) v /= kSide * kSide;
  return out;
}

// L'image n de la vidéo (décodée dans l'ordre), son centre ; rien si elle manque. Formats planaires 4:2:0.
std::optional<Sample> sample_at(const fs::path& path, int wanted) {
  AVFormatContext* format = nullptr;
  if (avformat_open_input(&format, kirari::prepare::utf8_path(path).c_str(), nullptr, nullptr) < 0) return {};
  const std::unique_ptr<AVFormatContext*, void (*)(AVFormatContext**)> format_owner(&format,
                                                                                     avformat_close_input);
  if (avformat_find_stream_info(format, nullptr) < 0) return {};
  const int index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (index < 0) return {};
  const AVCodec* codec = avcodec_find_decoder(format->streams[index]->codecpar->codec_id);
  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  const std::unique_ptr<AVCodecContext*, void (*)(AVCodecContext**)> ctx_owner(&ctx, avcodec_free_context);
  if (ctx == nullptr || avcodec_parameters_to_context(ctx, format->streams[index]->codecpar) < 0 ||
      avcodec_open2(ctx, codec, nullptr) < 0)
    return {};
  AVPacket* packet = av_packet_alloc();
  const std::unique_ptr<AVPacket*, void (*)(AVPacket**)> packet_owner(&packet, av_packet_free);
  AVFrame* frame = av_frame_alloc();
  const std::unique_ptr<AVFrame*, void (*)(AVFrame**)> frame_owner(&frame, av_frame_free);
  int seen = 0;
  bool draining = false;
  while (true) {
    if (!draining) {
      if (av_read_frame(format, packet) < 0) {
        draining = true;
        avcodec_send_packet(ctx, nullptr);
      } else {
        if (packet->stream_index == index) avcodec_send_packet(ctx, packet);
        av_packet_unref(packet);
      }
    }
    while (avcodec_receive_frame(ctx, frame) == 0) {
      if (seen++ == wanted) return center_of(*frame);
    }
    if (draining) return {};
  }
}

double pq_to_linear(double e) {  // ST 2084, en fraction de 10 000 nits
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

// La lumière linéaire en primaires BT.709 d'un échantillon : BT.709 (BT.1886, exposant 2,4) ou BT.2020 PQ,
// niveaux limités, comme le disent ses étiquettes.
std::array<double, 3> linear_709(const Sample& s) {
  const double scale = std::pow(2.0, s.depth - 8);
  const double y = (s.ycbcr[0] / scale - 16.0) / 219.0;
  const double cb = (s.ycbcr[1] / scale - 128.0) / 224.0;
  const double cr = (s.ycbcr[2] / scale - 128.0) / 224.0;
  const bool bt2020 = s.matrix == AVCOL_SPC_BT2020_NCL;
  const double kr = bt2020 ? 0.2627 : 0.2126;
  const double kb = bt2020 ? 0.0593 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const double r = y + 2.0 * (1.0 - kr) * cr;
  const double b = y + 2.0 * (1.0 - kb) * cb;
  const double g = (y - kr * r - kb * b) / kg;
  std::array<double, 3> rgb{r, g, b};
  for (double& c : rgb) {
    c = s.transfer == AVCOL_TRC_SMPTE2084 ? pq_to_linear(c) : std::pow(std::clamp(c, 0.0, 1.0), 2.4);
  }
  if (s.primaries != AVCOL_PRI_BT2020) return rgb;
  // BT.2020 vers BT.709 (ITU-R BT.2087), en lumière linéaire.
  return {1.6605 * rgb[0] - 0.5876 * rgb[1] - 0.0728 * rgb[2], -0.1246 * rgb[0] + 1.1329 * rgb[1] - 0.0083 * rgb[2],
          -0.0182 * rgb[0] - 0.1006 * rgb[1] + 1.1187 * rgb[2]};
}

// La teinte d'une couleur linéaire, en degrés (0 rouge, 120 vert, 240 bleu), et l'écart entre deux teintes.
double hue_of(const std::array<double, 3>& rgb) {
  const double angle = std::atan2(std::sqrt(3.0) * (rgb[1] - rgb[2]), 2.0 * rgb[0] - rgb[1] - rgb[2]);
  return angle * 180.0 / std::numbers::pi;
}

double hue_gap(double a, double b) {
  const double d = std::fmod(std::abs(a - b), 360.0);
  return std::min(d, 360.0 - d);
}

}  // namespace

TEST_CASE("Bout en bout : l'export HDR10 gradé garde la teinte de la source", "[.e2e][app]") {
  const fs::path dir = fresh_dir("kirari-e2e-export");
  // Deux plans d'une seconde, à luma moyenne : le premier vert pâle (Cb et Cr sous 128), le second rosé (Cb à
  // 135, Cr à 148). Assez saturés pour qu'une teinte fausse se voie, assez sombres pour que le grade ne les
  // blanchisse pas.
  kirari::media::test::ClipSpec spec{.name = "e2e-export.mkv",
                                     .width = 320,
                                     .height = 180,
                                     .frames = 50,
                                     .shot_frames = 25,
                                     .shot_lumas = {110, 110}};
  spec.color = {.primaries = AVCOL_PRI_BT709,
                .transfer = AVCOL_TRC_BT709,
                .matrix = AVCOL_SPC_BT709,
                .range = AVCOL_RANGE_MPEG};
  const auto clip = kirari::media::test::write_clip(spec);
  REQUIRE(clip);
  const fs::path source = alone(*clip, dir);
  const fs::path exported = dir / "export.mkv";
  const Run run = run_player({"play", kirari::prepare::utf8_path(source), "--software-gpu", "--mute",
                              "--no-resume", "--window=640x360", "--seconds=2", "--no-upscale", "--rife=0",
                              "--data-dir=" + kirari::prepare::utf8_path(dir / "reglages"),
                              "--keys=0.5:export-lossless=" + kirari::prepare::utf8_path(exported)});
  INFO(run.output);
  REQUIRE(run.code == 0);
  REQUIRE(fs::exists(exported));
  // Le grade est bien passé : sans lui, l'export garde la couleur de la source et le cas ne prouve rien.
  REQUIRE(run.output.find("grade HDR anime, sortie HDR10") != std::string::npos);

  for (const int frame : {12, 37}) {
    const auto in = sample_at(source, frame);
    const auto out = sample_at(exported, frame);
    REQUIRE(in);
    REQUIRE(out);
    INFO("image " << frame << " : source Y/Cb/Cr " << in->ycbcr[0] << "/" << in->ycbcr[1] << "/" << in->ycbcr[2]
                  << ", export " << out->ycbcr[0] << "/" << out->ycbcr[1] << "/" << out->ycbcr[2]);
    REQUIRE(out->depth == 10);
    REQUIRE(out->matrix == AVCOL_SPC_BT2020_NCL);
    REQUIRE(out->transfer == AVCOL_TRC_SMPTE2084);
    REQUIRE(out->primaries == AVCOL_PRI_BT2020);
    // La chroma garde son sens : un Cb ou un Cr sous le neutre dans la source le reste dans l'export.
    CHECK((in->ycbcr[1] < 128.0) == (out->ycbcr[1] < 512.0));
    CHECK((in->ycbcr[2] < 128.0) == (out->ycbcr[2] < 512.0));
    // Et la teinte, en lumière linéaire BT.709 : le grade sature, il ne tourne pas les couleurs.
    const double hue_in = hue_of(linear_709(*in));
    const double hue_out = hue_of(linear_709(*out));
    INFO("teinte source " << hue_in << "°, export " << hue_out << "°");
    CHECK(hue_gap(hue_in, hue_out) < 20.0);
  }
}
