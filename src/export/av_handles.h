// Outils FFmpeg internes au module : propriété RAII, refus nommés depuis un code AVERROR, couleurs
// de core/frame.h vers celles de FFmpeg. Seuls les .cpp de export l'incluent.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include "core/frame.h"
#include "core/types.h"
#include "export/encoder_choice.h"

namespace ajp::exporter {

struct InputCloser {
  void operator()(AVFormatContext* ctx) const { avformat_close_input(&ctx); }
};
struct CodecContextFree {
  void operator()(AVCodecContext* ctx) const { avcodec_free_context(&ctx); }
};
struct FrameFree {
  void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};
struct PacketFree {
  void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};
struct BufferUnref {
  void operator()(AVBufferRef* ref) const { av_buffer_unref(&ref); }
};

using InputPtr = std::unique_ptr<AVFormatContext, InputCloser>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextFree>;
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;
using PacketPtr = std::unique_ptr<AVPacket, PacketFree>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferUnref>;

inline constexpr AVRational kMicroseconds{1, 1'000'000};

// FFmpeg attend des chemins en UTF-8, Windows compris : path::string() y passerait par la page de
// code ANSI, ou lèverait une exception devant un nom en japonais.
std::string utf8(const std::filesystem::path& path);

std::string av_error_text(int error);
std::unexpected<Refusal> refuse(RefusalCode code, std::string detail);
std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error);

// Le format de pixels FFmpeg d'une disposition.
AVPixelFormat pixel_format(Layout layout);

// La couleur d'une image ou d'un flux encodés, telle que FFmpeg l'étiquette.
struct AvColor {
  AVColorPrimaries primaries = AVCOL_PRI_BT709;
  AVColorTransferCharacteristic transfer = AVCOL_TRC_BT709;
  AVColorSpace matrix = AVCOL_SPC_BT709;
  AVColorRange range = AVCOL_RANGE_MPEG;
  AVChromaLocation siting = AVCHROMA_LOC_LEFT;
};
AvColor av_color(const ColorInfo& color);
// La couleur d'un flux vidéo de la source, en niveaux vidéo : primaires, transfert, matrice et siting
// de ses étiquettes, ceux qui manquent devinés comme mpv (BT.601 en SD, BT.709 au-delà). Sans HDR10.
ColorInfo color_of(const AVCodecParameters& video);

}  // namespace ajp::exporter
