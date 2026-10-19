// Outils FFmpeg internes à l'export : propriété RAII et refus nommés depuis un code AVERROR.
// Seuls les .cpp de export l'incluent ; les en-têtes publics restent sans FFmpeg.
#pragma once

#include <expected>
#include <memory>
#include <string>
#include <string_view>

extern "C" {
#include <libswscale/swscale.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

#include "core/types.h"

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
struct SwsFree {
  void operator()(SwsContext* ctx) const { sws_freeContext(ctx); }
};

using InputPtr = std::unique_ptr<AVFormatContext, InputCloser>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextFree>;
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;
using PacketPtr = std::unique_ptr<AVPacket, PacketFree>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferUnref>;
using SwsPtr = std::unique_ptr<SwsContext, SwsFree>;

inline constexpr AVRational kMicroseconds{1, 1'000'000};

// Texte de av_strerror pour un code AVERROR.
std::string av_error_text(int error);

// Refus nommé : « où : texte FFmpeg ».
std::unexpected<Refusal> refuse(RefusalCode code, std::string detail);
std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error);

// Ouvre un fichier en lecture et lit ses flux ; refus MediaUnreadable qui cite l'appel.
std::expected<InputPtr, Refusal> open_input(const std::string& path);

}  // namespace ajp::exporter
