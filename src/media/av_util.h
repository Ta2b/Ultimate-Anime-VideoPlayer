// Outils FFmpeg internes au module : propriété RAII, temps en microsecondes,
// refus nommés depuis un code AVERROR. Seuls les .cpp de media l'incluent.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include "core/types.h"
#include "media/track.h"

namespace ajp::media {

struct FormatCloser {
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

using FormatPtr = std::unique_ptr<AVFormatContext, FormatCloser>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextFree>;
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;
using PacketPtr = std::unique_ptr<AVPacket, PacketFree>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferUnref>;

inline constexpr AVRational kMicroseconds{1, 1'000'000};

// Texte de av_strerror pour un code AVERROR.
std::string av_error_text(int error);

// Refus nommé : « où : texte FFmpeg ».
std::unexpected<Refusal> refuse(RefusalCode code, std::string detail);
std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error);

// Horodatage d'un flux en microsecondes depuis le début du fichier ;
// kNoTimestamp reste kNoTimestamp.
std::int64_t to_us(std::int64_t ts, AVRational time_base, std::int64_t start_us);
// Inverse de to_us, pour viser une position dans un flux. Arrondi vers le bas :
// un seek AVSEEK_FLAG_BACKWARD ne doit jamais viser après la cible.
std::int64_t from_us(std::int64_t us, AVRational time_base, std::int64_t start_us);
// Durée d'un flux en microsecondes, 0 si inconnue ou négative.
std::int64_t duration_to_us(std::int64_t duration, AVRational time_base);

// Nouvelle référence sur un tampon, ou nullptr.
BufferPtr ref_buffer(const AVBufferRef* ref);

}  // namespace ajp::media
