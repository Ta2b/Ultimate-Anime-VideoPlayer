#include "media/picture.h"

#include <cstring>

extern "C" {
#include <libswscale/swscale.h>
}

#include "media/av_util.h"

namespace ajp::media {

std::expected<Picture, Refusal> decode_picture(std::string_view bytes) {
  const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
  AVCodecID id = AV_CODEC_ID_NONE;
  if (bytes.size() > 3 && data[0] == 0xFF && data[1] == 0xD8) id = AV_CODEC_ID_MJPEG;
  if (bytes.size() > 8 && std::memcmp(data, "\x89PNG", 4) == 0) id = AV_CODEC_ID_PNG;
  if (id == AV_CODEC_ID_NONE) return refuse(RefusalCode::FormatUnsupported, "image : ni JPEG ni PNG");
  const AVCodec* codec = avcodec_find_decoder(id);
  if (codec == nullptr) return refuse(RefusalCode::FormatUnsupported, "image : décodeur absent de FFmpeg");
  CodecContextPtr context(avcodec_alloc_context3(codec));
  if (!context) return refuse(RefusalCode::DecodeFailed, "image : contexte du décodeur");
  if (const int error = avcodec_open2(context.get(), codec, nullptr); error < 0)
    return refuse_av(RefusalCode::DecodeFailed, "avcodec_open2 (image)", error);
  PacketPtr packet(av_packet_alloc());
  FramePtr frame(av_frame_alloc());
  if (!packet || !frame || av_new_packet(packet.get(), static_cast<int>(bytes.size())) < 0)
    return refuse(RefusalCode::DecodeFailed, "image : mémoire");
  std::memcpy(packet->data, bytes.data(), bytes.size());
  if (const int error = avcodec_send_packet(context.get(), packet.get()); error < 0)
    return refuse_av(RefusalCode::DecodeFailed, "avcodec_send_packet (image)", error);
  avcodec_send_packet(context.get(), nullptr);
  if (const int error = avcodec_receive_frame(context.get(), frame.get()); error < 0)
    return refuse_av(RefusalCode::DecodeFailed, "avcodec_receive_frame (image)", error);
  Picture out{.width = frame->width, .height = frame->height};
  out.rgba.resize(static_cast<std::size_t>(out.width) * static_cast<std::size_t>(out.height) * 4);
  // Les formats « yuvj » (JPEG) sont du YUV pleine plage : swscale les veut en YUV avec la plage dite à part.
  auto format = static_cast<AVPixelFormat>(frame->format);
  bool full_range = frame->color_range == AVCOL_RANGE_JPEG;
  if (format == AV_PIX_FMT_YUVJ420P || format == AV_PIX_FMT_YUVJ422P || format == AV_PIX_FMT_YUVJ444P) {
    format = format == AV_PIX_FMT_YUVJ420P   ? AV_PIX_FMT_YUV420P
             : format == AV_PIX_FMT_YUVJ422P ? AV_PIX_FMT_YUV422P
                                             : AV_PIX_FMT_YUV444P;
    full_range = true;
  }
  SwsContext* sws =
      sws_getContext(frame->width, frame->height, format, frame->width, frame->height, AV_PIX_FMT_RGBA64LE,
                     SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr);
  if (sws == nullptr) return refuse(RefusalCode::FormatUnsupported, "image : conversion en RGBA 16 bits");
  // La matrice du JPEG : BT.601, pleine plage (data/calibration/SOURCES.md).
  const int* bt601 = sws_getCoefficients(SWS_CS_ITU601);
  sws_setColorspaceDetails(sws, bt601, full_range ? 1 : 0, bt601, 1, 0, 1 << 16, 1 << 16);
  uint8_t* planes[1]{reinterpret_cast<uint8_t*>(out.rgba.data())};
  const int strides[1]{out.width * 8};
  const int rows = sws_scale(sws, frame->data, frame->linesize, 0, frame->height, planes, strides);
  sws_freeContext(sws);
  if (rows != out.height) return refuse(RefusalCode::DecodeFailed, "image : conversion incomplète");
  return out;
}

}  // namespace ajp::media
