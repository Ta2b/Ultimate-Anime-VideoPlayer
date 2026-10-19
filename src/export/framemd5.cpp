#include "export/framemd5.h"

#include <cstring>
#include <format>
#include <optional>
#include <utility>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
}

#include "export/av_util.h"

namespace ajp::exporter {
namespace {

// Le muxer framemd5 de libavformat, écrit dans un tampon mémoire.
class Md5Muxer {
 public:
  static std::expected<Md5Muxer, Refusal> create(const AVCodecParameters& par, AVRational time_base) {
    AVFormatContext* raw = nullptr;
    if (const int error = avformat_alloc_output_context2(&raw, nullptr, "framemd5", nullptr); error < 0) {
      return refuse_av(RefusalCode::MuxFailed, "avformat_alloc_output_context2 framemd5", error);
    }
    Md5Muxer muxer;
    muxer.ctx_.reset(raw);
    AVStream* stream = avformat_new_stream(raw, nullptr);
    if (stream == nullptr) return refuse(RefusalCode::MuxFailed, "avformat_new_stream framemd5");
    if (const int error = avcodec_parameters_copy(stream->codecpar, &par); error < 0) {
      return refuse_av(RefusalCode::MuxFailed, "avcodec_parameters_copy framemd5", error);
    }
    stream->time_base = time_base;
    if (const int error = avio_open_dyn_buf(&raw->pb); error < 0) {
      return refuse_av(RefusalCode::MuxFailed, "avio_open_dyn_buf framemd5", error);
    }
    if (const int error = avformat_write_header(raw, nullptr); error < 0) {
      return refuse_av(RefusalCode::MuxFailed, "avformat_write_header framemd5", error);
    }
    muxer.time_base_ = time_base;
    return muxer;
  }

  std::expected<void, Refusal> write(AVPacket& packet) {
    packet.stream_index = 0;
    av_packet_rescale_ts(&packet, time_base_, ctx_->streams[0]->time_base);
    // av_write_frame ne prend pas la référence du paquet : elle se rend ici.
    const int error = av_write_frame(ctx_.get(), &packet);
    av_packet_unref(&packet);
    if (error < 0) return refuse_av(RefusalCode::MuxFailed, "av_write_frame framemd5", error);
    return {};
  }

  std::expected<std::string, Refusal> finish() {
    if (const int error = av_write_trailer(ctx_.get()); error < 0) {
      return refuse_av(RefusalCode::MuxFailed, "av_write_trailer framemd5", error);
    }
    std::uint8_t* buffer = nullptr;
    const int size = avio_close_dyn_buf(ctx_->pb, &buffer);
    ctx_->pb = nullptr;
    std::string text(reinterpret_cast<const char*>(buffer), static_cast<std::size_t>(size));
    av_free(buffer);
    return text;
  }

 private:
  struct Closer {
    void operator()(AVFormatContext* ctx) const {
      if (ctx->pb != nullptr) {
        std::uint8_t* buffer = nullptr;
        avio_close_dyn_buf(ctx->pb, &buffer);
        av_free(buffer);
      }
      avformat_free_context(ctx);
    }
  };
  std::unique_ptr<AVFormatContext, Closer> ctx_;
  AVRational time_base_{1, 1};
};

AVCodecParameters raw_video(int width, int height, AVPixelFormat format) {
  AVCodecParameters par{};
  par.codec_type = AVMEDIA_TYPE_VIDEO;
  par.codec_id = AV_CODEC_ID_RAWVIDEO;
  par.width = width;
  par.height = height;
  par.format = format;
  par.sample_aspect_ratio = AVRational{0, 1};
  return par;
}

// Une image décodée, plans tassés, hachée par le muxer ; il est créé à la première image,
// quand le format de sortie du décodeur est connu.
std::expected<void, Refusal> hash_frame(const AVFrame& frame, AVRational time_base,
                                        std::optional<Md5Muxer>& muxer, AVPacket& packet) {
  const auto format = static_cast<AVPixelFormat>(frame.format);
  if (!muxer) {
    auto created = Md5Muxer::create(raw_video(frame.width, frame.height, format), time_base);
    if (!created) return std::unexpected(created.error());
    muxer.emplace(std::move(*created));
  }
  const int size = av_image_get_buffer_size(format, frame.width, frame.height, 1);
  if (size < 0) return refuse_av(RefusalCode::DecodeFailed, "av_image_get_buffer_size", size);
  if (const int error = av_new_packet(&packet, size); error < 0) {
    return refuse_av(RefusalCode::DecodeFailed, "av_new_packet", error);
  }
  const int copied = av_image_copy_to_buffer(packet.data, size, frame.data, frame.linesize, format,
                                             frame.width, frame.height, 1);
  if (copied < 0) return refuse_av(RefusalCode::DecodeFailed, "av_image_copy_to_buffer", copied);
  packet.pts = frame.pts != AV_NOPTS_VALUE ? frame.pts : frame.best_effort_timestamp;
  packet.dts = packet.pts;
  packet.duration = frame.duration;
  packet.flags |= AV_PKT_FLAG_KEY;
  return muxer->write(packet);
}

std::expected<void, Refusal> drain(AVCodecContext& decoder, AVFrame& frame, AVRational time_base,
                                   std::optional<Md5Muxer>& muxer, AVPacket& packet) {
  while (true) {
    const int error = avcodec_receive_frame(&decoder, &frame);
    if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) return {};
    if (error < 0) return refuse_av(RefusalCode::DecodeFailed, "avcodec_receive_frame", error);
    auto hashed = hash_frame(frame, time_base, muxer, packet);
    av_frame_unref(&frame);
    if (!hashed) return hashed;
  }
}

}  // namespace

std::expected<std::string, Refusal> framemd5_of_file(const std::filesystem::path& path) {
  auto input = open_input(path.string());
  if (!input) return std::unexpected(input.error());
  const AVCodec* codec = nullptr;
  const int index = av_find_best_stream(input->get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (index < 0)
    return refuse_av(RefusalCode::DecoderUnavailable, std::format("av_find_best_stream {}", path.string()),
                     index);
  const AVStream& stream = *(*input)->streams[index];
  CodecContextPtr decoder(avcodec_alloc_context3(codec));
  FramePtr frame(av_frame_alloc());
  PacketPtr packet(av_packet_alloc());
  PacketPtr hashed(av_packet_alloc());
  if (!decoder || !frame || !packet || !hashed)
    return refuse(RefusalCode::DecodeFailed, "framemd5 : mémoire insuffisante");
  if (const int error = avcodec_parameters_to_context(decoder.get(), stream.codecpar); error < 0) {
    return refuse_av(RefusalCode::DecoderUnavailable, "avcodec_parameters_to_context", error);
  }
  decoder->pkt_timebase = stream.time_base;
  if (const int error = avcodec_open2(decoder.get(), codec, nullptr); error < 0) {
    return refuse_av(RefusalCode::DecoderUnavailable, std::format("avcodec_open2 {}", codec->name), error);
  }

  std::optional<Md5Muxer> muxer;
  while (true) {
    const int error = av_read_frame(input->get(), packet.get());
    if (error == AVERROR_EOF) break;
    if (error < 0) return refuse_av(RefusalCode::MediaUnreadable, "av_read_frame", error);
    if (packet->stream_index != index) {
      av_packet_unref(packet.get());
      continue;
    }
    const int sent = avcodec_send_packet(decoder.get(), packet.get());
    av_packet_unref(packet.get());
    if (sent < 0) return refuse_av(RefusalCode::DecodeFailed, "avcodec_send_packet", sent);
    if (auto ok = drain(*decoder, *frame, stream.time_base, muxer, *hashed); !ok)
      return std::unexpected(ok.error());
  }
  if (const int sent = avcodec_send_packet(decoder.get(), nullptr); sent < 0 && sent != AVERROR_EOF) {
    return refuse_av(RefusalCode::DecodeFailed, "avcodec_send_packet (fin)", sent);
  }
  if (auto ok = drain(*decoder, *frame, stream.time_base, muxer, *hashed); !ok)
    return std::unexpected(ok.error());
  if (!muxer)
    return refuse(RefusalCode::DecodeFailed, std::format("{} : aucune image décodée", path.string()));
  return muxer->finish();
}

struct FrameMd5::State {
  Md5Muxer muxer;
  int width = 0;
  int height = 0;
  PacketPtr packet{av_packet_alloc()};
};

FrameMd5::FrameMd5(std::unique_ptr<State> state) : state_(std::move(state)) {}
FrameMd5::FrameMd5(FrameMd5&&) noexcept = default;
FrameMd5& FrameMd5::operator=(FrameMd5&&) noexcept = default;
FrameMd5::~FrameMd5() = default;

std::expected<FrameMd5, Refusal> FrameMd5::create(int width, int height) {
  if (width <= 0 || height <= 0) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("framemd5 : taille {}x{}", width, height));
  }
  auto muxer = Md5Muxer::create(raw_video(width, height, AV_PIX_FMT_RGBA64), kMicroseconds);
  if (!muxer) return std::unexpected(muxer.error());
  auto state = std::make_unique<State>(State{.muxer = std::move(*muxer), .width = width, .height = height});
  if (!state->packet) return refuse(RefusalCode::MuxFailed, "av_packet_alloc : mémoire insuffisante");
  return FrameMd5(std::move(state));
}

std::expected<void, Refusal> FrameMd5::add(const HostFrame& frame) {
  const auto row = static_cast<std::size_t>(frame.width) * 8;
  const auto rows = static_cast<std::size_t>(frame.height);
  if (frame.width != state_->width || frame.height != state_->height || frame.stride < row ||
      frame.rgba.size_bytes() < (rows - 1) * frame.stride + row || frame.pts_us == kNoPts) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("framemd5 : image {}x{} (stride {}), attendue {}x{}, pts connu", frame.width,
                              frame.height, frame.stride, state_->width, state_->height));
  }
  AVPacket& packet = *state_->packet;
  if (const int error = av_new_packet(&packet, static_cast<int>(row * rows)); error < 0) {
    return refuse_av(RefusalCode::MuxFailed, "av_new_packet framemd5", error);
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(frame.rgba.data());
  for (std::size_t y = 0; y < rows; ++y) std::memcpy(packet.data + y * row, bytes + y * frame.stride, row);
  packet.pts = frame.pts_us;
  packet.dts = frame.pts_us;
  packet.flags |= AV_PKT_FLAG_KEY;
  return state_->muxer.write(packet);
}

std::expected<std::string, Refusal> FrameMd5::finish() {
  return state_->muxer.finish();
}

}  // namespace ajp::exporter
