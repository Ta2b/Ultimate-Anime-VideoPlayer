#include "media/bench_clip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <vector>

#include "media/av_util.h"

namespace ajp::media {
namespace {

struct Rgb {
  std::uint8_t r, g, b;
};

// Une teinte de la scène numéro `scene` : couleurs plates, comme un aplat d'animation.
Rgb palette(int scene, int index) {
  constexpr Rgb kColors[]{{232, 92, 120},  {80, 150, 230},  {250, 210, 90}, {96, 200, 140},
                          {150, 100, 220}, {245, 245, 240}, {30, 34, 52},   {255, 140, 60}};
  return kColors[static_cast<std::size_t>((scene * 3 + index) % 8)];
}

void put(AVFrame& frame, int x, int y, Rgb c) {
  // BT.709 limité, YUV 4:2:0 : Y partout, U et V au pixel pair.
  const float r = c.r;
  const float g = c.g;
  const float b = c.b;
  const auto luma =
      static_cast<std::uint8_t>(std::clamp(16.0F + 0.1826F * r + 0.6142F * g + 0.0620F * b, 16.0F, 235.0F));
  frame.data[0][y * frame.linesize[0] + x] = luma;
  if (((x | y) & 1) == 0) {
    frame.data[1][(y / 2) * frame.linesize[1] + x / 2] = static_cast<std::uint8_t>(
        std::clamp(128.0F - 0.1006F * r - 0.3386F * g + 0.4392F * b, 16.0F, 240.0F));
    frame.data[2][(y / 2) * frame.linesize[2] + x / 2] = static_cast<std::uint8_t>(
        std::clamp(128.0F + 0.4392F * r - 0.3989F * g - 0.0403F * b, 16.0F, 240.0F));
  }
}

void draw_frame(AVFrame& frame, int width, int height, int index, int fps) {
  const int scene = index / (fps * 2);  // une coupe toutes les deux secondes
  const int t = index % (fps * 2);      // l'instant dans la scène (sur deux secondes)
  const int pan = (index * 12) % 4096;  // le fond défile : 12 px par image (un travelling vif)
  const int band = std::max(1, height / 9);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Fond : bandes verticales plates qui défilent, avec un dégradé léger vertical.
      const int column = ((x + pan) / (width / 6)) % 4;
      Rgb color = palette(scene, column);
      const float shade = 0.82F + 0.18F * static_cast<float>(y) / static_cast<float>(height);
      color = {static_cast<std::uint8_t>(color.r * shade), static_cast<std::uint8_t>(color.g * shade),
               static_cast<std::uint8_t>(color.b * shade)};
      // Hachures fines (cheveux, herbe) : de la haute fréquence au contour franc.
      if (((x + y * 2 + pan / 3) % 14) < 2 && (y / band) % 2 == 0) color = palette(scene, 5);
      put(frame, x, y, color);
    }
  }
  // Deux personnages : un aplat arrondi à contour sombre, qui se croisent à des vitesses différentes.
  for (int who = 0; who < 2; ++who) {
    const float phase = static_cast<float>(t) / static_cast<float>(fps * 2);
    const float cx = static_cast<float>(width) * (who == 0 ? phase : 1.0F - phase);
    const float cy = static_cast<float>(height) * (who == 0 ? 0.45F : 0.62F) +
                     60.0F * std::sin(6.2832F * (phase * 2.0F + static_cast<float>(who)));
    const float rx = static_cast<float>(width) * 0.09F;
    const float ry = static_cast<float>(height) * 0.22F;
    const int x0 = std::max(0, static_cast<int>(cx - rx - 6));
    const int x1 = std::min(width - 1, static_cast<int>(cx + rx + 6));
    const int y0 = std::max(0, static_cast<int>(cy - ry - 6));
    const int y1 = std::min(height - 1, static_cast<int>(cy + ry + 6));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float dx = (static_cast<float>(x) - cx) / rx;
        const float dy = (static_cast<float>(y) - cy) / ry;
        const float d = dx * dx + dy * dy;
        if (d <= 1.0F)
          put(frame, x, y, palette(scene + who + 1, 1 + who));
        else if (d <= 1.12F)
          put(frame, x, y, {20, 20, 28});
      }
    }
  }
}

}  // namespace

std::expected<void, Refusal> write_bench_clip(const std::filesystem::path& file, const BenchClipSpec& spec) {
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
  if (codec == nullptr) return refuse(RefusalCode::LibraryUnavailable, "encodeur MPEG-4 absent de FFmpeg");
  AVFormatContext* raw = nullptr;
  const std::string path = file.string();
  if (const int err = avformat_alloc_output_context2(&raw, nullptr, "matroska", path.c_str());
      err < 0 || raw == nullptr)
    return refuse_av(RefusalCode::LibraryUnavailable, "conteneur matroska", err);
  struct OutputFree {
    void operator()(AVFormatContext* c) const {
      if (c != nullptr && !(c->oformat->flags & AVFMT_NOFILE) && c->pb != nullptr) avio_closep(&c->pb);
      avformat_free_context(c);
    }
  };
  std::unique_ptr<AVFormatContext, OutputFree> out(raw);
  CodecContextPtr encoder(avcodec_alloc_context3(codec));
  if (!encoder) return refuse(RefusalCode::LibraryUnavailable, "contexte d'encodeur");
  encoder->width = spec.width;
  encoder->height = spec.height;
  encoder->time_base = AVRational{1, spec.fps};
  encoder->framerate = AVRational{spec.fps, 1};
  encoder->pix_fmt = AV_PIX_FMT_YUV420P;
  encoder->bit_rate = 20'000'000;
  encoder->gop_size = spec.fps;
  encoder->max_b_frames = 0;
  encoder->colorspace = AVCOL_SPC_BT709;
  encoder->color_primaries = AVCOL_PRI_BT709;
  encoder->color_trc = AVCOL_TRC_BT709;
  encoder->color_range = AVCOL_RANGE_MPEG;
  if (out->oformat->flags & AVFMT_GLOBALHEADER) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  if (const int err = avcodec_open2(encoder.get(), codec, nullptr); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "avcodec_open2", err);
  AVStream* stream = avformat_new_stream(out.get(), nullptr);
  if (stream == nullptr) return refuse(RefusalCode::LibraryUnavailable, "flux de sortie");
  stream->time_base = encoder->time_base;
  stream->avg_frame_rate = encoder->framerate;
  if (const int err = avcodec_parameters_from_context(stream->codecpar, encoder.get()); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "paramètres du flux", err);
  if (!(out->oformat->flags & AVFMT_NOFILE)) {
    if (const int err = avio_open(&out->pb, path.c_str(), AVIO_FLAG_WRITE); err < 0)
      return refuse_av(RefusalCode::FileUnwritable, path, err);
  }
  if (const int err = avformat_write_header(out.get(), nullptr); err < 0)
    return refuse_av(RefusalCode::FileUnwritable, "en-tête", err);

  FramePtr frame(av_frame_alloc());
  PacketPtr packet(av_packet_alloc());
  if (!frame || !packet) return refuse(RefusalCode::LibraryUnavailable, "mémoire FFmpeg");
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = spec.width;
  frame->height = spec.height;
  if (const int err = av_frame_get_buffer(frame.get(), 0); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "image", err);
  const auto drain = [&]() -> std::expected<void, Refusal> {
    while (true) {
      const int got = avcodec_receive_packet(encoder.get(), packet.get());
      if (got == AVERROR(EAGAIN) || got == AVERROR_EOF) return {};
      if (got < 0) return refuse_av(RefusalCode::DecodeFailed, "encodage", got);
      av_packet_rescale_ts(packet.get(), encoder->time_base, stream->time_base);
      packet->stream_index = stream->index;
      if (const int err = av_interleaved_write_frame(out.get(), packet.get()); err < 0)
        return refuse_av(RefusalCode::FileUnwritable, "écriture", err);
    }
  };
  const int total = spec.fps * spec.seconds;
  for (int i = 0; i < total; ++i) {
    if (const int err = av_frame_make_writable(frame.get()); err < 0)
      return refuse_av(RefusalCode::LibraryUnavailable, "image", err);
    draw_frame(*frame, spec.width, spec.height, i, spec.fps);
    frame->pts = i;
    if (const int err = avcodec_send_frame(encoder.get(), frame.get()); err < 0)
      return refuse_av(RefusalCode::DecodeFailed, "envoi", err);
    if (auto ok = drain(); !ok) return ok;
  }
  avcodec_send_frame(encoder.get(), nullptr);
  if (auto ok = drain(); !ok) return ok;
  if (const int err = av_write_trailer(out.get()); err < 0)
    return refuse_av(RefusalCode::FileUnwritable, "fin", err);
  return {};
}

}  // namespace ajp::media
