#include "support.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <system_error>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
}

namespace ajp::exporter::test {
namespace {

struct OutputCloser {
  void operator()(AVFormatContext* ctx) const {
    if (ctx->pb != nullptr) avio_closep(&ctx->pb);
    avformat_free_context(ctx);
  }
};

AVStream* new_stream(AVFormatContext& ctx, AVMediaType type, AVCodecID codec, AVRational time_base) {
  AVStream* stream = avformat_new_stream(&ctx, nullptr);
  REQUIRE(stream != nullptr);
  stream->codecpar->codec_type = type;
  stream->codecpar->codec_id = codec;
  stream->time_base = time_base;
  return stream;
}

void add_chapter(AVFormatContext& ctx, std::int64_t id, std::int64_t start_us, std::int64_t end_us,
                 const char* title) {
  void* grown = av_realloc_array(static_cast<void*>(ctx.chapters), ctx.nb_chapters + 1, sizeof(AVChapter*));
  REQUIRE(grown != nullptr);
  ctx.chapters = static_cast<AVChapter**>(grown);
  auto* chapter = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
  REQUIRE(chapter != nullptr);
  chapter->id = id;
  chapter->time_base = kMicroseconds;
  chapter->start = start_us;
  chapter->end = end_us;
  av_dict_set(&chapter->metadata, "title", title, 0);
  ctx.chapters[ctx.nb_chapters++] = chapter;
}

// Un paquet à écrire : instant en microsecondes, flux, pts et durée dans sa base, données.
struct Planned {
  std::int64_t at_us;
  int stream;
  std::int64_t pts;
  std::int64_t duration;
  std::vector<std::uint8_t> data;
};

}  // namespace

std::filesystem::path work_dir(std::string_view name) {
  const std::filesystem::path dir = std::filesystem::path(AJP_EXPORT_WORK_DIR) / name;
  std::error_code ignored;
  std::filesystem::remove_all(dir, ignored);
  std::filesystem::create_directories(dir);
  return dir;
}

std::int64_t frame_pts(int index, FrameRate rate) {
  return av_rescale(index, std::int64_t{1'000'000} * rate.den, rate.num);
}

HostFrame OwnedFrame::view() const {
  return HostFrame{.width = width,
                   .height = height,
                   .stride = static_cast<std::size_t>(width) * 8,
                   .rgba = pixels,
                   .pts_us = pts_us};
}

OwnedFrame pattern_frame(int width, int height, int index, std::int64_t pts_us) {
  OwnedFrame frame{.width = width, .height = height, .pts_us = pts_us};
  frame.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
  std::size_t i = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 4; ++c) {
        const auto value = static_cast<unsigned>(x * 977 + y * 131 + index * 4099 + c * 21011);
        frame.pixels[i++] =
            c == 3 ? std::uint16_t{65535} : static_cast<std::uint16_t>(value * 2654435761U >> 16);
      }
    }
  }
  return frame;
}

OwnedFrame flat_frame(int width, int height, std::uint16_t r, std::uint16_t g, std::uint16_t b) {
  OwnedFrame frame{.width = width, .height = height};
  frame.pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
  for (int p = 0; p < width * height; ++p) frame.pixels.insert(frame.pixels.end(), {r, g, b, 65535});
  return frame;
}

std::vector<OwnedFrame> pattern_frames(int width, int height, int count, FrameRate rate, int first_index) {
  std::vector<OwnedFrame> frames;
  for (int i = first_index; i < first_index + count; ++i) {
    frames.push_back(pattern_frame(width, height, i, frame_pts(i, rate)));
  }
  return frames;
}

std::expected<std::optional<SourceFrame>, Refusal> VectorSource::next() {
  if (before_next) before_next(index_);
  if (index_ >= frames_.size()) return std::nullopt;
  return SourceFrame{frames_[index_++].view()};
}

ExportJob ffv1_job(const std::filesystem::path& output, int width, int height, ExportFrameSource& source) {
  return ExportJob{.output = output,
                   .width = width,
                   .height = height,
                   .encoder = EncoderSettings{.encoder = "ffv1", .format = PixelFormat::Rgb16},
                   .source = &source};
}

std::vector<std::uint8_t> audio_payload(int packet) {
  std::vector<std::uint8_t> data(static_cast<std::size_t>(SourceLayout::kAudioSamples) * 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<std::uint8_t>((i * 7 + static_cast<std::size_t>(packet) * 13) & 0xFF);
  return data;
}

std::vector<std::uint8_t> font_payload() {
  std::vector<std::uint8_t> data(300);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i * 31 + 5);
  return data;
}

void write_source_media(const std::filesystem::path& path) {
  AVFormatContext* raw = nullptr;
  REQUIRE(avformat_alloc_output_context2(&raw, nullptr, "matroska", path.string().c_str()) >= 0);
  const std::unique_ptr<AVFormatContext, OutputCloser> ctx(raw);
  av_dict_set(&raw->metadata, "title", "Source synthétique", 0);

  AVStream* video = new_stream(*raw, AVMEDIA_TYPE_VIDEO, AV_CODEC_ID_RAWVIDEO, {1, 1000});
  video->codecpar->width = 16;
  video->codecpar->height = 16;
  video->codecpar->format = AV_PIX_FMT_GRAY8;
  AVStream* audio =
      new_stream(*raw, AVMEDIA_TYPE_AUDIO, AV_CODEC_ID_PCM_S16LE, {1, SourceLayout::kSampleRate});
  audio->codecpar->sample_rate = SourceLayout::kSampleRate;
  av_channel_layout_default(&audio->codecpar->ch_layout, 2);
  audio->codecpar->format = AV_SAMPLE_FMT_S16;
  audio->codecpar->bits_per_coded_sample = 16;
  audio->codecpar->block_align = 4;
  av_dict_set(&audio->metadata, "language", "jpn", 0);
  AVStream* subs = new_stream(*raw, AVMEDIA_TYPE_SUBTITLE, AV_CODEC_ID_SUBRIP, {1, 1000});
  av_dict_set(&subs->metadata, "language", "fre", 0);
  AVStream* font = new_stream(*raw, AVMEDIA_TYPE_ATTACHMENT, AV_CODEC_ID_TTF, {1, 1000});
  const auto font_data = font_payload();
  font->codecpar->extradata =
      static_cast<std::uint8_t*>(av_mallocz(font_data.size() + AV_INPUT_BUFFER_PADDING_SIZE));
  REQUIRE(font->codecpar->extradata != nullptr);
  std::memcpy(font->codecpar->extradata, font_data.data(), font_data.size());
  font->codecpar->extradata_size = static_cast<int>(font_data.size());
  av_dict_set(&font->metadata, "filename", SourceLayout::kFontName.data(), 0);
  av_dict_set(&font->metadata, "mimetype", "font/ttf", 0);
  add_chapter(*raw, 1, 0, SourceLayout::kChapterSplitUs, "Début");
  add_chapter(*raw, 2, SourceLayout::kChapterSplitUs, SourceLayout::kChapterEndUs, "Suite");

  REQUIRE(avio_open(&raw->pb, path.string().c_str(), AVIO_FLAG_WRITE) >= 0);
  // La vidéo brute n'a pas d'identifiant Matroska natif : le mode VFW la porte quand même.
  AVDictionary* options = nullptr;
  av_dict_set(&options, "allow_raw_vfw", "1", 0);
  const int written = avformat_write_header(raw, &options);
  av_dict_free(&options);
  REQUIRE(written >= 0);

  std::vector<Planned> plan;
  for (int i = 0; i < 25; ++i)
    plan.push_back({i * 40'000, 0, i * 40, 40, std::vector<std::uint8_t>(256, static_cast<std::uint8_t>(i))});
  for (int k = 0; k < SourceLayout::kAudioPackets; ++k) {
    const std::int64_t pts = std::int64_t{k} * SourceLayout::kAudioSamples;
    plan.push_back({av_rescale(pts, 1'000'000, SourceLayout::kSampleRate), 1, pts,
                    SourceLayout::kAudioSamples, audio_payload(k)});
  }
  for (const auto& [at, text] : {std::pair{SourceLayout::kSubtitle1Us, "Ligne une"},
                                 std::pair{SourceLayout::kSubtitle2Us, "Ligne deux"}}) {
    plan.push_back({at, 2, at / 1000, SourceLayout::kSubtitleDurationUs / 1000,
                    std::vector<std::uint8_t>(text, text + std::strlen(text))});
  }
  std::ranges::stable_sort(plan, {}, [](const Planned& p) { return std::tuple(p.at_us, p.stream); });
  AVPacket* packet = av_packet_alloc();
  REQUIRE(packet != nullptr);
  for (const Planned& p : plan) {
    REQUIRE(av_new_packet(packet, static_cast<int>(p.data.size())) >= 0);
    std::memcpy(packet->data, p.data.data(), p.data.size());
    packet->stream_index = p.stream;
    packet->pts = p.pts;
    packet->dts = p.pts;
    packet->duration = p.duration;
    packet->flags = AV_PKT_FLAG_KEY;
    av_packet_rescale_ts(packet,
                         p.stream == 1 ? AVRational{1, SourceLayout::kSampleRate} : AVRational{1, 1000},
                         raw->streams[p.stream]->time_base);
    REQUIRE(av_interleaved_write_frame(raw, packet) >= 0);
  }
  av_packet_free(&packet);
  REQUIRE(av_write_trailer(raw) >= 0);
}

std::vector<StreamDump> dump_streams(const std::filesystem::path& path) {
  auto input = open_input(path.string());
  REQUIRE(input.has_value());
  std::vector<StreamDump> dumps((*input)->nb_streams);
  for (unsigned i = 0; i < (*input)->nb_streams; ++i) {
    dumps[i].type = (*input)->streams[i]->codecpar->codec_type;
    dumps[i].codec = (*input)->streams[i]->codecpar->codec_id;
  }
  const PacketPtr packet(av_packet_alloc());
  while (av_read_frame(input->get(), packet.get()) >= 0) {
    StreamDump& dump = dumps[static_cast<std::size_t>(packet->stream_index)];
    const AVRational tb = (*input)->streams[packet->stream_index]->time_base;
    dump.pts_us.push_back(packet->pts == AV_NOPTS_VALUE ? kNoPts
                                                        : av_rescale_q(packet->pts, tb, kMicroseconds));
    dump.data.emplace_back(packet->data, packet->data + packet->size);
    av_packet_unref(packet.get());
  }
  return dumps;
}

void decode_video(const std::filesystem::path& path, const std::function<void(const AVFrame&)>& on_frame) {
  auto input = open_input(path.string());
  REQUIRE(input.has_value());
  const AVCodec* codec = nullptr;
  const int index = av_find_best_stream(input->get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  REQUIRE(index >= 0);
  const CodecContextPtr decoder(avcodec_alloc_context3(codec));
  REQUIRE(avcodec_parameters_to_context(decoder.get(), (*input)->streams[index]->codecpar) >= 0);
  REQUIRE(avcodec_open2(decoder.get(), codec, nullptr) >= 0);
  const PacketPtr packet(av_packet_alloc());
  const FramePtr frame(av_frame_alloc());
  auto drain = [&] {
    while (avcodec_receive_frame(decoder.get(), frame.get()) >= 0) {
      on_frame(*frame);
      av_frame_unref(frame.get());
    }
  };
  while (av_read_frame(input->get(), packet.get()) >= 0) {
    if (packet->stream_index == index) REQUIRE(avcodec_send_packet(decoder.get(), packet.get()) >= 0);
    av_packet_unref(packet.get());
    drain();
  }
  REQUIRE(avcodec_send_packet(decoder.get(), nullptr) >= 0);
  drain();
}

}  // namespace ajp::exporter::test
