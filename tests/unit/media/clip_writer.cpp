#include "clip_writer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <random>
#include <string_view>
#include <system_error>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
}

namespace ajp::media::test {
namespace {

constexpr int kSampleRate = 48000;
constexpr int kSamplesPerPacket = 960;   // 20 ms
constexpr int kAudioBytesPerSample = 4;  // s16 stéréo

std::unexpected<std::string> fail(std::string_view where, int error) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  av_strerror(error, text.data(), text.size());
  return std::unexpected(std::format("{} : {}", where, text.data()));
}

std::unexpected<std::string> fail(std::string_view where) {
  return std::unexpected(std::string(where));
}

struct Output {
  AVFormatContext* ctx = nullptr;
  AVCodecContext* encoder = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  Output() = default;
  Output(const Output&) = delete;
  Output& operator=(const Output&) = delete;
  ~Output() {
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&encoder);
    if (ctx != nullptr && ctx->pb != nullptr) avio_closep(&ctx->pb);
    avformat_free_context(ctx);
  }
};

void set_tags(AVStream& stream, const std::string& language, const std::string& title) {
  if (!language.empty()) av_dict_set(&stream.metadata, "language", language.c_str(), 0);
  if (!title.empty()) av_dict_set(&stream.metadata, "title", title.c_str(), 0);
}

bool set_extradata(AVCodecParameters& par, std::string_view bytes) {
  par.extradata = static_cast<std::uint8_t*>(av_mallocz(bytes.size() + AV_INPUT_BUFFER_PADDING_SIZE));
  if (par.extradata == nullptr) return false;
  std::copy(bytes.begin(), bytes.end(), par.extradata);
  par.extradata_size = static_cast<int>(bytes.size());
  return true;
}

void add_hdr10(AVCodecParameters& par) {
  auto* mastering_sd =
      av_packet_side_data_new(&par.coded_side_data, &par.nb_coded_side_data,
                              AV_PKT_DATA_MASTERING_DISPLAY_METADATA, sizeof(AVMasteringDisplayMetadata), 0);
  auto* mastering = reinterpret_cast<AVMasteringDisplayMetadata*>(mastering_sd->data);
  *mastering = AVMasteringDisplayMetadata{};
  const std::array<std::array<int, 2>, 3> primaries{{{708, 292}, {170, 797}, {131, 46}}};
  for (std::size_t i = 0; i < 3; ++i) {
    mastering->display_primaries[i][0] = av_make_q(primaries[i][0], 1000);
    mastering->display_primaries[i][1] = av_make_q(primaries[i][1], 1000);
  }
  mastering->white_point[0] = av_make_q(3127, 10000);
  mastering->white_point[1] = av_make_q(3290, 10000);
  mastering->max_luminance = av_make_q(1000, 1);
  mastering->min_luminance = av_make_q(1, 10000);
  mastering->has_primaries = 1;
  mastering->has_luminance = 1;

  auto* light_sd =
      av_packet_side_data_new(&par.coded_side_data, &par.nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
                              sizeof(AVContentLightMetadata), 0);
  auto* light = reinterpret_cast<AVContentLightMetadata*>(light_sd->data);
  light->MaxCLL = 1000;
  light->MaxFALL = 400;
}

void fill_frame(AVFrame& frame, const ClipSpec& spec, int n) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(spec.pixel_format);
  const int bits = desc->comp[0].depth;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? spec.width : AV_CEIL_RSHIFT(spec.width, desc->log2_chroma_w);
    const int h = plane == 0 ? spec.height : AV_CEIL_RSHIFT(spec.height, desc->log2_chroma_h);
    for (int y = 0; y < h; ++y) {
      std::uint8_t* row = frame.data[plane] + static_cast<std::ptrdiff_t>(y) * frame.linesize[plane];
      for (int x = 0; x < w; ++x) {
        const int value = spec.codec == VideoCodec::Mpeg4 ? (plane == 0 ? flat_luma(n) : 128)
                                                          : pattern(plane, x, y, n, bits);
        if (bits > 8) {
          row[2 * x] = static_cast<std::uint8_t>(value & 0xFF);
          row[2 * x + 1] = static_cast<std::uint8_t>(value >> 8);
        } else {
          row[x] = static_cast<std::uint8_t>(value);
        }
      }
    }
  }
}

class Writer {
 public:
  explicit Writer(const ClipSpec& spec) : spec_(spec) {}

  std::expected<void, std::string> write(const std::filesystem::path& path) {
    const std::string file = path.string();
    if (const int err =
            avformat_alloc_output_context2(&out_.ctx, nullptr, spec_.container.c_str(), file.c_str());
        err < 0) {
      return fail("conteneur", err);
    }
    if (auto ok = add_video(); !ok) return ok;
    for (const AudioSpec& audio : spec_.audio) add_audio(audio);
    for (const SubtitleSpec& subs : spec_.subtitles) add_subtitles(subs);
    for (const AttachmentSpec& attachment : spec_.attachments) add_attachment(attachment);
    add_chapters();
    if (const int err = avio_open(&out_.ctx->pb, file.c_str(), AVIO_FLAG_WRITE); err < 0)
      return fail("avio", err);
    if (const int err = avformat_write_header(out_.ctx, nullptr); err < 0) return fail("en-tête", err);
    if (auto ok = encode_video(); !ok) return ok;
    if (auto ok = write_audio(); !ok) return ok;
    if (auto ok = write_subtitles(); !ok) return ok;
    if (const int err = av_write_trailer(out_.ctx); err < 0) return fail("fin", err);
    return {};
  }

 private:
  std::expected<void, std::string> add_video() {
    const AVCodec* codec =
        avcodec_find_encoder(spec_.codec == VideoCodec::Mpeg4 ? AV_CODEC_ID_MPEG4 : AV_CODEC_ID_FFV1);
    out_.encoder = avcodec_alloc_context3(codec);
    AVCodecContext& enc = *out_.encoder;
    enc.width = spec_.width;
    enc.height = spec_.height;
    enc.pix_fmt = spec_.pixel_format;
    enc.time_base = av_inv_q(spec_.frame_rate);
    enc.framerate = spec_.frame_rate;
    enc.gop_size = spec_.gop;
    enc.max_b_frames = spec_.b_frames;
    enc.color_primaries = spec_.color.primaries;
    enc.color_trc = spec_.color.transfer;
    enc.colorspace = spec_.color.matrix;
    enc.color_range = spec_.color.range;
    enc.chroma_sample_location = spec_.color.siting;
    if (spec_.codec == VideoCodec::Mpeg4) {
      enc.flags |= AV_CODEC_FLAG_QSCALE;
      enc.global_quality = FF_QP2LAMBDA * 2;
    } else {
      enc.level = 3;
    }
    if ((out_.ctx->oformat->flags & AVFMT_GLOBALHEADER) != 0) enc.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    // Sans détection de changement de scène : les points clés tombent tous les `gop`.
    AVDictionary* options = nullptr;
    if (spec_.codec == VideoCodec::Mpeg4) av_dict_set(&options, "sc_threshold", "1000000000", 0);
    const int opened = avcodec_open2(&enc, codec, &options);
    av_dict_free(&options);
    if (opened < 0) return fail("encodeur vidéo", opened);

    video_ = avformat_new_stream(out_.ctx, nullptr);
    avcodec_parameters_from_context(video_->codecpar, &enc);
    video_->time_base = enc.time_base;
    video_->avg_frame_rate = spec_.frame_rate;
    video_->disposition = AV_DISPOSITION_DEFAULT;
    if (spec_.color.hdr10) add_hdr10(*video_->codecpar);
    return {};
  }

  void add_audio(const AudioSpec& audio) {
    AVStream* st = avformat_new_stream(out_.ctx, nullptr);
    AVCodecParameters& par = *st->codecpar;
    par.codec_type = AVMEDIA_TYPE_AUDIO;
    par.codec_id = AV_CODEC_ID_PCM_S16LE;
    par.format = AV_SAMPLE_FMT_S16;
    par.sample_rate = kSampleRate;
    av_channel_layout_default(&par.ch_layout, 2);
    par.bits_per_coded_sample = 16;
    par.block_align = kAudioBytesPerSample;
    par.bit_rate = static_cast<std::int64_t>(kSampleRate) * kAudioBytesPerSample * 8;
    st->time_base = {1, kSampleRate};
    st->disposition = audio.is_default ? AV_DISPOSITION_DEFAULT : 0;
    set_tags(*st, audio.language, audio.title);
    audio_.push_back(st);
  }

  void add_subtitles(const SubtitleSpec& subs) {
    AVStream* st = avformat_new_stream(out_.ctx, nullptr);
    st->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;
    st->codecpar->codec_id = AV_CODEC_ID_ASS;
    set_extradata(*st->codecpar, kAssHeader);
    st->time_base = {1, 1000};
    st->disposition =
        (subs.is_default ? AV_DISPOSITION_DEFAULT : 0) | (subs.forced ? AV_DISPOSITION_FORCED : 0);
    set_tags(*st, subs.language, subs.title);
    subtitles_.push_back(st);
  }

  void add_attachment(const AttachmentSpec& attachment) {
    AVStream* st = avformat_new_stream(out_.ctx, nullptr);
    st->codecpar->codec_type = AVMEDIA_TYPE_ATTACHMENT;
    st->codecpar->codec_id = AV_CODEC_ID_TTF;
    set_extradata(*st->codecpar, attachment.data);
    av_dict_set(&st->metadata, "filename", attachment.filename.c_str(), 0);
    av_dict_set(&st->metadata, "mimetype", attachment.mime_type.c_str(), 0);
  }

  void add_chapters() {
    if (spec_.chapters.empty()) return;
    out_.ctx->chapters = static_cast<AVChapter**>(av_calloc(spec_.chapters.size(), sizeof(AVChapter*)));
    for (const ChapterSpec& spec : spec_.chapters) {
      auto* chapter = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
      chapter->id = out_.ctx->nb_chapters + 1;
      chapter->time_base = {1, 1000};
      chapter->start = spec.start_ms;
      chapter->end = spec.end_ms;
      av_dict_set(&chapter->metadata, "title", spec.title.c_str(), 0);
      out_.ctx->chapters[out_.ctx->nb_chapters++] = chapter;
    }
  }

  std::expected<void, std::string> drain_encoder() {
    while (true) {
      const int err = avcodec_receive_packet(out_.encoder, out_.packet);
      if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return {};
      if (err < 0) return fail("paquet vidéo", err);
      av_packet_rescale_ts(out_.packet, out_.encoder->time_base, video_->time_base);
      out_.packet->stream_index = video_->index;
      if (const int w = av_interleaved_write_frame(out_.ctx, out_.packet); w < 0)
        return fail("écriture vidéo", w);
    }
  }

  std::expected<void, std::string> encode_video() {
    out_.frame = av_frame_alloc();
    out_.packet = av_packet_alloc();
    out_.frame->format = spec_.pixel_format;
    out_.frame->width = spec_.width;
    out_.frame->height = spec_.height;
    if (const int err = av_frame_get_buffer(out_.frame, 0); err < 0) return fail("image", err);
    for (int n = 0; n < spec_.frames; ++n) {
      av_frame_make_writable(out_.frame);
      fill_frame(*out_.frame, spec_, n);
      out_.frame->pts = n;
      out_.frame->quality = out_.encoder->global_quality;
      if (const int err = avcodec_send_frame(out_.encoder, out_.frame); err < 0) return fail("encodage", err);
      if (auto ok = drain_encoder(); !ok) return ok;
    }
    avcodec_send_frame(out_.encoder, nullptr);
    return drain_encoder();
  }

  std::expected<void, std::string> write_audio() {
    const double seconds = spec_.frames * av_q2d(av_inv_q(spec_.frame_rate));
    const auto packets = static_cast<int>(std::ceil(seconds * kSampleRate / kSamplesPerPacket));
    std::string pcm(static_cast<std::size_t>(kSamplesPerPacket * kAudioBytesPerSample), '\0');
    for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] = static_cast<char>(i % 251);
    for (AVStream* st : audio_) {
      for (int k = 0; k < packets; ++k) {
        if (av_new_packet(out_.packet, static_cast<int>(pcm.size())) < 0) return fail("paquet audio");
        std::copy(pcm.begin(), pcm.end(), out_.packet->data);
        out_.packet->pts = out_.packet->dts =
            av_rescale_q(static_cast<std::int64_t>(k) * kSamplesPerPacket, {1, kSampleRate}, st->time_base);
        out_.packet->duration = av_rescale_q(kSamplesPerPacket, {1, kSampleRate}, st->time_base);
        out_.packet->stream_index = st->index;
        out_.packet->flags = AV_PKT_FLAG_KEY;
        if (const int err = av_interleaved_write_frame(out_.ctx, out_.packet); err < 0)
          return fail("audio", err);
      }
    }
    return {};
  }

  std::expected<void, std::string> write_subtitles() {
    for (std::size_t s = 0; s < subtitles_.size(); ++s) {
      AVStream* st = subtitles_[s];
      int order = 0;
      for (const SubtitleEvent& event : spec_.subtitles[s].events) {
        const std::string line = std::format("{},0,Default,,0,0,0,,{}", order++, event.text);
        if (av_new_packet(out_.packet, static_cast<int>(line.size())) < 0) return fail("paquet ASS");
        std::copy(line.begin(), line.end(), out_.packet->data);
        out_.packet->pts = out_.packet->dts = av_rescale_q(event.start_ms, {1, 1000}, st->time_base);
        out_.packet->duration = av_rescale_q(event.duration_ms, {1, 1000}, st->time_base);
        out_.packet->stream_index = st->index;
        out_.packet->flags = AV_PKT_FLAG_KEY;
        if (const int err = av_interleaved_write_frame(out_.ctx, out_.packet); err < 0)
          return fail("ASS", err);
      }
    }
    return {};
  }

  const ClipSpec& spec_;
  Output out_;
  AVStream* video_ = nullptr;
  std::vector<AVStream*> audio_;
  std::vector<AVStream*> subtitles_;
};

}  // namespace

int pattern(int plane, int x, int y, int n, int bits) {
  const int max = (1 << bits) - 1;
  switch (plane) {
    case 0:
      return (x * 3 + y * 5 + n * 7) * (bits > 8 ? 4 : 1) & max;
    case 1:
      return (x * 11 + n * 13 + 64) * (bits > 8 ? 4 : 1) & max;
    default:
      return (y * 17 + n * 3 + 128) * (bits > 8 ? 4 : 1) & max;
  }
}

int flat_luma(int n) {
  return 32 + 6 * n;
}

int frame_of_flat_luma(int value) {
  const int n = static_cast<int>(std::lround((value - 32) / 6.0));
  return std::abs(flat_luma(n) - value) <= 3 ? n : -1;
}

std::expected<std::filesystem::path, std::string> write_clip(const ClipSpec& spec) {
  const std::filesystem::path dir = AJP_MEDIA_CLIP_DIR;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return fail(std::format("dossier {} : {}", dir.string(), ec.message()));
  const std::filesystem::path path = dir / spec.name;
  // Les tests tournent en processus parallèles : chacun écrit son temporaire.
  const std::filesystem::path temporary = dir / std::format("{}.{}.tmp", spec.name, std::random_device{}());
  {
    Writer writer(spec);  // fermé avant le renommage
    if (auto written = writer.write(temporary); !written) return std::unexpected(written.error());
  }
  std::filesystem::rename(temporary, path, ec);
  if (ec) return fail(std::format("renommage : {}", ec.message()));
  return path;
}

}  // namespace ajp::media::test
