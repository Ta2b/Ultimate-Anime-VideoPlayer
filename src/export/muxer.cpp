#include "export/muxer.h"

#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
}

#include "core/log.h"

namespace ajp::exporter {
namespace {

// Les paquets copiés passent jusqu'à 0,5 s devant la vidéo : le mux les entrelace sans attendre.
constexpr std::int64_t kCopyLeadUs = 500'000;

std::int64_t to_us(std::int64_t ts, AVRational time_base) {
  return av_rescale_q(ts, time_base, kMicroseconds);
}

const char* tag(const AVDictionary* metadata, const char* key) {
  const AVDictionaryEntry* entry = av_dict_get(metadata, key, nullptr, 0);
  return entry != nullptr ? entry->value : nullptr;
}

// Une pièce jointe que le MKV sait écrire : un nom, et un type MIME donné, déductible du codec ou
// de l'extension d'une police. Le type manquant est posé sur la sortie.
bool attachment_ready(const AVStream& in, AVDictionary** out_metadata) {
  const char* filename = tag(in.metadata, "filename");
  if (filename == nullptr) return false;
  if (tag(in.metadata, "mimetype") != nullptr) return true;
  const AVCodecDescriptor* descriptor = avcodec_descriptor_get(in.codecpar->codec_id);
  if (descriptor != nullptr && descriptor->mime_types != nullptr) return true;
  const std::string_view name(filename);
  const char* mime = nullptr;
  if (name.ends_with(".ttf") || name.ends_with(".TTF") || name.ends_with(".ttc") || name.ends_with(".TTC")) {
    mime = "application/x-truetype-font";
  } else if (name.ends_with(".otf") || name.ends_with(".OTF")) {
    mime = "application/vnd.ms-opentype";
  }
  if (mime == nullptr) return false;
  av_dict_set(out_metadata, "mimetype", mime, 0);
  return true;
}

}  // namespace

std::expected<Muxer, Refusal> Muxer::open(const std::filesystem::path& source,
                                          const std::filesystem::path& output) {
  Muxer m;
  m.output_path_ = output;
  AVFormatContext* input = nullptr;
  const std::string source_text = utf8(source);
  if (const int err = avformat_open_input(&input, source_text.c_str(), nullptr, nullptr); err < 0)
    return refuse_av(RefusalCode::MediaUnreadable, source_text, err);
  m.input_.reset(input);
  if (const int err = avformat_find_stream_info(input, nullptr); err < 0)
    return refuse_av(RefusalCode::MediaUnreadable, std::format("{} : flux", source_text), err);
  m.source_video_ = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (input->start_time != AV_NOPTS_VALUE)
    m.start_us_ = av_rescale_q(input->start_time, AV_TIME_BASE_Q, kMicroseconds);
  const std::string output_text = utf8(output);
  if (const int err = avformat_alloc_output_context2(&m.output_, nullptr, "matroska", output_text.c_str());
      err < 0)
    return refuse_av(RefusalCode::FileUnwritable, output_text, err);
  m.pending_.reset(av_packet_alloc());
  if (!m.pending_) return refuse(RefusalCode::EncodeFailed, "mux : mémoire");
  return m;
}

Muxer::Muxer(Muxer&& other) noexcept
    : output_path_(std::move(other.output_path_)),
      input_(std::move(other.input_)),
      output_(std::exchange(other.output_, nullptr)),
      map_(std::move(other.map_)),
      source_video_(other.source_video_),
      encoder_time_base_(other.encoder_time_base_),
      start_us_(other.start_us_),
      pending_(std::move(other.pending_)),
      pending_us_(other.pending_us_),
      written_(other.written_),
      copies_done_(other.copies_done_),
      opened_(std::exchange(other.opened_, false)),
      finished_(other.finished_) {}

Muxer::~Muxer() {
  if (opened_ && !finished_) {
    abandon();
    return;
  }
  close_output();
}

void Muxer::close_output() {
  if (output_ == nullptr) return;
  if (output_->pb != nullptr && (output_->oformat->flags & AVFMT_NOFILE) == 0) avio_closep(&output_->pb);
  avformat_free_context(output_);
  output_ = nullptr;
}

void Muxer::abandon() {
  close_output();
  if (opened_) {
    std::error_code ignored;
    std::filesystem::remove(output_path_, ignored);
    opened_ = false;
  }
}

AVRational Muxer::source_aspect() const {
  if (source_video_ < 0) return {0, 1};
  const AVStream& video = *input_->streams[source_video_];
  return video.sample_aspect_ratio.num > 0 ? video.sample_aspect_ratio : video.codecpar->sample_aspect_ratio;
}

ColorInfo Muxer::source_color() const {
  return source_video_ < 0 ? ColorInfo{} : color_of(*input_->streams[source_video_]->codecpar);
}

std::int64_t Muxer::bytes() const {
  return output_ != nullptr && output_->pb != nullptr ? avio_tell(output_->pb) : written_;
}

std::expected<void, Refusal> Muxer::begin(const AVCodecContext& encoder) {
  const auto log = log_channel("export");
  encoder_time_base_ = encoder.time_base;
  const AVFormatContext& in = *input_;
  AVFormatContext& out = *output_;
  av_dict_copy(&out.metadata, in.metadata, 0);

  // La vidéo encodée d'abord : les lecteurs prennent la première piste vidéo.
  AVStream* video = avformat_new_stream(&out, nullptr);
  if (video == nullptr) return refuse(RefusalCode::EncodeFailed, "mux : piste vidéo");
  if (const int err = avcodec_parameters_from_context(video->codecpar, &encoder); err < 0)
    return refuse_av(RefusalCode::EncodeFailed, "mux : paramètres vidéo", err);
  video->time_base = encoder.time_base;
  video->avg_frame_rate = encoder.framerate;
  video->sample_aspect_ratio = encoder.sample_aspect_ratio;
  video->disposition = AV_DISPOSITION_DEFAULT;
  if (source_video_ >= 0) {
    for (const char* key : {"language", "title"}) {
      if (const char* value = tag(in.streams[source_video_]->metadata, key))
        av_dict_set(&video->metadata, key, value, 0);
    }
  }

  // Les autres pistes de la source, copiées, si le MKV sait les porter.
  map_.assign(in.nb_streams, -1);
  for (unsigned i = 0; i < in.nb_streams; ++i) {
    AVStream& st = *in.streams[i];
    const AVMediaType type = st.codecpar->codec_type;
    const bool media = type == AVMEDIA_TYPE_AUDIO || type == AVMEDIA_TYPE_SUBTITLE;
    AVDictionary* extra = nullptr;
    bool keep = false;
    if (media) {
      keep = avformat_query_codec(out.oformat, st.codecpar->codec_id, FF_COMPLIANCE_NORMAL) == 1;
      if (!keep)
        log->warn("piste {} ({}) non copiée : le MKV ne la porte pas", i,
                  avcodec_get_name(st.codecpar->codec_id));
    } else if (type == AVMEDIA_TYPE_ATTACHMENT) {
      keep = attachment_ready(st, &extra);
      if (!keep) log->warn("pièce jointe {} non copiée : ni nom ni type", i);
    }
    if (!keep) {
      av_dict_free(&extra);
      st.discard = AVDISCARD_ALL;
      continue;
    }
    AVStream* copy = avformat_new_stream(&out, nullptr);
    if (copy == nullptr) {
      av_dict_free(&extra);
      return refuse(RefusalCode::EncodeFailed, "mux : piste copiée");
    }
    if (const int err = avcodec_parameters_copy(copy->codecpar, st.codecpar); err < 0) {
      av_dict_free(&extra);
      return refuse_av(RefusalCode::EncodeFailed, "mux : paramètres copiés", err);
    }
    copy->codecpar->codec_tag = 0;
    copy->time_base = st.time_base;
    copy->disposition = st.disposition;
    av_dict_copy(&copy->metadata, st.metadata, 0);
    av_dict_copy(&copy->metadata, extra, 0);
    av_dict_free(&extra);
    map_[i] = copy->index;
  }

  // Les chapitres, comme ffmpeg -map_chapters les copie.
  if (in.nb_chapters > 0) {
    out.chapters = static_cast<AVChapter**>(av_calloc(in.nb_chapters, sizeof(AVChapter*)));
    if (out.chapters == nullptr) return refuse(RefusalCode::EncodeFailed, "mux : chapitres");
    for (unsigned i = 0; i < in.nb_chapters; ++i) {
      const AVChapter& from = *in.chapters[i];
      auto* to = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
      if (to == nullptr) return refuse(RefusalCode::EncodeFailed, "mux : chapitre");
      to->id = from.id;
      to->time_base = from.time_base;
      to->start = from.start;
      to->end = from.end;
      av_dict_copy(&to->metadata, from.metadata, 0);
      out.chapters[out.nb_chapters++] = to;
    }
  }

  if ((out.oformat->flags & AVFMT_NOFILE) == 0) {
    const std::string path = utf8(output_path_);
    if (const int err = avio_open(&out.pb, path.c_str(), AVIO_FLAG_WRITE); err < 0)
      return refuse_av(RefusalCode::FileUnwritable, path, err);
    opened_ = true;
  }
  if (const int err = avformat_write_header(&out, nullptr); err < 0)
    return refuse_av(RefusalCode::EncodeFailed, "mux : en-tête", err);
  return read_copy();
}

std::expected<void, Refusal> Muxer::read_copy() {
  while (true) {
    const int err = av_read_frame(input_.get(), pending_.get());
    if (err == AVERROR_EOF) {
      copies_done_ = true;
      return {};
    }
    if (err < 0) {  // la fin de la source illisible : la vidéo encodée reste, les copies s'arrêtent
      log_channel("export")->warn("lecture des pistes copiées arrêtée : {}", av_error_text(err));
      copies_done_ = true;
      return {};
    }
    const int index = pending_->stream_index;
    if (index < 0 || static_cast<std::size_t>(index) >= map_.size() ||
        map_[static_cast<std::size_t>(index)] < 0) {
      av_packet_unref(pending_.get());
      continue;
    }
    const AVRational time_base = input_->streams[index]->time_base;
    const std::int64_t ts = pending_->dts != AV_NOPTS_VALUE ? pending_->dts : pending_->pts;
    if (ts != AV_NOPTS_VALUE) pending_us_ = to_us(ts, time_base);  // sinon : avec le précédent
    return {};
  }
}

std::expected<void, Refusal> Muxer::write_copies(std::int64_t until_us) {
  while (!copies_done_ && pending_us_ <= until_us) {
    const auto index = static_cast<std::size_t>(pending_->stream_index);
    AVStream* to = output_->streams[map_[index]];
    av_packet_rescale_ts(pending_.get(), input_->streams[index]->time_base, to->time_base);
    pending_->stream_index = to->index;
    pending_->pos = -1;
    if (const int err = av_interleaved_write_frame(output_, pending_.get()); err < 0)
      return refuse_av(RefusalCode::EncodeFailed, "mux : paquet copié", err);
    if (auto next = read_copy(); !next) return next;
  }
  return {};
}

std::expected<void, Refusal> Muxer::write_video(AVPacket& packet) {
  const std::int64_t ts = packet.dts != AV_NOPTS_VALUE ? packet.dts : packet.pts;
  if (ts != AV_NOPTS_VALUE) {
    if (auto copied = write_copies(to_us(ts, encoder_time_base_) + kCopyLeadUs); !copied) return copied;
  }
  AVStream* video = output_->streams[0];
  av_packet_rescale_ts(&packet, encoder_time_base_, video->time_base);
  packet.stream_index = video->index;
  if (const int err = av_interleaved_write_frame(output_, &packet); err < 0)
    return refuse_av(RefusalCode::EncodeFailed, "mux : paquet vidéo", err);
  return {};
}

std::expected<void, Refusal> Muxer::finish() {
  if (auto copied = write_copies(std::numeric_limits<std::int64_t>::max()); !copied) return copied;
  if (const int err = av_write_trailer(output_); err < 0)
    return refuse_av(RefusalCode::EncodeFailed, "mux : fin du fichier", err);
  written_ = bytes();
  finished_ = true;
  close_output();
  return {};
}

}  // namespace ajp::exporter
