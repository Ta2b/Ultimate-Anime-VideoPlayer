#include "media/demuxer.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

extern "C" {
#include <libavutil/dict.h>
}

#include "media/av_util.h"
#include "media/log.h"

namespace ajp::media {

struct Demuxer::State {
  FormatPtr format;
  std::vector<TrackInfo> tracks;
  std::vector<Chapter> chapters;
  std::int64_t start_us = 0;
  std::int64_t duration_us = 0;
  PacketPtr scratch;  // reçoit av_read_frame avant d'être partagé
  std::shared_ptr<spdlog::logger> log = make_logger();
};

namespace {

std::string tag(const AVDictionary* metadata, const char* key) {
  const AVDictionaryEntry* entry = av_dict_get(metadata, key, nullptr, 0);
  return entry != nullptr && entry->value != nullptr ? std::string(entry->value) : std::string{};
}

std::optional<TrackKind> kind_of(AVMediaType type) {
  switch (type) {
    case AVMEDIA_TYPE_VIDEO:
      return TrackKind::Video;
    case AVMEDIA_TYPE_AUDIO:
      return TrackKind::Audio;
    case AVMEDIA_TYPE_SUBTITLE:
      return TrackKind::Subtitle;
    case AVMEDIA_TYPE_ATTACHMENT:
      return TrackKind::Attachment;
    default:
      return std::nullopt;
  }
}

Rational frame_rate_of(const AVStream& stream) {
  AVRational rate = stream.avg_frame_rate;
  if (rate.num <= 0 || rate.den <= 0) rate = stream.r_frame_rate;
  if (rate.num <= 0 || rate.den <= 0) return {};
  return {rate.num, rate.den};
}

std::expected<TrackInfo, Refusal> describe(const AVStream& stream, TrackKind kind) {
  const AVCodecParameters& par = *stream.codecpar;
  auto parameters = CodecParameters::copy_of(par);
  if (!parameters) return std::unexpected(parameters.error());

  TrackInfo info{
      .index = stream.index,
      .kind = kind,
      .codec = avcodec_get_name(par.codec_id),
      .language = tag(stream.metadata, "language"),
      .title = tag(stream.metadata, "title"),
      .is_default = (stream.disposition & AV_DISPOSITION_DEFAULT) != 0,
      .forced = (stream.disposition & AV_DISPOSITION_FORCED) != 0,
      .attached_picture = (stream.disposition & AV_DISPOSITION_ATTACHED_PIC) != 0,
      .duration_us = duration_to_us(stream.duration, stream.time_base),
      .parameters = std::move(*parameters),
  };
  if (info.language == "und") info.language.clear();  // « indéterminée » : absente
  if (kind == TrackKind::Video) {
    info.width = par.width;
    info.height = par.height;
    info.frame_rate = frame_rate_of(stream);
  } else if (kind == TrackKind::Audio) {
    info.sample_rate = par.sample_rate;
    info.channels = par.ch_layout.nb_channels;
  } else if (kind == TrackKind::Attachment) {
    info.filename = tag(stream.metadata, "filename");
    info.mime_type = tag(stream.metadata, "mimetype");
  }
  return info;
}

}  // namespace

Demuxer::Demuxer(std::unique_ptr<State> state) : state_(std::move(state)) {}
Demuxer::Demuxer(Demuxer&&) noexcept = default;
Demuxer& Demuxer::operator=(Demuxer&&) noexcept = default;
Demuxer::~Demuxer() = default;

std::expected<Demuxer, Refusal> Demuxer::open(const std::filesystem::path& path) {
  auto state = std::make_unique<State>();
  const std::string where = path.string();
  AVFormatContext* raw = nullptr;
  if (const int err = avformat_open_input(&raw, where.c_str(), nullptr, nullptr); err < 0) {
    return refuse_av(RefusalCode::MediaUnreadable, where, err);
  }
  state->format.reset(raw);
  if (const int err = avformat_find_stream_info(raw, nullptr); err < 0) {
    return refuse_av(RefusalCode::MediaUnreadable, where, err);
  }
  state->start_us = raw->start_time == AV_NOPTS_VALUE ? 0 : raw->start_time;
  state->duration_us = raw->duration == AV_NOPTS_VALUE || raw->duration < 0 ? 0 : raw->duration;

  for (unsigned i = 0; i < raw->nb_streams; ++i) {
    const AVStream& stream = *raw->streams[i];
    const std::optional<TrackKind> kind = kind_of(stream.codecpar->codec_type);
    if (!kind) continue;
    auto info = describe(stream, *kind);
    if (!info) return std::unexpected(info.error());
    state->tracks.push_back(std::move(*info));
  }
  for (unsigned i = 0; i < raw->nb_chapters; ++i) {
    const AVChapter& chapter = *raw->chapters[i];
    state->chapters.push_back({.start_us = to_us(chapter.start, chapter.time_base, state->start_us),
                               .end_us = to_us(chapter.end, chapter.time_base, state->start_us),
                               .title = tag(chapter.metadata, "title")});
  }
  state->scratch.reset(av_packet_alloc());
  if (!state->scratch) return refuse(RefusalCode::MediaUnreadable, where + " : mémoire insuffisante");
  state->log->debug("{} : {} pistes, {} chapitres, {} µs", where, state->tracks.size(),
                    state->chapters.size(), state->duration_us);
  return Demuxer(std::move(state));
}

const std::vector<TrackInfo>& Demuxer::tracks() const {
  return state_->tracks;
}
const std::vector<Chapter>& Demuxer::chapters() const {
  return state_->chapters;
}
std::int64_t Demuxer::duration_us() const {
  return state_->duration_us;
}

const TrackInfo* Demuxer::track(int index) const {
  const auto it = std::ranges::find(state_->tracks, index, &TrackInfo::index);
  return it == state_->tracks.end() ? nullptr : &*it;
}

std::expected<void, Refusal> Demuxer::keep_only(std::span<const int> tracks) {
  for (const int index : tracks) {
    if (track(index) == nullptr) return refuse(RefusalCode::TrackUnavailable, std::format("piste {}", index));
  }
  const AVFormatContext& ctx = *state_->format;
  for (unsigned i = 0; i < ctx.nb_streams; ++i) {
    const bool kept = std::ranges::find(tracks, static_cast<int>(i)) != tracks.end();
    ctx.streams[i]->discard = kept ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
  }
  return {};
}

std::expected<std::optional<Packet>, Refusal> Demuxer::read() {
  AVFormatContext& ctx = *state_->format;
  AVPacket* raw = state_->scratch.get();
  while (true) {
    if (const int err = av_read_frame(&ctx, raw); err < 0) {
      if (err == AVERROR_EOF) return std::nullopt;
      return refuse_av(RefusalCode::DecodeFailed, "lecture d'un paquet", err);
    }
    // Tous les conteneurs ne sautent pas eux-mêmes les flux écartés (MKV non).
    if (ctx.streams[raw->stream_index]->discard < AVDISCARD_ALL) break;
    av_packet_unref(raw);
  }
  const AVRational time_base = ctx.streams[raw->stream_index]->time_base;
  // Une seule échelle pour tout le lecteur : microsecondes depuis le début.
  raw->pts = raw->pts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE : to_us(raw->pts, time_base, state_->start_us);
  raw->dts = raw->dts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE : to_us(raw->dts, time_base, state_->start_us);
  raw->duration = duration_to_us(raw->duration, time_base);
  raw->time_base = kMicroseconds;

  Packet packet{
      .track = raw->stream_index,
      .pts_us = raw->pts == AV_NOPTS_VALUE ? kNoTimestamp : raw->pts,
      .dts_us = raw->dts == AV_NOPTS_VALUE ? kNoTimestamp : raw->dts,
      .duration_us = raw->duration,
      .keyframe = (raw->flags & AV_PKT_FLAG_KEY) != 0,
  };
  std::shared_ptr<AVPacket> shared(av_packet_alloc(), PacketFree{});
  if (!shared) return refuse(RefusalCode::DecodeFailed, "lecture d'un paquet : mémoire insuffisante");
  av_packet_move_ref(shared.get(), raw);
  packet.av = std::move(shared);
  return packet;
}

std::expected<void, Refusal> Demuxer::seek(std::int64_t target_us, int reference, SeekTarget where) {
  if (target_us < 0 || (state_->duration_us > 0 && target_us > state_->duration_us)) {
    return refuse(RefusalCode::SeekFailed,
                  std::format("cible {} µs hors de [0, {}] µs", target_us, state_->duration_us));
  }
  if (track(reference) == nullptr)
    return refuse(RefusalCode::TrackUnavailable, std::format("piste {}", reference));

  AVFormatContext& ctx = *state_->format;
  AVStream* stream = ctx.streams[reference];
  const std::int64_t ts = from_us(target_us, stream->time_base, state_->start_us);
  if (const int err = av_seek_frame(&ctx, reference, ts, AVSEEK_FLAG_BACKWARD); err < 0) {
    return refuse_av(RefusalCode::SeekFailed, std::format("seek à {} µs", target_us), err);
  }
  if (where == SeekTarget::PreviousKeyframe) return {};

  // Le premier seek a chargé l'index (les Cues d'un MKV) : on regarde si le
  // point clé suivant est plus proche que le précédent.
  const int before = av_index_search_timestamp(stream, ts, AVSEEK_FLAG_BACKWARD);
  const int after = av_index_search_timestamp(stream, ts, 0);
  if (before < 0 || after < 0) return {};
  const std::int64_t ts_before = avformat_index_get_entry(stream, before)->timestamp;
  const std::int64_t ts_after = avformat_index_get_entry(stream, after)->timestamp;
  if (ts_after - ts >= ts - ts_before) return {};
  if (const int err = av_seek_frame(&ctx, reference, ts_after, AVSEEK_FLAG_BACKWARD); err < 0) {
    return refuse_av(RefusalCode::SeekFailed, std::format("seek au point clé suivant {} µs", target_us), err);
  }
  return {};
}

}  // namespace ajp::media
