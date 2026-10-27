#include "media/demuxer.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <format>
#include <string>
#include <string_view>
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
  PacketPtr scratch;            // reçoit av_read_frame avant d'être partagé
  std::deque<Packet> pushback;  // lus par le seek pour vérifier son point clé, rendus par read()
  std::shared_ptr<spdlog::logger> log = make_logger();
  std::uint32_t file_seeks = 0;
  std::atomic<bool> interrupted{false};  // lu par le rappel d'interruption de FFmpeg

  std::expected<std::optional<Packet>, Refusal> read_file() const;
  std::expected<std::optional<Packet>, Refusal> peek_keyframe(int reference);
  std::expected<void, Refusal> seek_stream(int reference, std::int64_t ts, int flags, std::string_view what);
  std::expected<std::int64_t, Refusal> seek_before(std::int64_t target_us, int reference);
  std::expected<std::optional<std::int64_t>, Refusal> seek_after(std::int64_t target_us, int reference);
  std::int64_t indexed_before(std::int64_t target_us, int reference) const;
};

namespace {

// Paquets lus au plus pour trouver le premier point clé après un seek.
constexpr int kMaxPeekPackets = 1024;
// Reculs au plus pour trouver un point clé qui s'affiche à ou avant la cible.
constexpr int kMaxRewinds = 8;

// FFmpeg attend un chemin UTF-8, Windows compris : path::string() y rendrait la
// page de code ANSI, ou lèverait une exception devant un nom en japonais.
std::string utf8_path(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

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
    const AVRational sar =
        stream.sample_aspect_ratio.num > 0 ? stream.sample_aspect_ratio : par.sample_aspect_ratio;
    if (sar.num > 0 && sar.den > 0) info.sample_aspect = {sar.num, sar.den};
    switch (par.color_space) {
      case AVCOL_SPC_BT709:
        info.color_matrix = ColorMatrix::Bt709;
        break;
      case AVCOL_SPC_BT470BG:
      case AVCOL_SPC_SMPTE170M:
        info.color_matrix = ColorMatrix::Bt601;
        break;
      case AVCOL_SPC_BT2020_NCL:
      case AVCOL_SPC_BT2020_CL:
        info.color_matrix = ColorMatrix::Bt2020Ncl;
        break;
      default:
        break;
    }
    info.full_range = par.color_range == AVCOL_RANGE_JPEG;
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
  const std::string where = utf8_path(path);
  AVFormatContext* raw = avformat_alloc_context();
  if (raw == nullptr) return refuse(RefusalCode::MediaUnreadable, where + " : mémoire insuffisante");
  // Le rappel d'interruption lit le drapeau de l'état, qui garde son adresse quand le Demuxer bouge.
  raw->interrupt_callback = {
      .callback = [](void* opaque) { return static_cast<State*>(opaque)->interrupted.load() ? 1 : 0; },
      .opaque = state.get()};
  // Un partage réseau muet rend la main au bout de 30 s ; sans effet sur un fichier local.
  AVDictionary* options = nullptr;
  av_dict_set(&options, "rw_timeout", "30000000", 0);
  const int opened = avformat_open_input(&raw, where.c_str(), nullptr, &options);
  av_dict_free(&options);
  if (opened < 0) return refuse_av(RefusalCode::MediaUnreadable, where, opened);
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

void Demuxer::interrupt() {
  state_->interrupted.store(true);
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
  const auto kept = [tracks](int index) { return std::ranges::find(tracks, index) != tracks.end(); };
  const AVFormatContext& ctx = *state_->format;
  for (unsigned i = 0; i < ctx.nb_streams; ++i) {
    ctx.streams[i]->discard = kept(static_cast<int>(i)) ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
  }
  std::erase_if(state_->pushback, [&](const Packet& packet) { return !kept(packet.track); });
  return {};
}

std::expected<std::optional<Packet>, Refusal> Demuxer::read() {
  if (!state_->pushback.empty()) {
    Packet packet = std::move(state_->pushback.front());
    state_->pushback.pop_front();
    return packet;
  }
  return state_->read_file();
}

std::expected<std::optional<Packet>, Refusal> Demuxer::State::read_file() const {
  AVFormatContext& ctx = *format;
  AVPacket* raw = scratch.get();
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
  raw->pts = raw->pts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE : to_us(raw->pts, time_base, start_us);
  raw->dts = raw->dts == AV_NOPTS_VALUE ? AV_NOPTS_VALUE : to_us(raw->dts, time_base, start_us);
  raw->duration = duration_to_us(raw->duration, time_base);
  raw->time_base = kMicroseconds;

  Packet packet{
      .track = raw->stream_index,
      .pts_us = raw->pts == AV_NOPTS_VALUE ? kNoPts : raw->pts,
      .dts_us = raw->dts == AV_NOPTS_VALUE ? kNoPts : raw->dts,
      .duration_us = raw->duration,
      .keyframe = (raw->flags & AV_PKT_FLAG_KEY) != 0,
  };
  std::shared_ptr<AVPacket> shared(av_packet_alloc(), PacketFree{});
  if (!shared) return refuse(RefusalCode::DecodeFailed, "lecture d'un paquet : mémoire insuffisante");
  av_packet_move_ref(shared.get(), raw);
  packet.av = std::move(shared);
  return packet;
}

// Lit jusqu'au premier point clé de `reference` ; tout ce qui est lu reste dans
// pushback, dans l'ordre du fichier. std::nullopt : fin du fichier ou borne atteinte.
std::expected<std::optional<Packet>, Refusal> Demuxer::State::peek_keyframe(int reference) {
  for (int count = 0; count < kMaxPeekPackets; ++count) {
    auto packet = read_file();
    if (!packet) return std::unexpected(packet.error());
    if (!*packet) return std::nullopt;
    pushback.push_back(**packet);
    if ((*packet)->track == reference && (*packet)->keyframe) return packet;
  }
  return std::nullopt;
}

std::expected<void, Refusal> Demuxer::State::seek_stream(int reference, std::int64_t ts, int flags,
                                                         std::string_view what) {
  pushback.clear();
  ++file_seeks;
  if (const int err = av_seek_frame(format.get(), reference, ts, flags); err < 0) {
    return refuse_av(RefusalCode::SeekFailed, what, err);
  }
  return {};
}

// av_seek_frame peut rendre un point clé qui s'affiche après la cible : index en
// dts (MP4 avec images B), GOP ouvert. Les images entre la cible et lui ne se
// décoderaient jamais. On recule alors avant son dts, au plus kMaxRewinds fois,
// ou jusqu'à retrouver le même point clé : la cible précède la première image.
// Rend le pts du point clé où la lecture reprend, kNoPts s'il est inconnu.
std::expected<std::int64_t, Refusal> Demuxer::State::seek_before(std::int64_t target_us, int reference) {
  const AVRational time_base = format->streams[reference]->time_base;
  const auto land = [&](std::int64_t aim_us) -> std::expected<std::optional<Packet>, Refusal> {
    auto moved = seek_stream(reference, from_us(aim_us, time_base, start_us), AVSEEK_FLAG_BACKWARD,
                             std::format("seek à {} µs", aim_us));
    if (!moved) return std::unexpected(moved.error());
    return peek_keyframe(reference);
  };
  std::int64_t aim_us = target_us;
  std::int64_t landed_us = target_us;  // visée du dernier seek abouti
  std::int64_t last_key_us = kNoPts;
  for (int rewind = 0; rewind <= kMaxRewinds; ++rewind) {
    auto key = land(aim_us);
    if (!key && rewind > 0) {
      // Un conteneur peut refuser une visée avant son premier point clé (avidec,
      // recherche générique) : celui déjà trouvé est le premier, on repart de lui.
      log->debug("seek à {} µs : recul refusé ({}), retour à {} µs", target_us, key.error().detail,
                 landed_us);
      key = land(landed_us);
    }
    if (!key) return std::unexpected(key.error());
    if (!*key) return kNoPts;
    const std::int64_t key_pts = (*key)->pts_us;
    if (key_pts == kNoPts || key_pts <= target_us || key_pts == last_key_us) return key_pts;
    last_key_us = key_pts;
    landed_us = aim_us;
    const std::int64_t key_us = (*key)->dts_us != kNoPts ? (*key)->dts_us : key_pts;
    aim_us = std::min(aim_us, key_us) - 1;
    log->debug("seek à {} µs : point clé affiché à {} µs, recul à {} µs", target_us, key_pts, aim_us);
  }
  return last_key_us;
}

// Premier point clé à ou après la cible, d'après le pts lu ; std::nullopt s'il
// n'y en a pas (cible après le dernier point clé).
std::expected<std::optional<std::int64_t>, Refusal> Demuxer::State::seek_after(std::int64_t target_us,
                                                                               int reference) {
  const AVRational time_base = format->streams[reference]->time_base;
  pushback.clear();
  ++file_seeks;
  if (av_seek_frame(format.get(), reference, from_us(target_us, time_base, start_us), 0) < 0) {
    return std::nullopt;
  }
  auto key = peek_keyframe(reference);
  if (!key) return std::unexpected(key.error());
  if (!*key || (*key)->pts_us == kNoPts) return std::nullopt;
  return (*key)->pts_us;
}

// Le point clé à ou avant la cible d'après l'index du conteneur, quand cet index est en pts
// (MKV : les Cues, chargées par le premier seek) ; kNoPts sinon (MP4 range le sien en dts,
// d'autres n'en ont pas).
std::int64_t Demuxer::State::indexed_before(std::int64_t target_us, int reference) const {
  if (std::string_view(format->iformat->name).find("matroska") == std::string_view::npos) return kNoPts;
  AVStream* stream = format->streams[reference];
  const int at = av_index_search_timestamp(stream, from_us(target_us, stream->time_base, start_us),
                                           AVSEEK_FLAG_BACKWARD);
  const AVIndexEntry* entry = at < 0 ? nullptr : avformat_index_get_entry(stream, at);
  if (entry == nullptr || (entry->flags & AVINDEX_KEYFRAME) == 0) return kNoPts;
  return to_us(entry->timestamp, stream->time_base, start_us);
}

std::expected<void, Refusal> Demuxer::seek(std::int64_t target_us, int reference, SeekTarget where) {
  if (target_us < 0 || (state_->duration_us > 0 && target_us > state_->duration_us)) {
    return refuse(RefusalCode::SeekFailed,
                  std::format("cible {} µs hors de [0, {}] µs", target_us, state_->duration_us));
  }
  if (track(reference) == nullptr)
    return refuse(RefusalCode::TrackUnavailable, std::format("piste {}", reference));

  if (where == SeekTarget::PreviousKeyframe) {
    auto before = state_->seek_before(target_us, reference);
    if (!before) return std::unexpected(before.error());
    return {};
  }
  // Seek rapide : les deux points clés qui encadrent la cible, comparés sur
  // leurs pts lus (un index en dts tromperait), et le plus proche gagne.
  auto after = state_->seek_after(target_us, reference);
  if (!after) return std::unexpected(after.error());
  // Index en pts : le point clé d'avant s'y lit sans repositionner le fichier, et la lecture
  // reste sur celui d'après quand il est le plus proche.
  if (*after) {
    const std::int64_t indexed = state_->indexed_before(target_us, reference);
    if (indexed != kNoPts && indexed <= target_us && **after - target_us < target_us - indexed) return {};
  }
  auto before = state_->seek_before(target_us, reference);
  if (!before) return std::unexpected(before.error());
  if (!*after || *before == kNoPts || **after - target_us >= target_us - *before) return {};
  auto again = state_->seek_after(target_us, reference);
  if (!again) return std::unexpected(again.error());
  return {};
}

std::uint32_t Demuxer::file_seeks() const {
  return state_->file_seeks;
}

}  // namespace ajp::media
