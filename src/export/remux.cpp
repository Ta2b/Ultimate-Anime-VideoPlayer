#include "export/remux.h"

#include <algorithm>
#include <format>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
}

#include "export/av_util.h"
#include "export/mux.h"
#include "export/side_data.h"

namespace ajp::exporter {
namespace {

constexpr std::int64_t kWriteNow = std::numeric_limits<std::int64_t>::min();

bool wanted(const AVStream& stream, const StreamCopy& copy) {
  switch (stream.codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      return copy.audio;
    case AVMEDIA_TYPE_SUBTITLE:
      return copy.subtitles;
    case AVMEDIA_TYPE_ATTACHMENT:
      return copy.attachments;
    default:
      return false;  // la vidéo source est remplacée par l'export ; les données sont ignorées
  }
}

std::string describe(const AVStream& stream) {
  const char* type = av_get_media_type_string(stream.codecpar->codec_type);
  return std::format("piste source {} ({}, {})", stream.index, type == nullptr ? "?" : type,
                     avcodec_get_name(stream.codecpar->codec_id));
}

// Une entrée lue paquet par paquet, avec son paquet en attente d'écriture.
struct Reader {
  InputPtr input;
  PacketPtr packet{av_packet_alloc()};
  std::vector<int> out_index;        // flux de sortie par flux d'entrée, -1 : ignoré
  std::vector<std::int64_t> offset;  // décalage à retirer, dans la base de temps du flux
  std::int64_t window_us = std::numeric_limits<std::int64_t>::max();
  bool pending = false;
  bool done = false;

  // Le prochain paquet à recopier, hors de la fenêtre écarté. Refus MediaUnreadable.
  std::expected<void, Refusal> advance(const std::string& path) {
    while (!pending && !done) {
      const int error = av_read_frame(input.get(), packet.get());
      if (error == AVERROR_EOF) {
        done = true;
        break;
      }
      if (error < 0)
        return refuse_av(RefusalCode::MediaUnreadable, std::format("av_read_frame {}", path), error);
      if (!shift_and_keep(*packet)) {
        av_packet_unref(packet.get());
        continue;
      }
      pending = true;
    }
    return {};
  }

  // Retire le décalage de la fenêtre ; faux pour un paquet à écarter.
  bool shift_and_keep(AVPacket& p) const {
    const auto index = static_cast<std::size_t>(p.stream_index);
    if (index >= out_index.size() || out_index[index] < 0) return false;
    const AVStream& stream = *input->streams[p.stream_index];
    const AVRational tb = stream.time_base;
    if (p.pts != AV_NOPTS_VALUE) p.pts -= offset[index];
    if (p.dts != AV_NOPTS_VALUE) p.dts -= offset[index];
    if (p.pts == AV_NOPTS_VALUE) return true;
    const std::int64_t start = av_rescale_q(p.pts, tb, kMicroseconds);
    const std::int64_t end = start + av_rescale_q(p.duration, tb, kMicroseconds);
    if (start >= window_us) return false;     // après la dernière image exportée
    if (start < 0 && end <= 0) return false;  // entièrement avant la première : écarté
    // Un sous-titre à cheval sur un bord de la fenêtre est coupé à ce bord, comme un chapitre :
    // gardé tel quel, son début négatif décalerait toute la sortie (Matroska n'écrit pas
    // d'horodatage négatif, avoid_negative_ts y décale tous les flux) et sa fin passerait la
    // dernière image. Un paquet audio, lui, ne se coupe pas : il garde son début.
    if (stream.codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE && p.duration > 0 &&
        (start < 0 || end > window_us)) {
      p.pts = av_rescale_q(std::max<std::int64_t>(start, 0), kMicroseconds, tb);
      if (p.dts != AV_NOPTS_VALUE) p.dts = p.pts;
      p.duration = av_rescale_q(std::min(end, window_us), kMicroseconds, tb) - p.pts;
    }
    return true;
  }

  // Clé d'entrelacement : dts (sinon pts) en microsecondes.
  std::int64_t key() const {
    const std::int64_t ts = packet->dts != AV_NOPTS_VALUE ? packet->dts : packet->pts;
    if (ts == AV_NOPTS_VALUE) return kWriteNow;
    return av_rescale_q(ts, input->streams[packet->stream_index]->time_base, kMicroseconds);
  }
};

std::expected<void, Refusal> add_source_stream(MuxOutput& out, Reader& reader, const AVStream& in,
                                               const RemuxRequest& request) {
  const bool attachment = in.codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT;
  if (attachment && request.container != Container::Matroska) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("{} : {} ne porte pas de pièce jointe", describe(in), name(request.container)));
  }
  auto stream = out.add_stream(*in.codecpar, in.time_base, describe(in), !attachment);
  if (!stream) return std::unexpected(stream.error());
  av_dict_copy(&(*stream)->metadata, in.metadata, 0);
  (*stream)->disposition = in.disposition;
  reader.out_index[static_cast<std::size_t>(in.index)] = (*stream)->index;
  return {};
}

// Chapitres recalés sur la fenêtre exportée, en microsecondes ; ceux qui en sortent sont écartés
// et ceux qui la chevauchent coupés à ses bords.
std::expected<void, Refusal> copy_chapters(AVFormatContext& out, const AVFormatContext& in,
                                           std::int64_t shift_us, std::int64_t window_us) {
  for (unsigned i = 0; i < in.nb_chapters; ++i) {
    const AVChapter& chapter = *in.chapters[i];
    const std::int64_t start = av_rescale_q(chapter.start, chapter.time_base, kMicroseconds) - shift_us;
    const std::int64_t end = av_rescale_q(chapter.end, chapter.time_base, kMicroseconds) - shift_us;
    if (end <= 0 || start >= window_us) continue;
    void* grown = av_realloc_array(static_cast<void*>(out.chapters), out.nb_chapters + 1, sizeof(AVChapter*));
    auto* copy = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
    if (grown == nullptr || copy == nullptr) {
      av_free(copy);
      if (grown != nullptr) out.chapters = static_cast<AVChapter**>(grown);
      return refuse(RefusalCode::MuxFailed, "chapitres : mémoire insuffisante");
    }
    out.chapters = static_cast<AVChapter**>(grown);
    copy->id = chapter.id;
    copy->time_base = kMicroseconds;
    copy->start = std::max<std::int64_t>(start, 0);
    copy->end = std::min(end, window_us);
    av_dict_copy(&copy->metadata, chapter.metadata, 0);
    out.chapters[out.nb_chapters++] = copy;
  }
  return {};
}

std::expected<void, Refusal> open_source(MuxOutput& out, Reader& reader, const RemuxRequest& request) {
  auto input = open_input(request.source.string());
  if (!input) return std::unexpected(input.error());
  reader.input = std::move(*input);
  const AVFormatContext& in = *reader.input;
  const std::int64_t start_us = in.start_time == AV_NOPTS_VALUE ? 0 : in.start_time;
  const std::int64_t shift_us = start_us + request.begin_us;
  reader.window_us = request.end_us - request.begin_us;
  reader.out_index.assign(in.nb_streams, -1);
  reader.offset.assign(in.nb_streams, 0);
  for (unsigned i = 0; i < in.nb_streams; ++i) {
    const AVStream& stream = *in.streams[i];
    reader.offset[i] = av_rescale_q(shift_us, kMicroseconds, stream.time_base);
    if (!wanted(stream, request.copy)) continue;
    if (auto ok = add_source_stream(out, reader, stream, request); !ok) return ok;
  }
  // Les métadonnées globales de la source, sans sa durée devenue fausse.
  av_dict_copy(&out.context().metadata, in.metadata, 0);
  av_dict_set(&out.context().metadata, "duration", nullptr, 0);
  if (request.copy.chapters) return copy_chapters(out.context(), in, shift_us, reader.window_us);
  return {};
}

}  // namespace

std::expected<void, Refusal> remux(const RemuxRequest& request) {
  auto out = MuxOutput::create(request.output, request.container, true);
  if (!out) return std::unexpected(out.error());

  Reader video;
  auto video_input = open_input(request.video.string());
  if (!video_input) return std::unexpected(video_input.error());
  video.input = std::move(*video_input);
  if (video.input->nb_streams != 1) {
    return refuse(RefusalCode::MuxFailed,
                  std::format("{} : un seul flux vidéo attendu", request.video.string()));
  }
  const AVStream& encoded = *video.input->streams[0];
  auto stream = out->add_stream(*encoded.codecpar, encoded.time_base, "vidéo exportée");
  if (!stream) return std::unexpected(stream.error());
  if (auto ok = set_stream_hdr10(*(*stream)->codecpar, request.mastering, request.light); !ok) return ok;
  (*stream)->disposition = AV_DISPOSITION_DEFAULT;
  video.out_index = {(*stream)->index};
  video.offset = {0};

  Reader source;
  source.done = request.source.empty();
  if (!source.done) {
    if (auto ok = open_source(*out, source, request); !ok) return ok;
  }
  for (const auto& [key, value] : request.tags)
    av_dict_set(&out->context().metadata, key.c_str(), value.c_str(), 0);
  if (!video.packet || !source.packet)
    return refuse(RefusalCode::MuxFailed, "av_packet_alloc : mémoire insuffisante");
  if (auto ok = out->write_header(); !ok) return ok;

  const std::string video_path = request.video.string();
  const std::string source_path = request.source.string();
  while (true) {
    if (request.stop.stop_requested())
      return refuse(RefusalCode::Cancelled, "export annulé pendant le mux final");
    if (auto ok = video.advance(video_path); !ok) return ok;
    if (auto ok = source.advance(source_path); !ok) return ok;
    if (!video.pending && !source.pending) break;
    Reader& next = !source.pending || (video.pending && video.key() <= source.key()) ? video : source;
    AVPacket& packet = *next.packet;
    const AVRational tb = next.input->streams[packet.stream_index]->time_base;
    packet.stream_index = next.out_index[static_cast<std::size_t>(packet.stream_index)];
    packet.pos = -1;
    next.pending = false;
    if (auto ok = out->write(packet, tb); !ok) return ok;
  }
  return out->finish();
}

std::expected<void, Refusal> check_copy(const RemuxRequest& request) {
  auto out = MuxOutput::create(request.output, request.container);
  if (!out) return std::unexpected(out.error());
  Reader source;
  return open_source(*out, source, request);
}

}  // namespace ajp::exporter
