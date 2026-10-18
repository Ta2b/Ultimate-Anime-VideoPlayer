#include "media/source.h"

#include <algorithm>
#include <exception>
#include <format>
#include <utility>

#include "media/source_loop.h"

namespace ajp::media {
namespace {

std::unexpected<Refusal> refuse_link(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::TrackUnavailable, std::move(detail)});
}

std::expected<void, Refusal> check_links(const SourceMedia& media, const SourceLinks& links) {
  if (links.frames == nullptr || links.listener == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, "SourceLinks : frames et listener sont exigés"});
  }
  for (const PacketRoute& route : links.packets) {
    const TrackInfo* track = media.demuxer.track(route.track);
    if (track == nullptr) return refuse_link(std::format("piste {} absente", route.track));
    if (track->kind != TrackKind::Audio && track->kind != TrackKind::Subtitle) {
      return refuse_link(std::format("piste {} : ni audio ni sous-titres", route.track));
    }
    if (route.sink == nullptr) return refuse_link(std::format("piste {} : puits nul", route.track));
    if (std::ranges::count(links.packets, route.track, &PacketRoute::track) > 1) {
      return refuse_link(std::format("piste {} routée deux fois", route.track));
    }
  }
  return {};
}

}  // namespace

std::expected<SourceMedia, Refusal> open_source(Demuxer demuxer, int video_track,
                                                const DecoderOptions& options) {
  const TrackInfo* track = demuxer.track(video_track);
  if (track == nullptr || track->kind != TrackKind::Video) {
    return refuse_link(std::format("piste {} : pas une piste vidéo", video_track));
  }
  auto decoder = VideoDecoder::open(*track, options);
  if (!decoder) return std::unexpected(decoder.error());
  return SourceMedia{
      .demuxer = std::move(demuxer), .decoder = std::move(*decoder), .video_track = video_track};
}

std::expected<std::unique_ptr<SourceThread>, Refusal> SourceThread::start(SourceMedia media,
                                                                          SourceLinks links,
                                                                          SourceInbox& inbox) {
  if (auto valid = check_links(media, links); !valid) return std::unexpected(valid.error());
  std::vector<int> wanted{media.video_track};
  for (const PacketRoute& route : links.packets) wanted.push_back(route.track);
  if (auto kept = media.demuxer.keep_only(wanted); !kept) return std::unexpected(kept.error());

  std::unique_ptr<SourceThread> source(new SourceThread());
  SourceListener* listener = links.listener;
  source->thread_ = std::jthread([loop = SourceLoop(std::move(media), std::move(links), inbox),
                                  listener](const std::stop_token& stop) mutable {
    // Aucune exception ne sort de la boucle : elle devient un refus signalé.
    try {
      loop.run(stop);
    } catch (const std::exception& error) {
      listener->failed(Refusal{RefusalCode::DecodeFailed, std::format("thread source : {}", error.what())});
    }
  });
  return source;
}

}  // namespace ajp::media
