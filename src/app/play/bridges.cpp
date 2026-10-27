#include "app/play/bridges.h"

#include <format>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/codec_desc.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
}

#include "core/frame.h"

namespace ajp::app::play {

std::expected<audio::AudioCodecInfo, Refusal> audio_codec_info(const media::TrackInfo& track) {
  if (track.kind != media::TrackKind::Audio || !track.parameters) {
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("piste {} : pas audio", track.index)});
  }
  const AVCodecParameters& par = track.parameters->av();
  const AVCodecDescriptor* descriptor = avcodec_descriptor_get(par.codec_id);
  if (descriptor == nullptr) {
    return std::unexpected(Refusal{
        RefusalCode::AudioCodecUnavailable,
        std::format("piste {} : codec {} inconnu de FFmpeg", track.index, static_cast<int>(par.codec_id))});
  }
  audio::AudioCodecInfo info{
      .codec = descriptor->name,
      .sample_rate = par.sample_rate,
      .channels = par.ch_layout.nb_channels,
      .channel_mask = par.ch_layout.order == AV_CHANNEL_ORDER_NATIVE ? par.ch_layout.u.mask : 0,
      .block_align = par.block_align,
      .bits_per_coded_sample = par.bits_per_coded_sample,
      .bit_rate = par.bit_rate};
  const auto extra = track.parameters->extradata();
  info.extradata.resize(extra.size());
  for (std::size_t i = 0; i < extra.size(); ++i) info.extradata[i] = static_cast<std::uint8_t>(extra[i]);
  return info;
}

void AudioBridge::expect_seek(std::uint32_t epoch, std::optional<std::int64_t> target_us) {
  const std::scoped_lock lock(mutex_);
  trim_epoch_ = epoch;
  trim_us_ = target_us;
}

void AudioBridge::select(int track, std::uint32_t from_epoch) {
  const std::scoped_lock lock(mutex_);
  undo_track_ = track_;
  undo_previous_ = previous_;
  undo_switch_epoch_ = switch_epoch_;
  previous_ = track_;
  track_ = track;
  switch_epoch_ = from_epoch;
}

void AudioBridge::undo_select() {
  const std::scoped_lock lock(mutex_);
  track_ = undo_track_;
  previous_ = undo_previous_;
  switch_epoch_ = undo_switch_epoch_;
}

int AudioBridge::track_for(std::uint32_t epoch) const {
  const std::scoped_lock lock(mutex_);
  return epoch >= switch_epoch_ ? track_ : previous_;
}

bool AudioBridge::try_push(media::Packet&& packet) {
  if (packet.track != track_for(packet.epoch)) return true;  // une autre piste : écartée
  const auto bytes = packet.data();
  audio::AudioPacket out{.pts_us = known_pts(packet.pts_us), .epoch = packet.epoch};
  out.data.resize(bytes.size());
  for (std::size_t i = 0; i < bytes.size(); ++i) out.data[i] = static_cast<std::uint8_t>(bytes[i]);
  if (packet.av) {
    for (int i = 0; i < packet.av->side_data_elems; ++i) {
      const AVPacketSideData& side = packet.av->side_data[i];
      if (side.type != AV_PKT_DATA_SKIP_SAMPLES && side.type != AV_PKT_DATA_NEW_EXTRADATA &&
          side.type != AV_PKT_DATA_PARAM_CHANGE) {
        continue;
      }
      out.side_data.push_back({.type = static_cast<int>(side.type),
                               .bytes = std::vector<std::uint8_t>(side.data, side.data + side.size)});
    }
  }
  {
    const std::scoped_lock lock(mutex_);
    if (trim_us_ && packet.epoch == trim_epoch_) out.discard_before_us = trim_us_;
  }
  return audio_.push_packet(std::move(out));
}

std::expected<subs::SelectTrack, Refusal> subtitle_track(const media::TrackInfo& track, std::uint32_t id) {
  if (track.kind != media::TrackKind::Subtitle || !track.parameters) {
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("piste {} : pas de sous-titres", track.index)});
  }
  if (track.codec == "ass" || track.codec == "ssa") {
    const auto extra = track.parameters->extradata();
    std::string header(extra.size(), '\0');
    for (std::size_t i = 0; i < extra.size(); ++i) header[i] = static_cast<char>(extra[i]);
    return subs::SelectTrack{.id = id, .source = subs::AssStream{.header = std::move(header)}};
  }
  if (track.codec == "subrip" || track.codec == "text" || track.codec == "srt") {
    return subs::SelectTrack{.id = id, .source = subs::TextStream{}};
  }
  return std::unexpected(
      Refusal{RefusalCode::TrackUnavailable,
              std::format("piste {} : sous-titres {} non pris en charge", track.index, track.codec)});
}

std::vector<subs::AttachFont> attached_fonts(std::span<const media::TrackInfo> tracks) {
  std::vector<subs::AttachFont> fonts;
  for (const media::TrackInfo& track : tracks) {
    if (!media::is_font(track) || !track.parameters) continue;
    const auto extra = track.parameters->extradata();
    fonts.push_back(subs::AttachFont{.name = track.filename, .data = {extra.begin(), extra.end()}});
  }
  return fonts;
}

bool SubsBridge::try_push(media::Packet&& packet) {
  if (packet.track != track_.load()) return true;  // piste non choisie : écartée
  if (packet.pts_us == kNoPts) return true;        // sans instant, rien à afficher : écarté
  const auto bytes = packet.data();
  subs::TrackPacket out{.track = subs_track_id(packet.track),
                        .packet = {.start_us = packet.pts_us, .duration_us = packet.duration_us, .data = {}}};
  out.packet.data.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (inbox_->try_push(std::move(out))) return true;
  refused_.store(true);
  return false;
}

}  // namespace ajp::app::play
