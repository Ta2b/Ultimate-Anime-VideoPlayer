#include "subs/subtitle_track.h"

#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>

#include <ass/ass.h>

#include "subs/srt.h"
#include "subs/timebase.h"

namespace ajp::subs {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string lowercase_extension(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  for (char& c : extension) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return extension;
}

std::expected<std::string, Refusal> read_script(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    return refuse(RefusalCode::SubtitleUnreadable, std::format("{} : {}", path.string(), error.message()));
  }
  if (size > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleUnreadable,
                  std::format("{} : {} octets, au-delà de {}", path.string(), size, kMaxScriptBytes));
  }
  std::ifstream file(path, std::ios::binary);
  std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file.is_open() || file.bad()) {
    return refuse(RefusalCode::SubtitleUnreadable, std::format("{} : lecture", path.string()));
  }
  return text;
}

}  // namespace

void SubtitleTrack::Free::operator()(ass_track* track) const {
  ass_free_track(track);
}

SubtitleTrack::SubtitleTrack(std::unique_ptr<ass_track, Free> track, TrackKind kind)
    : track_(std::move(track)), kind_(kind) {}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::create(const AssLibrary& library, TrackKind kind,
                                                            std::string_view header) {
  if (header.empty() || header.size() > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleFormat, std::format("en-tête ASS de {} octets", header.size()));
  }
  std::unique_ptr<ass_track, Free> track(ass_new_track(library.get()));
  if (track == nullptr) return refuse(RefusalCode::LibassFailed, "ass_new_track");
  ass_process_codec_private(track.get(), header.data(), static_cast<int>(header.size()));
  if (track->track_type == ass_track::TRACK_TYPE_UNKNOWN) {
    return refuse(RefusalCode::SubtitleFormat, "en-tête sans section [Script Info] reconnue");
  }
  return SubtitleTrack(std::move(track), kind);
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::from_header(const AssLibrary& library,
                                                                 std::string_view header) {
  return create(library, TrackKind::Ass, header);
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::text(const AssLibrary& library) {
  return create(library, TrackKind::Text, text_track_header());
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::from_ass(const AssLibrary& library,
                                                              std::string_view script) {
  if (script.size() > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleUnreadable, std::format("script ASS de {} octets", script.size()));
  }
  std::string copy(script);  // ass_read_memory prend un char* modifiable
  std::unique_ptr<ass_track, Free> track(ass_read_memory(library.get(), copy.data(), copy.size(), nullptr));
  if (track == nullptr) return refuse(RefusalCode::SubtitleFormat, "script ASS sans section [Script Info]");
  return SubtitleTrack(std::move(track), TrackKind::Ass);
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::from_srt(const AssLibrary& library,
                                                              std::string_view file) {
  if (file.size() > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleUnreadable, std::format("fichier SRT de {} octets", file.size()));
  }
  auto cues = parse_srt(file);
  if (!cues) return std::unexpected(std::move(cues.error()));
  auto track = text(library);
  if (!track) return track;
  for (SrtCue& cue : *cues) {
    const SubtitlePacket packet{
        .start_us = cue.start_us, .duration_us = cue.end_us - cue.start_us, .data = std::move(cue.text)};
    if (auto added = track->add_packet(packet); !added) return std::unexpected(std::move(added.error()));
  }
  return track;
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::load(const AssLibrary& library,
                                                          const std::filesystem::path& path) {
  const std::string extension = lowercase_extension(path);
  if (extension != ".ass" && extension != ".ssa" && extension != ".srt") {
    return refuse(RefusalCode::SubtitleFormat, std::format("{} : extension inconnue", path.string()));
  }
  auto script = read_script(path);
  if (!script) return std::unexpected(std::move(script.error()));
  return extension == ".srt" ? from_srt(library, *script) : from_ass(library, *script);
}

std::expected<void, Refusal> SubtitleTrack::add_packet(const SubtitlePacket& packet) {
  if (!timestamp_in_range(packet.start_us) || packet.duration_us < 0 ||
      packet.duration_us > kMaxEventDurationUs || packet.data.size() > kMaxPacketBytes) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("paquet de sous-titres : début {} µs, durée {} µs, {} octets", packet.start_us,
                              packet.duration_us, packet.data.size()));
  }
  const std::int64_t start_ms = to_ms(packet.start_us);
  const std::int64_t duration_ms = to_ms(packet.start_us + packet.duration_us) - start_ms;

  if (kind_ == TrackKind::Ass) {
    ass_process_chunk(track_.get(), packet.data.data(), static_cast<int>(packet.data.size()), start_ms,
                      duration_ms);
    return {};
  }
  // Texte : un ReadOrder par paquet distinct, et la ligne Matroska d'un événement Default.
  if (seen_text_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return refuse(RefusalCode::FieldOutOfRange, "piste texte : trop d'événements");
  }
  if (!seen_text_.emplace(packet.start_us, packet.duration_us, packet.data).second) return {};
  const std::string line =
      std::format("{},0,Default,,0,0,0,,{}", seen_text_.size() - 1, srt_text_to_ass(packet.data));
  ass_process_chunk(track_.get(), line.data(), static_cast<int>(line.size()), start_ms, duration_ms);
  return {};
}

int SubtitleTrack::event_count() const {
  return track_->n_events;
}

}  // namespace ajp::subs
