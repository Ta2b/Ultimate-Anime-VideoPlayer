#include "subs/subtitle_track.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <ass/ass.h>

#include "core/text.h"
#include "core/utf8_path.h"
#include "subs/log.h"
#include "subs/srt.h"
#include "subs/text_encoding.h"
#include "subs/timebase.h"

namespace ajp::subs {
namespace {

std::string lowercase_extension(const std::filesystem::path& path) {
  std::string extension = core::to_utf8(path.extension());
  for (char& c : extension) c = core::ascii_lower(c);
  return extension;
}

// libass lit les polices d'une section [Fonts], quelle que soit sa casse.
bool has_fonts_section(std::string_view script) {
  constexpr std::string_view kSection = "[fonts]";
  return !std::ranges::search(script, kSection, [](char a, char b) { return core::ascii_lower(a) == b; }).empty();
}

// Lit au plus la taille mesurée : un fichier qui grandit pendant la lecture
// n'agrandit pas l'allocation.
std::expected<std::string, Refusal> read_script(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    return refuse(RefusalCode::SubtitleUnreadable,
                  std::format("{} : {}", core::to_utf8(path.filename()), error.message()));
  }
  if (size > kMaxScriptBytes) {
    return refuse(
        RefusalCode::SubtitleUnreadable,
        std::format("{} : {} octets, au-delà de {}", core::to_utf8(path.filename()), size, kMaxScriptBytes));
  }
  std::ifstream file(path, std::ios::binary);
  std::string text(static_cast<std::size_t>(size), '\0');
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!file.is_open() || file.bad()) {
    return refuse(RefusalCode::SubtitleUnreadable,
                  std::format("{} : lecture", core::to_utf8(path.filename())));
  }
  text.resize(static_cast<std::size_t>(file.gcount()));
  // L'indice d'encodage : les suffixes du nom (« ep01.ja » donne « ja », « ep01.zh_TW.forced »
  // donne « zh_TW.forced »), que to_utf8 lit du dernier au premier.
  const std::string stem = core::to_utf8(path.stem());
  const std::size_t dot = stem.find('.');
  return to_utf8(text,
                 dot == std::string::npos ? std::string_view{} : std::string_view(stem).substr(dot + 1));
}

}  // namespace

void SubtitleTrack::Free::operator()(ass_track* track) const {
  ass_free_track(track);
}

SubtitleTrack::SubtitleTrack(std::unique_ptr<ass_track, Free> track, TrackKind kind, bool embeds_fonts)
    : track_(std::move(track)), kind_(kind), embeds_fonts_(embeds_fonts) {}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::create(const AssLibrary& library, TrackKind kind,
                                                            std::string_view header) {
  if (header.empty() || header.size() > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleFormat, std::format("en-tête ASS de {} octets", header.size()));
  }
  std::unique_ptr<ass_track, Free> track(ass_new_track(library.get()));
  if (track == nullptr) return refuse(RefusalCode::LibassFailed, "ass_new_track");
  ass_process_codec_private(track.get(), header.data(), static_cast<int>(header.size()));
  if (track->track_type == ass_track::TRACK_TYPE_UNKNOWN) {
    return refuse(RefusalCode::SubtitleFormat, "en-tête sans ScriptType ni section de styles reconnus");
  }
  return SubtitleTrack(std::move(track), kind, has_fonts_section(header));
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
  return SubtitleTrack(std::move(track), TrackKind::Ass, has_fonts_section(script));
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::from_srt(const AssLibrary& library,
                                                              std::string_view file) {
  if (file.size() > kMaxScriptBytes) {
    return refuse(RefusalCode::SubtitleUnreadable, std::format("fichier SRT de {} octets", file.size()));
  }
  std::vector<std::string> skipped;
  auto cues = parse_srt(file, &skipped);
  if (!cues) return std::unexpected(std::move(cues.error()));
  for (const std::string& what : skipped) make_logger()->warn("{} : cue ignorée", what);
  auto track = text(library);
  if (!track) return track;
  for (SrtCue& cue : *cues) {
    const SubtitlePacket packet{
        .start_us = cue.start_us, .duration_us = cue.end_us - cue.start_us, .data = std::move(cue.text)};
    // Une cue hors bornes (crédits de 30 h, texte de plus de 1 Mo) est écartée comme les cues mal
    // formées, sans faire refuser le fichier (A17-07).
    if (auto added = track->add_packet(packet); !added)
      make_logger()->warn("cue de {} µs écartée : {}", cue.start_us, added.error().detail);
  }
  return track;
}

std::expected<SubtitleTrack, Refusal> SubtitleTrack::load(const AssLibrary& library,
                                                          const std::filesystem::path& path) {
  const std::string extension = lowercase_extension(path);
  if (extension != ".ass" && extension != ".ssa" && extension != ".srt") {
    return refuse(RefusalCode::SubtitleFormat,
                  std::format("{} : extension inconnue", core::to_utf8(path.filename())));
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
  if (bytes_ >= kMaxTrackBytes || track_->n_events >= kMaxTrackEvents) {
    if (full_) return {};
    full_ = true;
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("piste de sous-titres pleine : {} événements, {} octets ; la suite est écartée",
                              track_->n_events, bytes_));
  }
  const std::int64_t start_ms = to_ms(packet.start_us);
  const std::int64_t duration_ms = to_ms(packet.start_us + packet.duration_us) - start_ms;

  if (kind_ == TrackKind::Ass) {
    const int before = track_->n_events;
    ass_process_chunk(track_.get(), packet.data.data(), static_cast<int>(packet.data.size()), start_ms,
                      duration_ms);
    if (track_->n_events > before) bytes_ += packet.data.size();  // libass écarte un ReadOrder revu
    return {};
  }
  // Texte : un ReadOrder par paquet distinct, et la ligne Matroska d'un événement Default.
  if (!seen_text_.emplace(packet.start_us, packet.duration_us, packet.data).second) return {};
  bytes_ += packet.data.size();
  const std::string line =
      std::format("{},0,Default,,0,0,0,,{}", seen_text_.size() - 1, srt_text_to_ass(packet.data));
  ass_process_chunk(track_.get(), line.data(), static_cast<int>(line.size()), start_ms, duration_ms);
  return {};
}

int SubtitleTrack::event_count() const {
  return track_->n_events;
}

ScriptMatrix SubtitleTrack::script_matrix() const {
  switch (track_->YCbCrMatrix) {
    case YCBCR_DEFAULT:
      return ScriptMatrix::Absent;
    case YCBCR_NONE:
      return ScriptMatrix::None;
    case YCBCR_BT601_TV:
      return ScriptMatrix::Bt601Tv;
    default:
      return ScriptMatrix::Other;
  }
}

}  // namespace ajp::subs
