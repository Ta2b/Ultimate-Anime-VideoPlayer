#include "app/play/pipeline.h"

#include <format>
#include <utility>

#include "audio/output.h"
#include "media/demuxer.h"
#include "media/video_decoder.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kSubtitleTrackId = 1;

std::string track_text(std::string_view kind, const media::TrackInfo& track) {
  std::string text = std::format("{} {}", kind, track.codec);
  if (!track.language.empty()) text += std::format(" {}", track.language);
  if (!track.title.empty()) text += std::format(" « {} »", track.title);
  return text;
}

}  // namespace

std::expected<std::unique_ptr<Pipeline>, Refusal> Pipeline::open(const PipelineConfig& config,
                                                                 const VideoStageFactory& make_video) {
  std::unique_ptr<Pipeline> p(new Pipeline());
  auto demuxer = media::Demuxer::open(config.file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), config.languages);
  if (!choice.video) {
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", config.file.string())});
  }
  p->video_track_ = *demuxer->track(*choice.video);
  p->duration_us_ = demuxer->duration_us();
  const std::vector<subs::AttachFont> fonts = attached_fonts(demuxer->tracks());
  std::vector<media::PacketRoute> routes;

  if (config.audio && choice.audio) {
    const media::TrackInfo& track = *demuxer->track(*choice.audio);
    auto info = audio_codec_info(track);
    if (!info) return std::unexpected(std::move(info.error()));
    auto output =
        audio::AudioOutput::open_sdl({.sample_rate = track.sample_rate, .channels = track.channels});
    if (!output) return std::unexpected(std::move(output.error()));
    audio::AudioThreadOptions options{
        .credit = [inbox = &p->source_inbox_](Credit credit) { inbox->try_push(credit); },
        .refused = [mail = &p->mail_](const Refusal& refusal) { mail->report("audio", refusal, false); }};
    auto thread = audio::AudioThread::start(std::move(*output), std::move(options));
    if (!thread) return std::unexpected(std::move(thread.error()));
    p->audio_ = std::move(*thread);
    p->audio_->post(audio::AudioTrack{.codec = std::move(*info)});
    p->audio_bridge_ = std::make_unique<AudioBridge>(*p->audio_);
    p->audio_clock_ = p->audio_->clock();
    p->clock_ = p->audio_clock_.get();
    p->mail_.attach_audio(p->audio_.get());
    routes.push_back({.track = track.index, .sink = p->audio_bridge_.get()});
    p->audio_track_ = track;
  }

  if (config.subtitles && choice.subtitle) {
    const media::TrackInfo& track = *demuxer->track(*choice.subtitle);
    auto select = subtitle_track(track, kSubtitleTrackId);
    if (select) {
      p->subs_inbox_ = std::make_shared<subs::SubsInbox>(subs::kSubsInboxBounds);
      auto thread = subs::SubsThread::start(config.subs, p->mail_, p->subs_inbox_);
      if (!thread) return std::unexpected(std::move(thread.error()));
      p->subs_thread_ = std::move(*thread);
      for (const subs::AttachFont& font : fonts) {
        if (!p->subs_inbox_->try_push(subs::AttachFont{font})) {
          p->mail_.report("subs", {RefusalCode::SubtitleBacklog, std::format("police {} écartée", font.name)},
                          false);
        }
      }
      p->subs_inbox_->try_push(std::move(*select));
      p->subs_bridge_ = std::make_unique<SubsBridge>(p->subs_inbox_, kSubtitleTrackId);
      routes.push_back({.track = track.index, .sink = p->subs_bridge_.get()});
      p->subtitle_track_ = track;
    } else {
      p->mail_.report("subs", select.error(), false);
    }
  }

  auto media = media::open_source(std::move(*demuxer), *choice.video, config.decoder);
  if (!media) return std::unexpected(std::move(media.error()));
  p->decoder_api_ = std::string(media::name(media->decoder.api()));

  auto video = make_video(VideoStageInit{.track = &p->video_track_,
                                         .ring = &p->mail_,
                                         .credit = [inbox = &p->source_inbox_] { inbox->try_push(Credit{}); },
                                         .report = p->mail_.reporter("gpu")});
  if (!video) return std::unexpected(std::move(video.error()));
  p->video_ = std::move(*video);
  p->stages_ =
      std::make_unique<ThreadStages>(StageLinks{.source = &p->source_inbox_,
                                                .video = p->video_.get(),
                                                .audio = p->audio_.get(),
                                                .audio_bridge = p->audio_bridge_.get(),
                                                .subs = p->subs_inbox_,
                                                .host_clock = p->audio_ ? nullptr : &p->host_clock_});

  auto source = media::SourceThread::start(
      std::move(*media),
      media::SourceLinks{.frames = &p->video_->frames(), .listener = &p->mail_, .packets = std::move(routes)},
      p->source_inbox_);
  if (!source) return std::unexpected(std::move(source.error()));
  p->source_ = std::move(*source);
  return p;
}

Pipeline::~Pipeline() = default;

void Pipeline::pump() {
  if (subs_bridge_ && subs_bridge_->take_refused()) source_inbox_.try_push(Credit{});
}

SessionConfig Pipeline::session_config() const {
  // Matrice de la vidéo pour les couleurs VSFilter : celle que mpv devine à la taille.
  const bool hd = video_track_.width >= 1280 || video_track_.height > 576;
  return SessionConfig{
      .duration_us = duration_us_,
      .subtitles = subtitle_track_.has_value(),
      .video_width = video_track_.width,
      .video_height = video_track_.height,
      .video_matrix = subs::VideoMatrix{.matrix = hd ? ColorMatrix::Bt709 : ColorMatrix::Bt601,
                                        .range = ColorRange::Limited},
  };
}

std::vector<std::string> Pipeline::describe() const {
  std::vector<std::string> lines;
  lines.push_back(std::format("{} {}x{} ({})", track_text("vidéo", video_track_), video_track_.width,
                              video_track_.height, decoder_api_));
  lines.push_back(audio_track_ ? track_text("audio", *audio_track_) : "audio : aucun, horloge hôte");
  lines.push_back(subtitle_track_ ? track_text("sous-titres", *subtitle_track_) : "sous-titres : aucun");
  return lines;
}

}  // namespace ajp::app::play
