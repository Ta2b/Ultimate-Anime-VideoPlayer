#include "app/play/pipeline.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>
#include <thread>
#include <utility>

#include "audio/devices.h"
#include "audio/output.h"
#include "media/demuxer.h"
#include "media/video_decoder.h"

namespace ajp::app::play {
namespace {

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
  media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), config.languages);
  const auto kind_of = [&](int index) {
    const media::TrackInfo* track = demuxer->track(index);
    return track != nullptr ? std::optional(track->kind) : std::nullopt;
  };
  if (config.audio_track && kind_of(*config.audio_track) == media::TrackKind::Audio)
    choice.audio = config.audio_track;
  if (config.subtitle_track) {
    const std::optional<int> wanted = *config.subtitle_track;
    if (!wanted || kind_of(*wanted) == media::TrackKind::Subtitle || *wanted >= kExternalSubtitleBase)
      choice.subtitle = wanted;
  }
  if (!choice.video) {
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", config.file.string())});
  }
  p->video_track_ = *demuxer->track(*choice.video);
  p->duration_us_ = demuxer->duration_us();
  for (const media::Chapter& chapter : demuxer->chapters())
    p->chapters_.push_back({.start_us = chapter.start_us, .title = chapter.title});
  std::ranges::sort(p->chapters_, {}, &ChapterMark::start_us);
  const std::vector<subs::AttachFont> fonts = attached_fonts(demuxer->tracks());
  std::vector<media::PacketRoute> routes;

  // Toutes les pistes audio et de sous-titres lisibles sont routées : la barre de lecture en
  // change pendant la lecture (les ponts ne gardent que la piste choisie).
  if (config.audio && choice.audio) {
    // Une piste audio illisible, ou une sortie qui ne s'ouvre pas, ne ferme plus la lecture : la
    // vidéo joue sans son, sur l'horloge hôte, et le refus paraît à l'OSD.
    std::optional<Refusal> audio_failure;
    for (const media::TrackInfo& track : demuxer->tracks()) {
      if (track.kind != media::TrackKind::Audio) continue;
      auto info = audio_codec_info(track);
      if (!info) {
        if (track.index == *choice.audio) audio_failure = std::move(info.error());
        continue;
      }
      p->audio_codecs_.emplace_back(track.index, std::move(*info));
      p->menu_.audio.push_back({.index = track.index, .label = track_label(track)});
    }
    const media::TrackInfo& track = *demuxer->track(*choice.audio);
    if (!audio_failure && p->audio_codec(track.index) == nullptr) {
      audio_failure =
          Refusal{RefusalCode::TrackUnavailable, std::format("piste audio {} illisible", track.index)};
    }
    if (!audio_failure) {
      // La sortie prend le format de la piste, plafonné : au-delà de ce que SDL ouvre, le
      // périphérique choisit (0) et le décodeur remixe vers lui.
      const bool rate_ok = track.sample_rate >= audio::limits::kMinSampleRate &&
                           track.sample_rate <= audio::limits::kMaxSampleRate;
      // Un appareil stéréo reçoit de la stéréo (le mixeur s'occupe du surround : casque, enceintes) ; un appareil
      // multicanal garde les canaux de la piste, SDL les adapte.
      const std::optional<audio::AudioDevice> device =
          audio::resolve_device(audio::playback_devices(), config.audio_device);
      const bool stereo_device = !device || device->channels <= 2;
      const int sink_channels = stereo_device ? std::clamp(track.channels, 0, 2)
                                              : std::clamp(track.channels, 0, audio::limits::kMaxChannels);
      auto output = audio::AudioOutput::open_sdl(
          {.sample_rate = rate_ok ? track.sample_rate : 0, .channels = sink_channels, .device = config.audio_device});
      if (!output) {
        audio_failure = std::move(output.error());
      } else {
        audio::AudioThreadOptions options{
            .credit = [inbox = &p->source_inbox_](Credit credit) { inbox->try_push(credit); },
            .refused = [mail = &p->mail_](const Refusal& refusal) { mail->report("audio", refusal, false); }};
        options.player.mix = config.mix;
        auto thread = audio::AudioThread::start(std::move(*output), std::move(options));
        if (!thread) {
          audio_failure = std::move(thread.error());
        } else {
          p->audio_ = std::move(*thread);
        }
      }
    }
    if (audio_failure) {
      p->audio_codecs_.clear();
      p->menu_.audio.clear();
      p->mail_.report("audio", *audio_failure, false);
    } else {
      p->audio_->post(audio::AudioTrack{.codec = *p->audio_codec(track.index)});
      p->audio_bridge_ = std::make_unique<AudioBridge>(*p->audio_, track.index);
      p->lead_ = std::make_shared<LeadClock>(p->audio_->clock(), p->host_clock_);
      p->clock_ = p->lead_.get();
      p->mail_.attach_audio(p->audio_.get());
      for (const auto& [index, codec] : p->audio_codecs_)
        routes.push_back({.track = index, .sink = p->audio_bridge_.get()});
      p->audio_track_ = track;
      p->menu_.audio_selected = track.index;
    }
  }

  for (const media::TrackInfo& track : demuxer->tracks()) {
    if (track.kind == media::TrackKind::Audio || track.kind == media::TrackKind::Subtitle)
      p->tracks_.push_back(track);
  }

  if (config.subtitles) {
    for (const media::TrackInfo& track : demuxer->tracks()) {
      if (track.kind != media::TrackKind::Subtitle || !subtitle_track(track, subs_track_id(track.index)))
        continue;
      p->menu_.subtitles.push_back({.index = track.index, .label = track_label(track)});
    }
    // Les fichiers à côté de la vidéo : après les pistes du conteneur, choisis d'office seulement
    // quand le conteneur n'en a aucune et qu'aucun choix n'a été fait.
    p->external_ = find_external_subtitles(config.file);
    for (std::size_t k = 0; k < p->external_.size(); ++k) {
      const int index = kExternalSubtitleBase + static_cast<int>(k);
      p->menu_.subtitles.push_back({.index = index, .label = p->external_[k].label});
      p->tracks_.push_back({.index = index,
                            .kind = media::TrackKind::Subtitle,
                            .codec = "externe",
                            .title = p->external_[k].path.filename().string()});
    }
    if (!choice.subtitle && !config.subtitle_track && !p->external_.empty() &&
        std::ranges::none_of(demuxer->tracks(), [](const media::TrackInfo& t) {
          return t.kind == media::TrackKind::Subtitle;
        })) {
      choice.subtitle = kExternalSubtitleBase;
    }
    if (choice.subtitle && !std::ranges::contains(p->menu_.subtitles, *choice.subtitle, &TrackEntry::index)) {
      auto refused = subtitle_track(*demuxer->track(*choice.subtitle), 0);
      p->mail_.report("subs", refused.error(), false);
      choice.subtitle.reset();
    }
  }
  if (!p->menu_.subtitles.empty()) {
    p->subs_inbox_ = std::make_shared<subs::SubsInbox>(subs::kSubsInboxBounds);
    auto thread = subs::SubsThread::start(config.subs, p->mail_, p->subs_inbox_);
    if (!thread) return std::unexpected(std::move(thread.error()));
    p->subs_thread_ = std::move(*thread);
    for (const subs::AttachFont& font : fonts) {
      // La boîte tient 64 polices à la fois : au-delà, on laisse au thread subs le temps d'en
      // prendre (une seconde au plus) avant d'en écarter une.
      bool pushed = p->subs_inbox_->try_push(subs::AttachFont{font});
      for (int wait = 0; !pushed && wait < 1000; ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        pushed = p->subs_inbox_->try_push(subs::AttachFont{font});
      }
      if (!pushed) {
        p->mail_.report("subs", {RefusalCode::SubtitleBacklog, std::format("police {} écartée", font.name)},
                        false);
      }
    }
    std::optional<int> chosen;
    if (choice.subtitle && std::ranges::contains(p->menu_.subtitles, *choice.subtitle, &TrackEntry::index)) {
      chosen = choice.subtitle;
      if (auto select = p->subtitle_select(*chosen)) {
        p->subs_inbox_->try_push(std::move(*select));
        p->subtitle_track_ = *p->track_info(*chosen);
      } else {
        chosen.reset();
      }
    }
    p->menu_.subtitle_selected = chosen;
    p->subs_bridge_ = std::make_unique<SubsBridge>(p->subs_inbox_, chosen);
    for (const TrackEntry& entry : p->menu_.subtitles) {
      if (entry.index < kExternalSubtitleBase)
        routes.push_back({.track = entry.index, .sink = p->subs_bridge_.get()});
    }
  }

  auto media = media::open_source(std::move(*demuxer), *choice.video, config.decoder);
  if (!media) return std::unexpected(std::move(media.error()));
  p->decoder_api_ = std::string(media::name(media->decoder.api()));

  auto video = make_video(VideoStageInit{.track = &p->video_track_,
                                         .ring = &p->mail_,
                                         .credit = [inbox = &p->source_inbox_] { inbox->try_push(Credit{}); },
                                         .report = p->mail_.reporter("gpu"),
                                         .clock = p->clock_});
  if (!video) return std::unexpected(std::move(video.error()));
  p->video_ = std::move(*video);
  p->mail_.attach_video(p->video_.get());
  p->stages_ = std::make_unique<ThreadStages>(StageLinks{.source = &p->source_inbox_,
                                                         .video = p->video_.get(),
                                                         .audio = p->audio_.get(),
                                                         .audio_bridge = p->audio_bridge_.get(),
                                                         .subs = p->subs_inbox_,
                                                         .host_clock = p->host_clock_,
                                                         .lead = p->lead_.get()});

  auto source = media::SourceThread::start(
      std::move(*media),
      media::SourceLinks{.frames = &p->video_->frames(), .listener = &p->mail_, .packets = std::move(routes)},
      p->source_inbox_);
  if (!source) return std::unexpected(std::move(source.error()));
  p->source_ = std::move(*source);
  return p;
}

Pipeline::~Pipeline() = default;

const audio::AudioCodecInfo* Pipeline::audio_codec(int track) const {
  const auto found = std::ranges::find(audio_codecs_, track, &std::pair<int, audio::AudioCodecInfo>::first);
  return found != audio_codecs_.end() ? &found->second : nullptr;
}

const media::TrackInfo* Pipeline::track_info(int track) const {
  const auto found = std::ranges::find(tracks_, track, &media::TrackInfo::index);
  return found != tracks_.end() ? &*found : nullptr;
}

std::expected<void, Refusal> Pipeline::select_audio(int track, std::uint32_t from_epoch,
                                                    const std::function<void()>& seek) {
  const audio::AudioCodecInfo* codec = audio_codec(track);
  if (codec == nullptr || !audio_bridge_)
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("piste audio {} absente", track)});
  // Ordre : les paquets de l'époque from_epoch sont de la nouvelle piste ; le seek pose le Flush
  // dans la boîte d'audio, puis AudioTrack le suit (la boîte garde l'ordre) : l'ancien décodeur
  // lit ses derniers paquets, le nouveau ceux de la nouvelle époque.
  audio_bridge_->select(track, from_epoch);
  seek();
  if (!audio_->post(audio::AudioTrack{.codec = *codec})) {
    // Le décodeur n'a pas changé : le pont garde la piste d'avant, comme le menu.
    audio_bridge_->undo_select();
    return std::unexpected(Refusal{RefusalCode::PacketBacklog, "piste audio : boîte audio pleine"});
  }
  menu_.audio_selected = track;
  if (const media::TrackInfo* info = track_info(track)) audio_track_ = *info;
  return {};
}

std::expected<subs::SelectTrack, Refusal> Pipeline::subtitle_select(int index) const {
  const media::TrackInfo* info = track_info(index);
  if (info == nullptr)
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("piste de sous-titres {} absente", index)});
  if (index >= kExternalSubtitleBase) {
    const auto k = static_cast<std::size_t>(index - kExternalSubtitleBase);
    if (k >= external_.size())
      return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "sous-titres externes absents"});
    return subs::SelectTrack{.id = subs_track_id(index),
                             .source = subs::ExternalFile{.path = external_[k].path}};
  }
  return subtitle_track(*info, subs_track_id(index));
}

std::expected<void, Refusal> Pipeline::select_subtitles(std::optional<int> track) {
  if (!subs_bridge_)
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "aucune piste de sous-titres lisible"});
  if (!track) {
    subs_bridge_->select(std::nullopt);
    menu_.subtitle_selected.reset();
    subtitle_track_.reset();
    return {};
  }
  const media::TrackInfo* info = track_info(*track);
  if (info == nullptr || !std::ranges::contains(menu_.subtitles, *track, &TrackEntry::index))
    return std::unexpected(
        Refusal{RefusalCode::TrackUnavailable, std::format("piste de sous-titres {} absente", *track)});
  auto select = subtitle_select(*track);
  if (!select) return std::unexpected(std::move(select.error()));
  if (!subs_inbox_->try_push(std::move(*select)))
    return std::unexpected(Refusal{RefusalCode::SubtitleBacklog, "piste de sous-titres : boîte pleine"});
  subs_bridge_->select(track);
  menu_.subtitle_selected = track;
  subtitle_track_ = *info;
  return {};
}

void Pipeline::pump() {
  if (subs_bridge_ && subs_bridge_->take_refused()) source_inbox_.try_push(Credit{});
}

SessionConfig Pipeline::session_config() const {
  // Matrice de la vidéo pour les couleurs VSFilter : celle que le flux annonce, sinon celle que mpv
  // devine à la taille. La plage vient du flux aussi : en plage pleine, VSFilter ne convertit pas.
  const bool hd = video_track_.width >= 1280 || video_track_.height > 576;
  return SessionConfig{
      .duration_us = duration_us_,
      .chapters = chapters_,
      .subtitles = subtitle_track_.has_value(),
      .video_width = video_track_.width,
      .video_height = video_track_.height,
      .video_sar = video_track_.sample_aspect.num > 0
                       ? static_cast<double>(video_track_.sample_aspect.num) / video_track_.sample_aspect.den
                       : 1.0,
      .video_matrix =
          subs::VideoMatrix{
              .matrix = video_track_.color_matrix.value_or(hd ? ColorMatrix::Bt709 : ColorMatrix::Bt601),
              .range = video_track_.full_range ? ColorRange::Full : ColorRange::Limited},
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
