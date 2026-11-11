#include "app/play/pipeline.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>
#include <thread>
#include <tuple>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/subtitle_lead.h"
#include "app/play/visible_error.h"
#include "audio/devices.h"
#include "audio/output.h"
#include "core/log.h"
#include "core/pacing.h"
#include "core/text.h"
#include "kernels/spec/color.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "media/video_decoder.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

std::string track_text(std::string_view kind, const media::TrackInfo& track) {
  std::string text = std::format("{} {}", i18n::tr(std::string(kind).c_str()), track.codec);
  if (!track.language.empty()) text += std::format(" {}", track.language);
  if (!track.title.empty()) text += std::format(" \"{}\"", track.title);
  return text;
}

// Un choix de sous-titres gardé d'une autre ouverture (réouverture, épisode précédent, fichier glissé
// d'ailleurs) : pris seulement s'il existe ici. Un index externe peut ne plus désigner aucun fichier
// trouvé à côté de la vidéo : il est oublié, jamais déréférencé (A13-01). nullopt (aucun) existe toujours.
bool subtitle_exists(std::optional<int> wanted, const media::Demuxer& demuxer, std::size_t externals) {
  if (!wanted) return true;
  if (*wanted >= kExternalSubtitleBase) {
    if (static_cast<std::size_t>(*wanted - kExternalSubtitleBase) < externals) return true;
    const auto log = log_channel("subs");
    log->info("sous-titres externes {} absents de cette ouverture : oubliés", *wanted);
    return false;
  }
  const media::TrackInfo* track = demuxer.track(*wanted);
  return track != nullptr && track->kind == media::TrackKind::Subtitle;
}

// La raison d'une piste choisie qui ne se lit pas, à l'OSD ; un index externe n'a pas de piste.
void report_unreadable(const media::Demuxer& demuxer, int index, MainMail& mail) {
  const media::TrackInfo* track = demuxer.track(index);
  if (track == nullptr) return;
  if (auto refused = subtitle_track(*track, 0); !refused) mail.report("subs", refused.error(), false);
}

// A17-02 : des pistes de sous-titres image (PGS, VobSub) et rien d'autre de choisi : dit une fois, au lieu
// d'une lecture sans sous-titres et sans un mot.
void note_image_subtitles(const media::Demuxer& demuxer, bool chosen, MainMail& mail) {
  if (chosen) return;
  std::string codecs;
  for (const media::TrackInfo& track : demuxer.tracks()) {
    if (track.kind != media::TrackKind::Subtitle || !media::subtitle_image(track.codec)) continue;
    if (codecs.find(track.codec) == std::string::npos) codecs += (codecs.empty() ? "" : ", ") + track.codec;
  }
  if (codecs.empty()) return;
  mail.report("subs",
              Refusal{.code = RefusalCode::TrackUnavailable,
                      .detail = std::format("sous-titres image ({}) non pris en charge", codecs),
                      .key = std::string(kImageSubtitlesKey)},
              false);
}

// Une piste de sous-titres que le lecteur sait lire. A17-02 : une piste image (PGS) que rien ne lit n'empêche
// plus le fichier d'à côté.
bool readable_subtitles(const media::Demuxer& demuxer) {
  return std::ranges::any_of(demuxer.tracks(), [](const media::TrackInfo& t) {
    return t.kind == media::TrackKind::Subtitle && media::subtitle_readable(t.codec);
  });
}

// A18-15 : une piste mov_text (MP4) arrive avec sa longueur en tête, que le pont retire.
void mark_mov_text(const media::Demuxer& demuxer, int index, SubsBridge& bridge, std::vector<int>& mov_text) {
  if (const media::TrackInfo* track = demuxer.track(index); track && track->codec == "mov_text") {
    bridge.mov_text(index);
    mov_text.push_back(index);
  }
}

// La piste que relit SubtitleLead : une piste du conteneur, pas un fichier externe.
std::optional<int> container_track(std::optional<int> track) {
  if (track && *track >= kExternalSubtitleBase) return std::nullopt;
  return track;
}

// A16-05 : la correction de dérive de l'audio couvre tout calage de vitesse, avec de la marge.
static_assert(audio::limits::kMaxSpeedChangeCovered == core::pacing::kMaxSpeedChange);

// A16-07 : la sortie prend le format de l'appareil quand il est connu, et libswresample fait la seule
// conversion de chaque piste vers lui : une piste 5.1 choisie après une 2.0 garde ses canaux sur un
// ampli 5.1, une piste 44,1 kHz n'est pas rééchantillonnée deux fois. Un appareil stéréo reçoit de la stéréo
// (libswresample mixe le surround). Sans format connu, celui de la piste, plafonné : au-delà de ce que SDL
// ouvre, le périphérique choisit (0).
audio::OutputRequest output_request(const media::TrackInfo& track, const std::string& name) {
  const std::optional<audio::AudioDevice> device = audio::resolve_device(audio::playback_devices(), name);
  const auto in_limits = [](int rate) {
    return rate >= audio::limits::kMinSampleRate && rate <= audio::limits::kMaxSampleRate;
  };
  const int rate = device && in_limits(device->sample_rate) ? device->sample_rate
                   : in_limits(track.sample_rate)           ? track.sample_rate
                                                            : 0;
  const bool stereo_device = !device || device->channels <= 2;
  const int channels = stereo_device ? std::clamp(track.channels, 0, 2)
                                     : std::clamp(device->channels, 1, audio::limits::kMaxChannels);
  return {.sample_rate = rate, .channels = channels, .device = name};
}

// Les langues de sous-titres voulues, préréglages d'abord : l'ordre des fichiers externes (A17-11).
std::vector<std::string> wanted_subtitles(const media::LanguagePriorities& languages) {
  std::vector<std::string> wanted;
  for (const media::LanguagePreset& preset : languages.presets)
    if (!preset.subtitles.empty()) wanted.push_back(preset.subtitles);
  wanted.insert(wanted.end(), languages.subtitles.begin(), languages.subtitles.end());
  return wanted;
}

}  // namespace

std::expected<std::unique_ptr<Pipeline>, Refusal> Pipeline::open(const PipelineConfig& config,
                                                                 const VideoStageFactory& make_video) {
  std::unique_ptr<Pipeline> p(new Pipeline());
  auto demuxer = media::Demuxer::open(config.file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  // P3.73 : les sous-titres à côté de la vidéo concourent avec ceux du conteneur, à leur langue lue dans le
  // nom (« Épisode 01.fr.ass »).
  std::vector<ExternalSubtitle> externals =
      config.subtitles ? find_external_subtitles(config.file, wanted_subtitles(config.languages))
                       : std::vector<ExternalSubtitle>{};
  media::TrackChoice choice = choose(config, *demuxer, externals);
  if (!choice.video) {
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable,
                                   std::format("{}: no video track", prepare::utf8_path(config.file))});
  }
  p->video_track_ = *demuxer->track(*choice.video);
  p->duration_us_ = demuxer->duration_us();
  for (const media::Chapter& chapter : demuxer->chapters())
    p->chapters_.push_back({.start_us = chapter.start_us, .title = chapter.title});
  std::ranges::sort(p->chapters_, {}, &ChapterMark::start_us);
  // A14-03 : une édition ordonnée se joue dans son ordre ; seuls les chapitres liés à d'autres fichiers
  // manquent.
  const media::ChapterLayout layout = demuxer->chapter_layout();
  p->chapters_flattened_ = layout.linked;
  for (const media::ChapterPart& part : layout.parts)
    p->parts_.push_back({.start_us = part.start_us, .end_us = part.end_us});
  AttachedFonts fonts = attached_fonts(demuxer->tracks());
  std::vector<media::PacketRoute> routes;

  // Toutes les pistes audio et de sous-titres lisibles sont routées : la barre de lecture en
  // change pendant la lecture (les ponts ne gardent que la piste choisie).
  if (config.audio && choice.audio) p->open_audio(config, *demuxer, *choice.audio, routes);

  for (const media::TrackInfo& track : demuxer->tracks()) {
    if (track.kind == media::TrackKind::Audio || track.kind == media::TrackKind::Subtitle)
      p->tracks_.push_back(track);
  }

  if (config.subtitles) {
    if (auto opened = p->open_subtitles(config, *demuxer, choice.subtitle, std::move(externals),
                                        std::move(fonts), routes);
        !opened)
      return std::unexpected(std::move(opened.error()));
  }

  auto media = media::open_source(std::move(*demuxer), *choice.video, config.decoder);
  if (!media) return std::unexpected(std::move(media.error()));
  p->decoder_api_ = std::string(media::name(media->decoder.api()));

  auto video = make_video(
      VideoStageInit{.track = &p->video_track_,
                     .ring = &p->mail_,
                     .credit = [inbox = &p->source_inbox_] { std::ignore = inbox->try_push(Credit{}); },
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
                                                         .lead = p->lead_.get(),
                                                         .subs_lead = p->subs_lead_.get()});

  auto source = media::SourceThread::start(
      std::move(*media),
      media::SourceLinks{.frames = &p->video_->frames(), .listener = &p->mail_, .packets = std::move(routes)},
      p->source_inbox_);
  if (!source) return std::unexpected(std::move(source.error()));
  p->source_ = std::move(*source);
  return p;
}

// Les pistes choisies : par langue, parmi celles du conteneur et les fichiers externes, ou celles d'avant
// une réouverture si elles existent encore.
media::TrackChoice Pipeline::choose(const PipelineConfig& config, const media::Demuxer& demuxer,
                                    const std::vector<ExternalSubtitle>& externals) {
  std::vector<media::TrackInfo> candidates(demuxer.tracks().begin(), demuxer.tracks().end());
  for (std::size_t k = 0; k < externals.size(); ++k) {
    const std::string extension = core::ascii_lower(prepare::utf8_path(externals[k].path.extension()));
    candidates.push_back({.index = kExternalSubtitleBase + static_cast<int>(k),
                          .kind = media::TrackKind::Subtitle,
                          .codec = extension == ".srt" ? "subrip" : "ass",
                          .language = externals[k].language,
                          .title = prepare::utf8_path(externals[k].path.filename())});
  }
  media::TrackChoice choice = media::choose_tracks(candidates, config.languages);
  const auto kind_of = [&](int index) {
    const media::TrackInfo* track = demuxer.track(index);
    return track != nullptr ? std::optional(track->kind) : std::nullopt;
  };
  if (config.audio_track && kind_of(*config.audio_track) == media::TrackKind::Audio)
    choice.audio = config.audio_track;
  if (config.subtitle_track && subtitle_exists(*config.subtitle_track, demuxer, externals.size()))
    choice.subtitle = *config.subtitle_track;
  return choice;
}

// L'audio : décodeurs des pistes lisibles, sortie, thread et horloge. Un échec ne ferme pas la lecture : la
// vidéo joue sans son, sur l'horloge hôte, et le refus paraît à l'OSD.
void Pipeline::open_audio(const PipelineConfig& config, const media::Demuxer& demuxer, int audio_index,
                          std::vector<media::PacketRoute>& routes) {
  std::optional<Refusal> audio_failure;
  for (const media::TrackInfo& track : demuxer.tracks()) {
    if (track.kind != media::TrackKind::Audio) continue;
    auto info = audio_codec_info(track);
    if (!info) {
      if (track.index == audio_index) audio_failure = std::move(info.error());
      continue;
    }
    audio_codecs_.emplace_back(track.index, std::move(*info));
    menu_.audio.push_back({.index = track.index, .label = track_label(track)});
  }
  const media::TrackInfo& track = *demuxer.track(audio_index);
  if (!audio_failure && audio_codec(track.index) == nullptr) {
    audio_failure =
        Refusal{RefusalCode::TrackUnavailable, std::format("piste audio {} illisible", track.index)};
  }
  if (!audio_failure) {
    auto output = audio::AudioOutput::open_sdl(output_request(track, config.audio_device));
    if (!output) {
      audio_failure = std::move(output.error());
    } else {
      audio::AudioThreadOptions options{
          .player = {.pll = config.audio_pll, .fallback_delay_us = config.audio_fallback_delay_us},
          .credit = [inbox = &source_inbox_](Credit credit) { std::ignore = inbox->try_push(credit); },
          .refused = [mail = &mail_](const Refusal& refusal) { mail->report("audio", refusal, false); }};
      auto thread = audio::AudioThread::start(std::move(*output), std::move(options));
      if (!thread) {
        audio_failure = std::move(thread.error());
      } else {
        audio_ = std::move(*thread);
      }
    }
  }
  if (audio_failure) {
    audio_codecs_.clear();
    menu_.audio.clear();
    mail_.report("audio", *audio_failure, false);
  } else {
    audio_->post(audio::AudioTrack{.codec = *audio_codec(track.index)});
    audio_bridge_ = std::make_unique<AudioBridge>(*audio_, track.index);
    lead_ = std::make_shared<LeadClock>(audio_->clock(), host_clock_);
    clock_ = lead_.get();
    mail_.attach_audio(audio_.get());
    for (const auto& [index, codec] : audio_codecs_)
      routes.push_back({.track = index, .sink = audio_bridge_.get()});
    audio_track_ = track;
    menu_.audio_selected = track.index;
  }
}

// Les sous-titres : menu, fichiers externes, thread subs (lancé même sans piste : un fichier glissé
// pendant la lecture s'y ajoute, P3.85), polices et piste choisie.
std::expected<void, Refusal> Pipeline::open_subtitles(
    const PipelineConfig& config, const media::Demuxer& demuxer, std::optional<int> subtitle,
    std::vector<ExternalSubtitle> externals, AttachedFonts fonts, std::vector<media::PacketRoute>& routes) {
  for (const media::TrackInfo& track : demuxer.tracks()) {
    if (track.kind != media::TrackKind::Subtitle || !subtitle_track(track, subs_track_id(track.index)))
      continue;
    menu_.subtitles.push_back({.index = track.index, .label = track_label(track)});
  }
  // Les fichiers à côté de la vidéo : après les pistes du conteneur, choisis d'office seulement
  // quand le conteneur n'en a aucune et qu'aucun choix n'a été fait.
  external_ = std::move(externals);
  for (std::size_t k = 0; k < external_.size(); ++k) {
    const int index = kExternalSubtitleBase + static_cast<int>(k);
    menu_.subtitles.push_back({.index = index, .label = external_[k].label});
    tracks_.push_back({.index = index,
                       .kind = media::TrackKind::Subtitle,
                       .codec = "externe",
                       .title = prepare::utf8_path(external_[k].path.filename())});
  }
  if (!subtitle && !config.subtitle_track && !external_.empty() && !readable_subtitles(demuxer))
    subtitle = kExternalSubtitleBase;
  note_image_subtitles(demuxer, subtitle.has_value(), mail_);
  if (subtitle && !std::ranges::contains(menu_.subtitles, *subtitle, &TrackEntry::index)) {
    report_unreadable(demuxer, *subtitle, mail_);
    subtitle.reset();
  }

  // Le fil des sous-titres tourne même sans piste : un fichier glissé pendant la lecture s'y ajoute (P3.85).
  subs_inbox_ = std::make_shared<subs::SubsInbox>(subs::kSubsInboxBounds);
  auto thread = subs::SubsThread::start(config.subs, mail_, subs_inbox_);
  if (!thread) return std::unexpected(std::move(thread.error()));
  subs_thread_ = std::move(*thread);
  send_fonts(std::move(fonts));
  std::optional<int> chosen;
  if (subtitle && std::ranges::contains(menu_.subtitles, *subtitle, &TrackEntry::index)) {
    chosen = subtitle;
    if (auto select = subtitle_select(*chosen)) {
      if (!subs_inbox_->try_push(std::move(*select)))  // P2.36 : un choix perdu se dit
        log_channel("app")->warn("subtitles: track {} choice lost, inbox full", *chosen);
      subtitle_track_ = *track_info(*chosen);
    } else {
      chosen.reset();
    }
  }
  menu_.subtitle_selected = chosen;
  subs_bridge_ = std::make_unique<SubsBridge>(subs_inbox_, chosen);
  std::vector<int> mov_text;
  bool contained = false;
  for (const TrackEntry& entry : menu_.subtitles) {
    if (entry.index >= kExternalSubtitleBase) continue;
    routes.push_back({.track = entry.index, .sink = subs_bridge_.get()});
    mark_mov_text(demuxer, entry.index, *subs_bridge_, mov_text);
    contained = true;
  }
  // A17-05, A17-06 : la piste choisie du conteneur, relue en avance au seek et pour un délai négatif.
  if (contained) {
    subs_lead_ = std::make_unique<SubtitleLead>(SubtitleLeadConfig{.file = config.file,
                                                                   .reference = video_track_.index,
                                                                   .mov_text = std::move(mov_text),
                                                                   .inbox = subs_inbox_});
    subs_lead_->select(container_track(chosen));
  }
  return {};
}

Pipeline::~Pipeline() = default;

void Pipeline::send_fonts(AttachedFonts fonts) {
  if (fonts.dropped > 0) {
    mail_.report("subs",
                 {RefusalCode::FieldOutOfRange, std::format("{} fonts dropped: attachments over {} MiB",
                                                            fonts.dropped, kMaxAttachedFontBytes >> 20)},
                 false);
  }
  for (subs::AttachFont& font : fonts.fonts) {
    // La boîte tient 64 polices à la fois : au-delà, on laisse au thread subs le temps d'en
    // prendre (une seconde au plus) avant d'en écarter une. try_push ne prend la police que s'il
    // réussit : elle n'est pas recopiée à chaque essai (A17-09).
    const std::string name = font.name;
    bool pushed = subs_inbox_->try_push(std::move(font));
    for (int wait = 0; !pushed && wait < 1000; ++wait) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      pushed = subs_inbox_->try_push(std::move(font));  // NOLINT(bugprone-use-after-move) : rendue intacte
    }
    if (!pushed)
      mail_.report("subs", {RefusalCode::SubtitleBacklog, std::format("font {} dropped", name)}, false);
  }
}

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
  // Ordre : les paquets de l'époque from_epoch sont de la nouvelle piste. AudioTrack part avant le seek, qui
  // pose le Flush : le thread audio sait déjà, quand les paquets de l'époque arrivent, qu'ils sont à elle, et
  // l'ancien décodeur ne lit que les siens (A16-06).
  audio_bridge_->select(track, from_epoch);
  if (!audio_->post(audio::AudioTrack{.codec = *codec, .from_epoch = from_epoch})) {
    // Le décodeur ne changera pas : le pont garde la piste d'avant, comme le menu, et rien n'est relu.
    audio_bridge_->undo_select();
    return std::unexpected(Refusal{RefusalCode::PacketBacklog, "audio track: audio mailbox full"});
  }
  seek();
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
    if (subs_lead_) subs_lead_->select(std::nullopt);
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
    return std::unexpected(Refusal{RefusalCode::SubtitleBacklog, "subtitle track: mailbox full"});
  subs_bridge_->select(track);
  if (subs_lead_) subs_lead_->select(container_track(track));
  menu_.subtitle_selected = track;
  subtitle_track_ = *info;
  return {};
}

std::expected<int, Refusal> Pipeline::add_subtitle_file(const std::filesystem::path& file) {
  if (!subs_inbox_)
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "pas de sous-titres pour cette lecture"});
  if (external_.size() >= kMaxExternalSubtitles)
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "trop de sous-titres externes"});
  const int index = kExternalSubtitleBase + static_cast<int>(external_.size());
  external_.push_back(
      {.path = file, .label = i18n::tr_format("External · {}", prepare::utf8_path(file.filename()))});
  menu_.subtitles.push_back({.index = index, .label = external_.back().label});
  tracks_.push_back({.index = index,
                     .kind = media::TrackKind::Subtitle,
                     .codec = "externe",
                     .title = prepare::utf8_path(file.filename())});
  return index;
}

void Pipeline::pump() {
  if (subs_bridge_ && subs_bridge_->take_refused())
    std::ignore = source_inbox_.try_push(Credit{});  // crédits bornés au ring : jamais de trop
}

SessionConfig Pipeline::session_config() const {
  // Matrice de la vidéo pour les couleurs VSFilter : celle que le flux annonce, sinon celle que mpv
  // devine à la taille. La plage vient du flux aussi : en plage pleine, VSFilter ne convertit pas.
  const bool hd = !kernels::sd_size(video_track_.width, video_track_.height);
  return SessionConfig{
      .duration_us = duration_us_,
      .audio_end_us = audio_track_ && audio_track_->duration_us > 0 &&
                              audio_track_->duration_us + 1'000'000 < duration_us_
                          ? audio_track_->duration_us
                          : 0,
      .chapters = chapters_,
      .parts = parts_,
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
  lines.push_back(std::format("{} {}x{} ({})", track_text("video", video_track_), video_track_.width,
                              video_track_.height, decoder_api_));
  lines.push_back(audio_track_ ? track_text("audio", *audio_track_) : i18n::tr("audio: none, host clock"));
  lines.push_back(subtitle_track_ ? track_text("subtitles", *subtitle_track_) : i18n::tr("subtitles: none"));
  return lines;
}

}  // namespace ajp::app::play
