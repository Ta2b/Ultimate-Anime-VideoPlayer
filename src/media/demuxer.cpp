#include "media/demuxer.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <deque>
#include <format>
#include <limits>
#include <set>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>

extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/dict.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/time.h>
}

#include "core/frame.h"
#include "core/utf8_path.h"
#include "media/av_util.h"
#include "media/frame_rate.h"
#include "media/tag_text.h"
#include "media/log.h"

namespace ajp::media {

struct Demuxer::State {
  FormatPtr format;
  std::vector<TrackInfo> tracks;
  std::vector<Chapter> chapters;
  ChapterLayout chapter_layout;  // A14-03 : édition ordonnée ou segments liés, que FFmpeg aplatit
  std::int64_t start_us = 0;
  std::int64_t duration_us = 0;
  PacketPtr scratch;            // reçoit av_read_frame avant d'être partagé
  std::deque<Packet> pushback;  // lus par le seek pour vérifier son point clé, rendus par read()
  std::shared_ptr<spdlog::logger> log = make_logger();
  std::uint32_t file_seeks = 0;
  std::atomic<bool> interrupted{false};  // lu par le rappel d'interruption de FFmpeg
  std::uint64_t packets_read = 0;        // paquets rendus par av_read_frame depuis l'ouverture
  ReadFault fault;                       // pour les tests : lectures qui échouent exprès

  std::expected<std::optional<Packet>, Refusal> read_file();
  int read_frame(AVPacket* packet);
  bool wait_before_retry(int attempt) const;
  struct RetryState;
  std::expected<bool, Refusal> on_read_error(int err, RetryState& retry);
  std::expected<std::optional<Packet>, Refusal> peek_keyframe(int reference);
  std::expected<void, Refusal> seek_stream(int reference, std::int64_t ts, int flags, std::string_view what);
  std::expected<std::int64_t, Refusal> seek_before(std::int64_t target_us, int reference);
  std::expected<std::optional<std::int64_t>, Refusal> seek_after(std::int64_t target_us, int reference);
  std::int64_t indexed_before(std::int64_t target_us, int reference) const;
  std::int64_t known_extent_us(int reference) const;
  std::int64_t last_end_us = 0;  // fin du dernier paquet lu, pour known_extent_us
  void note_chapter_layout(std::string_view shown);
};

namespace {

// Paquets lus au plus pour trouver le premier point clé après un seek.
constexpr int kMaxPeekPackets = 1024;
// Reculs au plus pour trouver un point clé qui s'affiche à ou avant la cible.
constexpr int kMaxRewinds = 8;
// Une erreur d'entrée-sortie (disque, partage réseau qui hésite) est retentée avec une attente qui double,
// 50, 100, 200, 400, 800 puis 1 s au plus, jusqu'à kIoRetryBudgetMs en tout : un Wi-Fi qui change de point
// d'accès ou un disque USB qui se réveille dépassent largement 1,5 s (B31-11).
constexpr int kIoRetryBudgetMs = 20'000;
constexpr int kFirstRetryWaitMs = 50;
constexpr int kMaxRetryWaitMs = 1000;
// Des données abîmées : le conteneur avance à chaque essai et se recale en général sur le bloc
// suivant, sans attente. Au-delà, le reste du fichier est tenu pour illisible.
constexpr int kDataRetries = 64;

// Une erreur de lecture qui peut guérir en attendant : une panne d'E/S ou de réseau (C14-02).
bool io_error_may_heal(int err) {
  for (const int code : {EIO, ETIMEDOUT, ECONNRESET, ECONNABORTED, ENETDOWN, ENETUNREACH, ENETRESET, EHOSTUNREACH,
                         EAGAIN, ENODEV, ENXIO, EPIPE}) {
    if (err == AVERROR(code)) return true;
  }
#ifdef ESTALE  // absent de la bibliothèque C de Windows
  if (err == AVERROR(ESTALE)) return true;
#endif
  return false;
}

// FFmpeg attend un chemin UTF-8, Windows compris : path::string() y rendrait la
// page de code ANSI, ou lèverait une exception devant un nom en japonais.
std::string utf8_path(const std::filesystem::path& path) {
  return core::to_utf8(path);
}

// B35-05 : un titre de piste ou de chapitre vient du fichier ; clean_tag_text le borne et le nettoie (C14-08).
std::string tag(const AVDictionary* metadata, const char* key) {
  const AVDictionaryEntry* entry = av_dict_get(metadata, key, nullptr, 0);
  if (entry == nullptr || entry->value == nullptr) return {};
  return clean_tag_text(entry->value);
}

// B14-01, B35-06 : FFmpeg lit tout préfixe « lettres: » d'au moins deux caractères comme un protocole
// (« Re:Zero 01.mkv » → « Protocol not found »). Le lecteur n'ouvre que des fichiers : chemin absolu et
// protocole « file: » explicite.
std::string open_url(const std::filesystem::path& path) {
  std::error_code ec;
  const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
  return "file:" + utf8_path(ec ? path : absolute);
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
  return standard_frame_rate({rate.num, rate.den});
}

// Les bornes d'une vidéo lisible (A35-06) et le nombre de chapitres gardés (A35-02).
constexpr int kMaxVideoSide = 8192;
constexpr int kMaxVideoAspect = 8;
constexpr std::size_t kMaxChapters = 1000;

// B35-01 : les polices jointes recopiées par ouverture ne dépassent pas ce total ; le reste est écarté.
constexpr std::int64_t kMaxAttachmentBytes = std::int64_t{512} << 20;

// Une piste vidéo hors des bornes d'une source lisible (A35-06), ou rien.
std::optional<Refusal> out_of_bounds(const AVCodecParameters& par) {
  const int longest = std::max(par.width, par.height);
  const int shortest = std::min(par.width, par.height);
  if (longest > kMaxVideoSide || (shortest > 0 && longest > kMaxVideoAspect * shortest))
    return Refusal{RefusalCode::FormatUnsupported,
                   std::format("vidéo de {}×{} : au-delà de {} de côté ou d'un rapport de 1:{}", par.width,
                               par.height, kMaxVideoSide, kMaxVideoAspect)};
  return std::nullopt;
}

// `attachment_bytes` : ce qui reste du total de polices à recopier. Une pièce jointe qui n'est pas une police
// n'est pas recopiée du tout (B35-01) : FFmpeg en garde déjà un exemplaire entier, une 2e est du gâchis.
std::expected<TrackInfo, Refusal> describe(const AVStream& stream, TrackKind kind,
                                           std::int64_t& attachment_bytes) {
  const AVCodecParameters& par = *stream.codecpar;
  std::shared_ptr<const CodecParameters> parameters;
  if (kind != TrackKind::Attachment ||
      (is_font_attachment(tag(stream.metadata, "filename"), tag(stream.metadata, "mimetype")) &&
       par.extradata_size <= attachment_bytes)) {
    auto copied = CodecParameters::copy_of(par);
    if (!copied) return std::unexpected(copied.error());
    parameters = std::move(*copied);
    if (kind == TrackKind::Attachment) attachment_bytes -= par.extradata_size;
  }

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
      .parameters = std::move(parameters),
  };
  if (info.language == "und") info.language.clear();  // « indéterminée » : absente
  if (kind == TrackKind::Video) {
    // Comme les images décodées (core/frame.h) : la chaîne se prépare pour la taille qu'elle recevra.
    info.width = chain_dimension(par.width);
    info.height = chain_dimension(par.height);
    info.frame_rate = frame_rate_of(stream);
    // A15-02 : 29,97 i/s entrelacé, le télécinéma d'un anime 23,976 : la chaîne reçoit les images d'origine.
    const bool interlaced = par.field_order != AV_FIELD_UNKNOWN && par.field_order != AV_FIELD_PROGRESSIVE;
    // Une hypothèse : Demuxer::set_inverse_telecine la corrige après mesure (B15-01).
    if (interlaced && std::int64_t{info.frame_rate.num} * 1001 == std::int64_t{info.frame_rate.den} * 30000) {
      info.inverse_telecine = true;
      info.frame_rate = {24000, 1001};
    }
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
        info.color_matrix = ColorMatrix::Bt2020Ncl;
        break;
      // B14-07 : la luminance constante (YcCbcCrc) n'est pas une matrice linéaire ; la lire comme NCL
      // fausserait luminance et couleurs. Sans étiquette, la matrice est devinée par la taille, et open() le
      // dit au journal.
      case AVCOL_SPC_BT2020_CL:
      default:
        break;
    }
    info.full_range = par.color_range == AVCOL_RANGE_JPEG;
    if (const AVPacketSideData* dovi =
            av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data, AV_PKT_DATA_DOVI_CONF);
        dovi != nullptr && dovi->size >= sizeof(AVDOVIDecoderConfigurationRecord))
      info.dolby_vision_profile =
          reinterpret_cast<const AVDOVIDecoderConfigurationRecord*>(dovi->data)->dv_profile;
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

// A14-03 : les parties d'une édition ordonnée passent sur la ligne de temps du Demuxer ; ce que le lecteur
// ne joue pas (chapitres d'un autre fichier) est dit au journal.
void Demuxer::State::note_chapter_layout(std::string_view shown) {
  for (ChapterPart& part : chapter_layout.parts) {
    part.start_us = std::max<std::int64_t>(part.start_us - start_us, 0);
    if (part.end_us != ChapterPart::kToEnd) part.end_us -= start_us;
  }
  if (!chapter_layout.parts.empty())
    log->info("{} : édition ordonnée, {} parties jouées dans son ordre", shown, chapter_layout.parts.size());
  else if (chapter_layout.ordered)
    log->info(
        "{} : édition ordonnée non jouée ou sans partie dans ce fichier, lecture dans l'ordre du fichier",
        shown);
  if (chapter_layout.linked)
    log->warn("{} : chapitres liés à d'autres fichiers non pris en charge, ils ne sont pas joués", shown);
}

// Les pistes du fichier, dans state->tracks.
static std::expected<void, Refusal> describe_streams(const AVFormatContext& format,
                                                     std::vector<TrackInfo>& tracks, spdlog::logger& log,
                                                     const std::string& shown, OpenOptions options) {
  const AVFormatContext* raw = &format;
  // B14-04, B35-04 : une piste vidéo hors des bornes (vignettes, bandeau 1920×200) est écartée avec un
  // avertissement ; le fichier n'est refusé que s'il ne reste aucune vidéo lisible.
  std::optional<Refusal> dropped_video;
  bool has_video = false;
  std::int64_t attachment_bytes = options.attachments ? kMaxAttachmentBytes : 0;
  for (unsigned i = 0; i < raw->nb_streams; ++i) {
    const AVStream& stream = *raw->streams[i];
    const std::optional<TrackKind> kind = kind_of(stream.codecpar->codec_type);
    if (!kind) continue;
    if (*kind == TrackKind::Attachment && !options.attachments) continue;
    if (*kind == TrackKind::Video && (stream.disposition & AV_DISPOSITION_ATTACHED_PIC) == 0) {
      // Une taille annoncée hors de tout épisode (16384×4, 16384×16384) ferait construire des moteurs de
      // cette forme pendant des minutes, gardés en cache, puis refusés faute de mémoire vidéo (A35-06).
      if (auto refused = out_of_bounds(*stream.codecpar)) {
        log.warn("{} : piste {} écartée, {}", shown, stream.index, refused->detail);
        if (!dropped_video) dropped_video = std::move(refused);
        continue;
      }
      has_video = true;
    }
    if (*kind == TrackKind::Attachment && stream.codecpar->extradata_size > attachment_bytes &&
        is_font_attachment(tag(stream.metadata, "filename"), tag(stream.metadata, "mimetype")))
      log.warn("{} : police jointe {} ({} octets) écartée, au-delà de {} Mio de polices", shown, stream.index,
               stream.codecpar->extradata_size, kMaxAttachmentBytes >> 20);
    if (*kind == TrackKind::Video && stream.codecpar->color_space == AVCOL_SPC_BT2020_CL)
      log.warn("{} : matrice BT.2020 à luminance constante non prise en charge, devinée par la taille",
               shown);
    auto info = describe(stream, *kind, attachment_bytes);
    if (!info) return std::unexpected(info.error());
    tracks.push_back(std::move(*info));
  }
  if (dropped_video && !has_video) return std::unexpected(std::move(*dropped_video));
  return {};
}

std::expected<Demuxer, Refusal> Demuxer::open(const std::filesystem::path& path, OpenOptions options,
                                              std::stop_token stop) {
  auto state = std::make_unique<State>();
  // L'arrêt pose le drapeau que lit le rappel d'interruption de FFmpeg, pendant l'ouverture aussi ; déjà
  // demandé, il est posé tout de suite et l'ouverture refuse dès sa première lecture.
  const std::stop_callback cancel(std::move(stop), [flag = &state->interrupted] { flag->store(true); });
  const std::string where = open_url(path);
  const std::string shown = utf8_path(path.filename());  // le journal n'a que le nom du fichier (A39-03)
  AVFormatContext* raw = avformat_alloc_context();
  if (raw == nullptr) return refuse(RefusalCode::MediaUnreadable, shown + " : mémoire insuffisante");
  // Le rappel d'interruption lit le drapeau de l'état, qui garde son adresse quand le Demuxer bouge.
  raw->interrupt_callback = {
      .callback = [](void* opaque) { return static_cast<State*>(opaque)->interrupted.load() ? 1 : 0; },
      .opaque = state.get()};
  // rw_timeout ne vaut que pour les protocoles réseau de FFmpeg (HTTP, TCP) : un fichier, même sur un partage
  // SMB ou NFS monté, passe par le protocole « file », un read() du noyau que ni ce délai ni le rappel
  // d'interruption ne coupent (B14-03, B31-05). Il reste posé pour les flux réseau. Seul « file » est permis
  // : le lecteur n'ouvre pas d'URL, et un fichier de liste ne doit pas en suivre une.
  AVDictionary* av_options = nullptr;
  av_dict_set(&av_options, "rw_timeout", "30000000", 0);
  av_dict_set(&av_options, "protocol_whitelist", "file", 0);
  const int opened = avformat_open_input(&raw, where.c_str(), nullptr, &av_options);
  av_dict_free(&av_options);
  if (opened < 0) return refuse_av(RefusalCode::MediaUnreadable, shown, opened);
  state->format.reset(raw);
  if (const int err = avformat_find_stream_info(raw, nullptr); err < 0) {
    return refuse_av(RefusalCode::MediaUnreadable, shown, err);
  }
  state->start_us = raw->start_time == AV_NOPTS_VALUE ? 0 : raw->start_time;
  state->duration_us = raw->duration == AV_NOPTS_VALUE || raw->duration < 0 ? 0 : raw->duration;

  if (auto described = describe_streams(*raw, state->tracks, *state->log, shown, options); !described)
    return std::unexpected(std::move(described.error()));
  // Plusieurs éditions Matroska (ou un chapitre répété) donnent le même chapitre plusieurs fois : un
  // ensemble les écarte en O(n log n), et au plus kMaxChapters sont gardés (A35-02 : 100 000 chapitres
  // prenaient 7 s par ouverture).
  std::set<std::pair<std::int64_t, std::string>> seen;
  for (unsigned i = 0; i < raw->nb_chapters && state->chapters.size() < kMaxChapters; ++i) {
    const AVChapter& chapter = *raw->chapters[i];
    Chapter next{.start_us = to_us(chapter.start, chapter.time_base, state->start_us),
                 .end_us = to_us(chapter.end, chapter.time_base, state->start_us),
                 .title = tag(chapter.metadata, "title")};
    if (seen.emplace(next.start_us, next.title).second) state->chapters.push_back(std::move(next));
  }
  if (raw->nb_chapters > kMaxChapters && state->chapters.size() == kMaxChapters)
    state->log->info("{} : {} chapitres, {} gardés", shown, raw->nb_chapters, kMaxChapters);
  if (std::string_view(raw->iformat->name).find("matroska") != std::string_view::npos) {
    // B14-05 : les Cues (l'index du MKV) ne se lisent qu'au premier seek ; un seek au début, sans effet
    // puisque rien n'est encore lu, les charge pour que known_extent_us voie jusqu'où va vraiment le fichier,
    // sans toucher à la position d'un seek refusé plus tard.
    for (unsigned i = 0; i < raw->nb_streams; ++i) {
      if (raw->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO ||
          raw->streams[i]->discard == AVDISCARD_ALL)
        continue;
      static_cast<void>(av_seek_frame(raw, static_cast<int>(i),
                                      from_us(0, raw->streams[i]->time_base, state->start_us),
                                      AVSEEK_FLAG_BACKWARD));
      break;
    }
    state->chapter_layout = read_chapter_layout(path);
    state->note_chapter_layout(shown);
  }
  state->scratch.reset(av_packet_alloc());
  if (!state->scratch) return refuse(RefusalCode::MediaUnreadable, shown + " : mémoire insuffisante");
  state->log->debug("{} : {} pistes, {} chapitres, {} µs", shown, state->tracks.size(),
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
ChapterLayout Demuxer::chapter_layout() const {
  return state_->chapter_layout;
}
std::int64_t Demuxer::duration_us() const {
  return state_->duration_us;
}

const TrackInfo* Demuxer::track(int index) const {
  const auto it = std::ranges::find(state_->tracks, index, &TrackInfo::index);
  return it == state_->tracks.end() ? nullptr : &*it;
}

void Demuxer::set_inverse_telecine(int index, bool telecine) {
  const auto it = std::ranges::find(state_->tracks, index, &TrackInfo::index);
  if (it == state_->tracks.end() || it->inverse_telecine == telecine) return;
  it->inverse_telecine = telecine;
  it->frame_rate = telecine ? Rational{24000, 1001} : Rational{30000, 1001};
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

void Demuxer::fail_reads(ReadFault fault) {
  state_->fault = fault;
}

int Demuxer::State::read_frame(AVPacket* packet) {
  if (fault.count > 0 && packets_read >= fault.after_packets) {
    --fault.count;
    if (!fault.sticky_eof || format->pb == nullptr) return fault.error;
    // Comme mov_read_packet après une erreur d'E/S : l'erreur reste dans l'AVIOContext et la
    // lecture rend une fin de fichier.
    format->pb->error = fault.error;
    format->pb->eof_reached = 1;
    return AVERROR_EOF;
  }
  const int err = av_read_frame(format.get(), packet);
  if (err >= 0) ++packets_read;
  return err;
}

// Attend avant le nouvel essai `attempt` (à partir de 0), par tranches de 10 ms pour qu'une
// fermeture n'attende pas. Rend faux si la lecture a été interrompue entre-temps.
bool Demuxer::State::wait_before_retry(int attempt) const {
  const int wait_ms = std::min(kFirstRetryWaitMs << std::min(attempt, 8), kMaxRetryWaitMs);
  for (int waited = 0; waited < wait_ms; waited += 10) {
    if (interrupted.load()) return false;
    av_usleep(10'000);
  }
  return !interrupted.load();
}

// Ce qui est réessayé dans une lecture : erreurs d'E/S (attente cumulée) et de données.
struct Demuxer::State::RetryState {
  int io_errors = 0;
  int io_waited_ms = 0;
  int data_errors = 0;
};

// Une erreur de lecture n'arrête pas la vidéo : retentée, puis, si elle dure, un refus `recoverable`. Rend
// vrai pour une vraie fin de fichier, faux pour « réessayer », un refus sinon.
std::expected<bool, Refusal> Demuxer::State::on_read_error(int err, RetryState& retry) {
  const AVFormatContext& ctx = *format;
  // Une erreur d'E/S pose aussi eof_reached : beaucoup de démultiplexeurs (MP4 par avio_feof,
  // AVI, MPEG-TS) rendent alors AVERROR_EOF. L'erreur collée à l'AVIOContext dit la vraie
  // cause, qui passe par les nouveaux essais au lieu de finir l'épisode en silence (A14-01).
  if (err == AVERROR_EOF && ctx.pb != nullptr && ctx.pb->error < 0 && ctx.pb->error != AVERROR_EOF)
    err = ctx.pb->error;
  if (err == AVERROR_EOF) return true;
  // A14-07 : trois causes, trois codes. Une interruption (fermeture) : Cancelled ; une panne du disque
  // ou du réseau : IoFailed ; des données abîmées : DecodeFailed.
  if (err == AVERROR_EXIT || interrupted.load())
    return refuse_av(RefusalCode::Cancelled, "lecture d'un paquet interrompue", err);
  const bool data = err == AVERROR_INVALIDDATA;
  // C14-02 : seules les pannes d'E/S qui peuvent guérir (disque qui se réveille, Wi-Fi qui change de point d'accès)
  // sont retentées. ENOMEM, EINVAL, AVERROR_BUG… ne guérissent pas : pas de gel de 20 s pour rien.
  if (!data && !io_error_may_heal(err)) {
    auto refused = refuse_av(RefusalCode::DecodeFailed, "lecture d'un paquet, erreur qui ne guérit pas", err);
    refused.error().recoverable = true;
    return refused;
  }
  int& errors = data ? retry.data_errors : retry.io_errors;
  const int io_budget_ms = fault.io_budget_ms > 0 ? fault.io_budget_ms : kIoRetryBudgetMs;
  if (data ? errors >= kDataRetries : retry.io_waited_ms >= io_budget_ms) {
    auto refused = refuse_av(data ? RefusalCode::DecodeFailed : RefusalCode::IoFailed,
                             std::format("lecture d'un paquet, après {} essais", errors + 1), err);
    refused.error().recoverable = true;
    return refused;
  }
  // Un seul avertissement par série : un bloc abîmé peut coûter des dizaines d'essais.
  if (errors == 0) log->warn("lecture d'un paquet : {}, nouvel essai", av_error_text(err));
  if (!data) {
    if (!wait_before_retry(errors))
      return refuse_av(RefusalCode::Cancelled, "lecture d'un paquet interrompue", AVERROR_EXIT);
    retry.io_waited_ms += std::min(kFirstRetryWaitMs << std::min(errors, 8), kMaxRetryWaitMs);
    // Une erreur de l'AVIOContext reste collée : sans l'effacer, chaque essai la rendrait.
    if (ctx.pb != nullptr) {
      ctx.pb->error = 0;
      ctx.pb->eof_reached = 0;
    }
  }
  ++errors;
  return false;
}

// Une erreur de lecture n'arrête pas la vidéo : elle est retentée, puis, si elle dure, la lecture
// s'arrête là comme à la fin du fichier, par un refus `recoverable` que la source traite comme
// une fin de flux. Seule une interruption (fermeture) reste un refus qui arrête la source.
std::expected<std::optional<Packet>, Refusal> Demuxer::State::read_file() {
  const AVFormatContext& ctx = *format;
  AVPacket* raw = scratch.get();
  RetryState retry;
  while (true) {
    if (const int err = read_frame(raw); err < 0) {
      auto verdict = on_read_error(err, retry);
      if (!verdict) return std::unexpected(std::move(verdict.error()));
      if (*verdict) return std::nullopt;  // fin du fichier
      continue;
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
  if (raw->pts != AV_NOPTS_VALUE)
    last_end_us = std::max(last_end_us, raw->pts + std::max<std::int64_t>(raw->duration, 0));

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

std::int64_t Demuxer::State::known_extent_us(int reference) const {
  if (duration_us <= 0) return 0;  // durée inconnue : pas de borne
  std::int64_t extent = std::max(duration_us, last_end_us);
  AVStream* stream = format->streams[reference];
  const int count = avformat_index_get_entries_count(stream);
  if (const AVIndexEntry* last = count > 0 ? avformat_index_get_entry(stream, count - 1) : nullptr;
      last != nullptr)
    extent = std::max(extent, to_us(last->timestamp, stream->time_base, start_us));
  return extent;
}

std::expected<void, Refusal> Demuxer::seek(std::int64_t target_us, int reference, SeekTarget where) {
  if (track(reference) == nullptr)
    return refuse(RefusalCode::TrackUnavailable, std::format("piste {}", reference));
  // B14-05 : la durée annoncée est parfois plus courte que le contenu (MKV remuxé sans mise à jour de
  // l'en-tête). Est permis tout ce que le fichier a déjà montré : la durée, le dernier paquet lu, le
  // dernier point de l'index du conteneur (les Cues d'un MKV).
  const std::int64_t extent_us = state_->known_extent_us(reference);
  if (target_us < 0 || (extent_us > 0 && target_us > extent_us)) {
    return refuse(RefusalCode::SeekFailed,
                  std::format("cible {} µs hors de [0, {}] µs", target_us, extent_us));
  }

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

std::expected<void, Refusal> Demuxer::seek_file(std::int64_t target_us, int reference) {
  if (track(reference) == nullptr)
    return refuse(RefusalCode::TrackUnavailable, std::format("piste {}", reference));
  const std::int64_t aim_us = std::max<std::int64_t>(target_us, 0);
  const AVRational time_base = state_->format->streams[reference]->time_base;
  return state_->seek_stream(reference, from_us(aim_us, time_base, state_->start_us), AVSEEK_FLAG_BACKWARD,
                             std::format("seek du fichier à {} µs", aim_us));
}

std::uint32_t Demuxer::file_seeks() const {
  return state_->file_seeks;
}

}  // namespace ajp::media
