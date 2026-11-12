#include "media/video_decoder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/cpu.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mathematics.h>
}

#include "av/input_whitelist.h"
#include "media/av_util.h"
#include "media/deinterlace.h"
#include "media/frame_data.h"
#include "media/hw_device.h"
#include "media/log.h"

namespace ajp::media {

namespace {

// pts + durée, kNoPts si l'un manque ou si la somme déborde (un fichier forgé, A15-10). La durée n'est
// jamais négative.
std::int64_t end_of(std::int64_t pts, std::int64_t duration) {
  if (pts == kNoPts || duration > std::numeric_limits<std::int64_t>::max() - pts) return kNoPts;
  return pts + duration;
}

// Threads du décodage logiciel : la valeur demandée, ou un par cœur
// (affinité comprise) jusqu'à kAutoDecoderThreads. FFmpeg en prendrait
// jusqu'à 16, soit 15 images de retard avant la première.
int software_threads(int requested) {
  if (requested > 0) return requested;
  return std::clamp(av_cpu_count(), 1, kAutoDecoderThreads);
}

}  // namespace

int background_decoder_threads() {
  // B15-07, B34-05 : la même règle partout. Une HEVC 4K 10 bits sur 1 thread décode à 20 i/s : la pré-analyse
  // des plans n'était pas finie quand l'épisode l'était, et la frontière analysée avançait 2 à 4 fois moins
  // vite sous Windows. Les threads de FFmpeg n'héritent pas de la priorité basse du créateur ; mais la
  // lecture, elle, décode par NVDEC et ses threads critiques sont en MMCSS : des ouvriers de priorité normale
  // ne les doublent pas.
  return std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 2, 4);
}

struct VideoDecoder::State {
  // Paquets gardés au plus avant la première image, pour la réouverture en logiciel.
  static constexpr std::size_t kMaxReplayPackets = 64;

  HwDevice hw;  // déclaré avant codec : détruit après lui
  CodecContextPtr codec;
  TrackInfo opened;           // la piste de l'ouverture, pour la réouverture en logiciel
  int requested_threads = 0;  // DecoderOptions::threads
  HwApi api = HwApi::Software;
  bool software_allowed = false;
  bool download = false;
  bool full_host_formats = false;  // DecoderOptions::full_host_formats
  bool host_422 = false;           // DecoderOptions::host_422
  int extra_hw_frames = 0;         // de l'essai en cours
  bool direct = false;             // NVDEC sans copie de sécurité (DecoderOptions::direct_surfaces)
  int safe_extra_hw_frames = 0;    // les surfaces de plus de l'essai quand la copie de sécurité reste
  std::vector<Attempt> rejected;
  std::optional<Packet> pending;  // refusé par EAGAIN, renvoyé au prochain receive()
  bool drained = false;
  std::int64_t next_pts_us = kNoPts;
  std::int64_t nominal_duration_us = 0;
  int data_errors = 0;
  // Bascule en logiciel avant la première image (get_format) : le contexte matériel ne décode que
  // sur un thread, il est rouvert en logiciel sur les threads demandés et les paquets envoyés
  // depuis l'ouverture (sent) sont rejoués (replay), la fin du flux après eux si elle était partie.
  bool output = false;       // une image rendue depuis l'ouverture
  bool replayable = true;    // sent tient tous les paquets envoyés depuis l'ouverture
  bool reopen = false;       // à rouvrir en logiciel au retour de FFmpeg
  bool ended = false;        // send_end appelé
  bool end_pending = false;  // send_end à renvoyer après replay
  std::vector<Packet> sent;
  std::deque<Packet> replay;
  // P3.22 : une bascule après la première image rouvre aussi en logiciel, sur les threads demandés, à partir
  // du dernier paquet s'il est une image clé, sinon de la suivante (wait_key).
  std::optional<Packet> last;
  bool wait_key = false;
  // B15-04 : les paquets envoyés depuis la dernière image clé, une fois la première image rendue (décodage
  // matériel seulement, borné). Une erreur du pilote rouvre alors en logiciel depuis cette image clé, sans
  // figer l'image jusqu'à la suivante ; delivered_us écarte ce qui a déjà été montré (catching_up).
  static constexpr std::size_t kMaxGopPackets = 2048;
  static constexpr std::size_t kMaxGopBytes = std::size_t{48} << 20;
  std::vector<Packet> gop;
  std::size_t gop_bytes = 0;
  bool gop_valid = false;
  std::int64_t delivered_us = kNoPts;  // pts de la dernière image remise à l'aval
  bool catching_up = false;
  Deinterlacer deinterlacer;  // P3.25 : les images logicielles entrelacées, par bwdif
  bool deinterlace_flushed = false;
  std::optional<std::int64_t> skip_before;  // VideoDecoder::skip_before
  FramePtr last_skipped;                    // la dernière image écartée, ressortie à la fin du flux
  // A15-09 : chaque image attend la suivante, dont le pts donne sa vraie durée. Celle du paquet ne vaut pas
  // mieux en VFR : le Matroska la tire de DefaultDuration, la période moyenne. Elle ne sert qu'en dernier
  // recours (dernière image, pts inconnu).
  FramePtr held;
  std::shared_ptr<spdlog::logger> log = make_logger();
  std::optional<ColorInfo>
      last_color;  // la dernière couleur valide, pour une image dont l'étiquette est refusée
  bool color_fallback_logged = false;
  std::chrono::steady_clock::time_point last_data_log{};
  int data_errors_unlogged = 0;

  // Données abîmées : l'image en cause est perdue, le décodage continue. Une ligne par seconde au plus avec
  // le compte : un flux abîmé à 24 i/s ne doit pas écrire 24 lignes par seconde sur le thread source
  // (C15-04).
  void data_error(std::string_view where) {
    ++data_errors;
    ++data_errors_unlogged;
    const auto now = std::chrono::steady_clock::now();
    if (data_errors > 1 && now - last_data_log < std::chrono::seconds(1)) return;
    log->warn("données invalides ({}) : {} image(s) perdue(s), le décodage continue", where,
              data_errors_unlogged);
    data_errors_unlogged = 0;
    last_data_log = now;
  }

  void fall_back(std::string_view why) {
    if (api == HwApi::Software) return;
    log->warn("{} : {} ; décodage logiciel", name(api), why);
    rejected.push_back(
        {api, Refusal{RefusalCode::HwDecodeUnavailable, std::format("{} : {}", name(api), why)}});
    api = HwApi::Software;
  }

  // A15-03 : une erreur du décodeur matériel en cours de flux (pilote, pool, profil hors de ses moyens) le
  // rouvre en logiciel, une fois, quand les essais le permettent. Vrai si rouvert.
  bool recover(int err, std::string_view where) {
    if (api == HwApi::Software || !software_allowed || err == AVERROR(EAGAIN) || err == AVERROR_INVALIDDATA ||
        err == AVERROR_EOF)
      return false;
    fall_back(refuse_av(RefusalCode::DecodeFailed, where, err).error().detail);
    return reopen_software();
  }

  // Choisit le format matériel s'il est proposé. extra_hw_frames se pose ici,
  // pour ce seul format : sous NVDEC il n'augmente que ulNumDecodeSurfaces
  // (nvdec.c:416), le pool CUDA allouant ses images à la demande.
  static AVPixelFormat get_format(AVCodecContext* ctx, const AVPixelFormat* formats) {
    auto& self = *static_cast<State*>(ctx->opaque);
    if (self.hw.device) {
      for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == self.hw.format) {
          // C15-06 : la taille du flux a pu changer depuis l'ouverture (concaténation, TS réassemblé) : une
          // taille au-delà du plafond reprend la copie de sécurité et ses surfaces, pas une cinquantaine de
          // surfaces 4K gardées.
          if (self.direct && std::int64_t{ctx->coded_width} * ctx->coded_height > kMaxDirectSurfacePixels) {
            self.direct = false;
            self.extra_hw_frames = self.safe_extra_hw_frames;
            ctx->hwaccel_flags &= ~AV_HWACCEL_FLAG_UNSAFE_OUTPUT;
            self.log->info("NVDEC : copie de sécurité reprise, {}x{} au-delà de 1440p", ctx->coded_width,
                           ctx->coded_height);
          }
          ctx->extra_hw_frames = self.extra_hw_frames;
          return *format;
        }
      }
      self.fall_back("format matériel non proposé par get_format (profil, taille ou pilote)");
      if (!self.software_allowed) return AV_PIX_FMT_NONE;
      self.reopen = self.output || self.replayable;
    }
    return avcodec_default_get_format(ctx, formats);
  }

  // avcodec_send_packet, le paquet gardé pour une réouverture tant qu'aucune image n'est sortie.
  int submit(const Packet& packet) {
    const bool keep = !output && replayable && sent.size() < kMaxReplayPackets;
    if (!output && !keep) replayable = false;
    if (keep) sent.push_back(packet);
    const int err = avcodec_send_packet(codec.get(), packet.av.get());
    if (keep && err == AVERROR(EAGAIN)) sent.pop_back();
    if (output && err != AVERROR(EAGAIN)) {
      last = packet;
      if (api != HwApi::Software) track_gop(packet);
    }
    return err;
  }

  void track_gop(const Packet& packet) {
    if ((packet.av->flags & AV_PKT_FLAG_KEY) != 0) {
      gop.clear();
      gop_bytes = 0;
      gop_valid = true;
    }
    if (!gop_valid) return;
    gop.push_back(packet);
    gop_bytes += packet.footprint();
    if (gop.size() > kMaxGopPackets || gop_bytes > kMaxGopBytes) {  // trop long : l'ancien repli reste
      gop.clear();
      gop_bytes = 0;
      gop_valid = false;
    }
  }

  // Vrai si le décodeur est rouvert, les paquets à rejouer dans replay. Sans décodeur logiciel
  // ouvrable, le contexte basculé garde le flux, sur un seul thread.
  bool reopen_software() {
    reopen = false;
    replayable = false;
    HwDevice old_hw = std::exchange(hw, HwDevice{});  // déclaré avant : détruit après le codec
    CodecContextPtr old_codec = std::move(codec);
    const AVCodec* decoder = avcodec_find_decoder(opened.parameters->av().codec_id);
    auto reopened =
        decoder != nullptr
            ? open_codec(*decoder, opened, requested_threads)
            : std::expected<void, Refusal>(std::unexpected(
                  Refusal{RefusalCode::DecoderUnavailable, std::format("aucun décodeur {}", opened.codec)}));
    if (!reopened) {
      codec = std::move(old_codec);
      hw = std::move(old_hw);
      sent.clear();
      log->warn("décodage logiciel sur un thread, réouverture refusée : {}", reopened.error().detail);
      return false;
    }
    direct = false;
    if (output) {
      // L'image tenue vient du pilote qu'on quitte : sa copie asynchrone n'a plus d'événement à attendre.
      // Elle est jetée, et se redécode en logiciel avec ce qui suit (B15-04).
      held.reset();
      const bool key = last && (last->av->flags & AV_PKT_FLAG_KEY) != 0;
      replay.clear();
      if (gop_valid && !gop.empty() && (gop.front().av->flags & AV_PKT_FLAG_KEY) != 0) {
        replay.assign(gop.begin(), gop.end());
        catching_up = delivered_us != kNoPts;
        wait_key = false;
      } else {
        if (key) replay.push_back(*last);
        wait_key = !key;
        if (wait_key) pending.reset();
      }
      gop.clear();
      gop_bytes = 0;
      gop_valid = false;
    } else {
      replay.assign(sent.begin(), sent.end());
    }
    sent.clear();
    if (pending) {
      replay.push_back(std::move(*pending));
      pending.reset();
    }
    end_pending = ended;
    log->info("décodage logiciel rouvert sur {} threads, {} paquets rejoués", codec->thread_count,
              replay.size());
    return true;
  }

  // Horodatage et événement de fin de copie d'une image prête. Un trou d'horodatage se comble par l'image
  // précédente plus sa durée. Sans image précédente (début, ou après un flush), il reste inconnu : kNoPts,
  // jamais un 0 inventé.
  bool skipped(FramePtr& frame);
  void release_covering(const AVFrame& next);
  bool already_shown(const AVFrame& frame);
  std::expected<std::optional<DecodedFrame>, Refusal> deliver(FramePtr frame);
  std::expected<std::optional<DecodedFrame>, Refusal> timed(FramePtr frame);

  std::expected<std::optional<DecodedFrame>, Refusal> finish(FramePtr frame) {
    const std::int64_t pts =
        frame->best_effort_timestamp == AV_NOPTS_VALUE ? next_pts_us : frame->best_effort_timestamp;
    const std::int64_t duration = frame->duration > 0 ? frame->duration : nominal_duration_us;
    next_pts_us = end_of(pts, duration);
    if (pts != kNoPts) delivered_us = pts;
    ReadyEvent ready;
    if (hw.stream && frame->hw_frames_ctx != nullptr) {
      auto recorded = hw.stream->record();
      if (!recorded) return std::unexpected(recorded.error());
      ready = std::move(*recorded);
    }
    bool fallback_used = false;
    auto decoded = to_decoded_frame(std::move(frame), pts, duration, std::move(ready), full_host_formats,
                                    host_422, last_color ? &*last_color : nullptr, &fallback_used);
    if (!decoded) return std::unexpected(std::move(decoded.error()));
    if (fallback_used) {
      if (!color_fallback_logged)
        log->warn("étiquette de couleur de l'image hors du vocabulaire : la couleur précédente reste");
      color_fallback_logged = true;
    } else {
      last_color = decoded->color;
    }
    return decoded;
  }

  // Les paquets à rejouer, puis la fin du flux ; arrêté par EAGAIN, repris au receive() suivant.
  std::expected<void, Refusal> resend() {
    while (!replay.empty()) {
      const int err = avcodec_send_packet(codec.get(), replay.front().av.get());
      if (err == AVERROR(EAGAIN)) return {};
      if (err == AVERROR_INVALIDDATA) {
        data_error("paquet rejoué");
      } else if (err < 0) {
        return refuse_av(RefusalCode::DecodeFailed, "paquet rejoué", err);
      }
      replay.pop_front();
    }
    if (end_pending) {
      end_pending = false;
      const int err = avcodec_send_packet(codec.get(), nullptr);
      if (err < 0 && err != AVERROR_EOF) return refuse_av(RefusalCode::DecodeFailed, "fin du flux", err);
    }
    return {};
  }

  std::expected<void, Refusal> open_codec(const AVCodec& decoder, const TrackInfo& track, int threads,
                                          const std::string& resize = {}) {
    CodecContextPtr ctx(avcodec_alloc_context3(&decoder));
    if (!ctx) return refuse(RefusalCode::DecoderUnavailable, "contexte de codec : mémoire insuffisante");
    if (const int err = avcodec_parameters_to_context(ctx.get(), &track.parameters->av()); err < 0) {
      return refuse_av(RefusalCode::DecoderUnavailable, "paramètres de la piste", err);
    }
    ctx->pkt_timebase = kMicroseconds;  // Demuxer rend des paquets en microsecondes
    ctx->opaque = this;
    ctx->get_format = &State::get_format;
    // Les hwaccels ne gagnent rien aux threads d'images ; le logiciel, si.
    ctx->thread_count = hw.device ? 1 : software_threads(threads);
    if (hw.device) ctx->hw_device_ctx = av_buffer_ref(hw.device.get());
    if (direct) ctx->hwaccel_flags |= AV_HWACCEL_FLAG_UNSAFE_OUTPUT;
    AVDictionary* settings = nullptr;
    (void)av_dict_set(&settings, "codec_whitelist", av::kAllowedDecoders, 0);  // C35-01
    if (!resize.empty()) (void)av_dict_set(&settings, "resize", resize.c_str(), 0);
    const int err = avcodec_open2(ctx.get(), &decoder, &settings);
    av_dict_free(&settings);
    if (err < 0)
      return refuse_av(RefusalCode::DecoderUnavailable, std::format("{} : avcodec_open2", name(api)), err);
    codec = std::move(ctx);
    return {};
  }

  std::expected<void, Refusal> try_attempt(HwApi attempt, const TrackInfo& track,
                                           const DecoderOptions& options) {
    const AVCodecID id = track.parameters->av().codec_id;
    api = attempt;
    hw = HwDevice{};
    direct = false;
    // Une surface de plus pour l'image que timed() retient jusqu'à la suivante (A15-09).
    extra_hw_frames = options.extra_hw_frames + 1;
    safe_extra_hw_frames = extra_hw_frames;
    if (attempt == HwApi::Software) {
      const AVCodec* decoder = avcodec_find_decoder(id);
      if (decoder == nullptr) {
        return refuse(RefusalCode::DecoderUnavailable,
                      std::format("aucun décodeur {}", avcodec_get_name(id)));
      }
      // 8.8 : l'image réduite par NVDEC, sinon le logiciel ordinaire.
      if (!options.cuvid_resize.empty()) {
        const std::string cuvid = std::format("{}_cuvid", avcodec_get_name(id));
        if (const AVCodec* reduced = avcodec_find_decoder_by_name(cuvid.c_str()); reduced == nullptr) {
          log->info("{} absent de FFmpeg : décodage logiciel en pleine taille", cuvid);
        } else if (auto opened_reduced = open_codec(*reduced, track, 1, options.cuvid_resize);
                   !opened_reduced) {
          log->info("{} refusé ({}) : décodage logiciel en pleine taille", cuvid,
                    opened_reduced.error().detail);
        } else {
          log->info("{} : images réduites à {}", cuvid, options.cuvid_resize);
          return {};
        }
      }
      return open_codec(*decoder, track, options.threads);
    }
    auto device = open_hw_device(id, attempt, options);
    if (!device) return std::unexpected(device.error());
    hw = std::move(*device);
    if (attempt == HwApi::Vulkan) {
      if (auto queue = vulkan_decode_queue(*hw.device, id); !queue) return std::unexpected(queue.error());
    }
    if (attempt == HwApi::Cuda && options.direct_surfaces > 0) {
      const std::int64_t pixels = std::int64_t{track.width} * track.height;
      direct = pixels <= kMaxDirectSurfacePixels;
      if (direct) {
        // B15-03 : l'aval en tient direct_surfaces et timed() une de plus (held) : sans ce +1, la surface de
        // trop ne se mapperait pas et l'image serait perdue.
        extra_hw_frames = std::max(extra_hw_frames, std::min(options.direct_surfaces + 1, kMaxExtraHwFrames));
      } else {
        log->info("NVDEC : copie de sécurité gardée, {}x{} au-delà de 1440p", track.width, track.height);
      }
    }
    return open_codec(*hw.codec, track, options.threads);
  }
};

namespace {

std::expected<void, Refusal> check_options(const TrackInfo& track, const DecoderOptions& options) {
  if (track.kind != TrackKind::Video || !track.parameters) {
    return refuse(RefusalCode::TrackUnavailable, std::format("piste {} : pas une piste vidéo", track.index));
  }
  if (options.attempts.empty()) return refuse(RefusalCode::FieldOutOfRange, "DecoderOptions.attempts vide");
  for (auto it = options.attempts.begin(); it != options.attempts.end(); ++it) {
    if (std::find(it + 1, options.attempts.end(), *it) != options.attempts.end()) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("DecoderOptions.attempts : {} en double", name(*it)));
    }
  }
  if (options.extra_hw_frames < 0 || options.extra_hw_frames > kMaxExtraHwFrames) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("DecoderOptions.extra_hw_frames = {}", options.extra_hw_frames));
  }
  if (options.direct_surfaces < 0 || options.direct_surfaces > kMaxExtraHwFrames) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("DecoderOptions.direct_surfaces = {}", options.direct_surfaces));
  }
  if (options.threads < 0 || options.threads > kMaxDecoderThreads) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("DecoderOptions.threads = {}", options.threads));
  }
  return {};
}

}  // namespace

VideoDecoder::VideoDecoder(std::unique_ptr<State> state) : state_(std::move(state)) {}
VideoDecoder::VideoDecoder(VideoDecoder&&) noexcept = default;
VideoDecoder& VideoDecoder::operator=(VideoDecoder&&) noexcept = default;
VideoDecoder::~VideoDecoder() = default;

std::expected<VideoDecoder, Refusal> VideoDecoder::open(const TrackInfo& track,
                                                        const DecoderOptions& options) {
  if (auto valid = check_options(track, options); !valid) return std::unexpected(valid.error());
  auto state = std::make_unique<State>();
  state->opened = track;
  state->requested_threads = options.threads;
  state->software_allowed = std::ranges::contains(options.attempts, HwApi::Software);
  state->download = options.download;
  state->full_host_formats = options.full_host_formats;
  state->host_422 = options.host_422;
  if (track.inverse_telecine) {
    state->deinterlacer = Deinterlacer(FieldMode::InverseTelecine);
  } else if (track.field_match) {
    state->deinterlacer = Deinterlacer(FieldMode::MatchFields);
  }
  if (track.frame_rate.num > 0 && track.frame_rate.den > 0) {
    state->nominal_duration_us = av_rescale(1'000'000, track.frame_rate.den, track.frame_rate.num);
  }
  // H.264 High 10 (Hi10P, les vieux fansubs), High 4:2:2 et 4:4:4 : aucun bloc vidéo (NVDEC, AMD,
  // Intel, Vulkan Video) ne les décode ; les essais matériels, qui s'ouvriraient puis reculeraient,
  // sont sautés quand le logiciel est permis.
  const AVCodecParameters& av = track.parameters->av();
  constexpr int kProfileMask = 0xFF;  // sans les bits Intra et Constraint
  const int h264_profile = av.profile == AV_PROFILE_UNKNOWN ? 0 : av.profile & kProfileMask;
  const bool hardware_hopeless =
      state->software_allowed && av.codec_id == AV_CODEC_ID_H264 &&
      (h264_profile == AV_PROFILE_H264_HIGH_10 || h264_profile == AV_PROFILE_H264_HIGH_422 ||
       h264_profile == AV_PROFILE_H264_HIGH_444_PREDICTIVE);
  // P3.25 : une piste entrelacée se décode en logiciel, où bwdif la désentrelace avant SPAN.
  const bool interlaced =
      state->software_allowed && av.field_order != AV_FIELD_UNKNOWN && av.field_order != AV_FIELD_PROGRESSIVE;
  // A15-01 : NVDEC rendrait une chroma pleine ou 4:2:2 que l'ingestion CUDA ne prend pas.
  const bool cuda_hopeless = state->software_allowed && chroma_beyond_420(av);
  for (const HwApi attempt : options.attempts) {
    if ((hardware_hopeless || interlaced) && attempt != HwApi::Software) {
      state->log->info("piste {} : {} sauté, {}", track.index, name(attempt),
                       interlaced ? "piste entrelacée" : "aucun décodeur matériel pour ce profil H.264");
      continue;
    }
    if (cuda_hopeless && attempt == HwApi::Cuda) {
      state->log->info("piste {} : NVDEC sauté, chroma au-delà du 4:2:0 (décodage logiciel, converti)",
                       track.index);
      continue;
    }
    auto opened = state->try_attempt(attempt, track, options);
    if (opened) {
      state->log->info("piste {} : {} décodé par {}", track.index, track.codec, name(attempt));
      if (state->direct) {
        state->log->info("NVDEC : surfaces rendues sans copie, {} images tenues au plus",
                         options.direct_surfaces);
      }
      return VideoDecoder(std::move(state));
    }
    state->log->info("piste {} : {} écarté : {}", track.index, name(attempt), opened.error().detail);
    state->rejected.push_back({attempt, opened.error()});
    state->hw = HwDevice{};
  }
  std::string detail = std::format("piste {} ({})", track.index, track.codec);
  for (const Attempt& attempt : state->rejected) detail += std::format(" ; {}", attempt.why.detail);
  return refuse(RefusalCode::DecoderUnavailable, std::move(detail));
}

HwApi VideoDecoder::api() const {
  return state_->api;
}
int VideoDecoder::threads() const {
  return state_->codec ? state_->codec->thread_count : 0;
}
std::span<const Attempt> VideoDecoder::rejected() const {
  return state_->rejected;
}
int VideoDecoder::data_errors() const {
  return state_->data_errors;
}
bool VideoDecoder::direct_surfaces() const {
  return state_->direct && state_->api == HwApi::Cuda;
}
int VideoDecoder::hw_surfaces() const {
  const AVBufferRef* frames = state_->codec->hw_frames_ctx;
  if (frames == nullptr) return 0;
  return reinterpret_cast<const AVHWFramesContext*>(frames->data)->initial_pool_size;
}
bool VideoDecoder::drained() const {
  return state_->drained;
}

// avcodec_send_packet prend le paquet même quand il rend AVERROR_INVALIDDATA :
// l'erreur peut venir d'une image précédente, décodée pendant l'envoi.
std::expected<void, Refusal> VideoDecoder::send(const Packet& packet) {
  State& s = *state_;
  if (s.pending) return refuse(RefusalCode::DecodeFailed, "paquet précédent encore en attente");
  if (!packet.av) return refuse(RefusalCode::DecodeFailed, "paquet sans données");
  if (s.wait_key) {  // P3.22 : rouvert en cours de flux, on repart d'une image clé
    if ((packet.av->flags & AV_PKT_FLAG_KEY) == 0) return {};
    s.wait_key = false;
  }
  if (!s.replay.empty()) {  // derrière les paquets rejoués
    s.replay.push_back(packet);
    return {};
  }
  const int err = s.submit(packet);
  if (s.reopen) {
    // Paquet reçu : il est dans le rejeu ; refusé par EAGAIN, il passe après les autres.
    if (err == AVERROR(EAGAIN)) s.pending = packet;
    if (s.reopen_software() || err == AVERROR(EAGAIN)) return {};
  }
  if (err == AVERROR(EAGAIN)) {
    s.pending = packet;
  } else if (err == AVERROR_INVALIDDATA) {
    s.data_error(std::format("envoi du paquet à {} µs", packet.pts_us));
  } else if (err < 0 && s.recover(err, "envoi d'un paquet")) {
    return {};
  } else if (err < 0) {
    return refuse_av(RefusalCode::DecodeFailed, std::format("envoi du paquet à {} µs", packet.pts_us), err);
  }
  return {};
}

std::expected<void, Refusal> VideoDecoder::send_end() {
  State& s = *state_;
  s.ended = true;
  if (!s.replay.empty()) {
    s.end_pending = true;
    return {};
  }
  const int err = avcodec_send_packet(s.codec.get(), nullptr);
  if (s.reopen && s.reopen_software()) return {};
  if (err < 0 && err != AVERROR_EOF) return refuse_av(RefusalCode::DecodeFailed, "fin du flux", err);
  return {};
}

std::expected<std::optional<DecodedFrame>, Refusal> VideoDecoder::receive() {
  State& s = *state_;
  if (auto resent = s.resend(); !resent) return std::unexpected(resent.error());
  if (s.pending && s.replay.empty()) {
    const int err = s.submit(*s.pending);
    if (s.reopen) {
      if (err != AVERROR(EAGAIN)) s.pending.reset();  // reçu : déjà dans sent
      if (s.reopen_software()) return std::nullopt;
    }
    if (err == AVERROR_INVALIDDATA) s.data_error("envoi d'un paquet en attente");
    if (err >= 0 || err == AVERROR_INVALIDDATA) s.pending.reset();
    if (err < 0 && err != AVERROR(EAGAIN) && err != AVERROR_INVALIDDATA) {
      if (s.recover(err, "envoi d'un paquet en attente")) return std::nullopt;
      return refuse_av(RefusalCode::DecodeFailed, "envoi d'un paquet en attente", err);
    }
  }
  // Une image retenue par bwdif (la fin du flux, ou une image d'avance) sort avant d'en décoder une autre.
  if (auto held = s.deinterlacer.pull(); !held) {
    return std::unexpected(held.error());
  } else if (*held) {
    return s.finish(std::move(*held));
  }
  FramePtr frame(av_frame_alloc());
  if (!frame) return refuse(RefusalCode::DecodeFailed, "image : mémoire insuffisante");
  // Avec des threads d'images, l'erreur d'une image abîmée sort ici, une fois par
  // image : chaque nouvel appel avance d'une image, au plus une par thread en vol.
  int err = avcodec_receive_frame(s.codec.get(), frame.get());
  for (int errors = 0; err == AVERROR_INVALIDDATA && errors < kMaxDecoderThreads; ++errors) {
    s.data_error("réception d'une image");
    err = avcodec_receive_frame(s.codec.get(), frame.get());
  }
  // L'image éventuelle du contexte basculé ressortira du rejeu.
  if (s.reopen && s.reopen_software()) return std::nullopt;
  if (err == AVERROR(EAGAIN)) return std::nullopt;
  if (err == AVERROR_EOF) {
    if (s.held) return s.deliver(std::move(s.held));  // la dernière, à la période moyenne
    if (s.deinterlacer.active() && !s.deinterlace_flushed) {
      s.deinterlace_flushed = true;
      if (auto flushed = s.deinterlacer.flush(); !flushed) return std::unexpected(flushed.error());
      if (auto last_frame = s.deinterlacer.pull(); !last_frame) {
        return std::unexpected(last_frame.error());
      } else if (*last_frame) {
        return s.finish(std::move(*last_frame));
      }
    }
    if (s.last_skipped) return s.deliver(std::move(s.last_skipped));
    s.drained = true;
    return std::nullopt;
  }
  if (err < 0 && s.recover(err, "réception d'une image")) return std::nullopt;
  if (err < 0) return refuse_av(RefusalCode::DecodeFailed, "réception d'une image", err);
  if (frame->hw_frames_ctx == nullptr) s.fall_back("le décodeur a rendu une image logicielle");
  // A15-02 : une image matérielle entrelacée (champ inconnu du conteneur) sortirait tissée, sans bwdif : le
  // décodage passe en logiciel, où elle est désentrelacée.
  if (frame->hw_frames_ctx != nullptr && (frame->flags & AV_FRAME_FLAG_INTERLACED) != 0 &&
      s.software_allowed && (s.output || s.replayable)) {
    s.fall_back("image entrelacée, désentrelacée en logiciel");
    if (s.reopen_software()) return std::nullopt;
  }
  if (!s.output) {
    s.output = true;
    s.sent.clear();
    s.sent.shrink_to_fit();
  }

  if (s.skipped(frame)) return std::nullopt;
  s.release_covering(*frame);
  return s.timed(std::move(frame));
}

// A15-09 : l'image sort quand la suivante arrive, avec l'écart de leurs pts pour durée.
std::expected<std::optional<DecodedFrame>, Refusal> VideoDecoder::State::timed(FramePtr frame) {
  FramePtr previous = std::exchange(held, std::move(frame));
  if (!previous) return std::nullopt;
  const std::int64_t from = previous->best_effort_timestamp;
  const std::int64_t to = held->best_effort_timestamp;
  const bool known = from != AV_NOPTS_VALUE && to != AV_NOPTS_VALUE && to > from;
  if (known && (from >= 0 || to <= std::numeric_limits<std::int64_t>::max() + from))
    previous->duration = to - from;
  return deliver(std::move(previous));
}

// Une image avant la cible de skip_before : son horodatage est fixé maintenant (le trou d'après se comble à
// partir d'elle), et elle reste la dernière écartée, la précédente étant alors passée pour de bon. B15-02 :
// la durée du paquet est la période moyenne du conteneur, fausse en cadence variable ; une image ne s'écarte
// donc que lorsque la suivante, de pts avant la cible aussi, prouve qu'elle ne couvre pas la cible (voir
// release_covering).
bool VideoDecoder::State::skipped(FramePtr& frame) {
  if (already_shown(*frame)) return true;
  if (!skip_before || deinterlacer.active() || (frame->flags & AV_FRAME_FLAG_INTERLACED) != 0) return false;
  const std::int64_t pts =
      frame->best_effort_timestamp == AV_NOPTS_VALUE ? next_pts_us : frame->best_effort_timestamp;
  if (pts == kNoPts || pts >= *skip_before) return false;
  const std::int64_t duration = frame->duration > 0 ? frame->duration : nominal_duration_us;
  frame->best_effort_timestamp = pts;
  next_pts_us = end_of(pts, std::max<std::int64_t>(duration, 1));
  last_skipped = std::move(frame);
  return true;
}

// B15-04 : le rejeu depuis l'image clé repasse sur des images déjà montrées, qui ne ressortent pas.
bool VideoDecoder::State::already_shown(const AVFrame& frame) {
  if (!catching_up) return false;
  const std::int64_t at = frame.best_effort_timestamp;
  if (at != AV_NOPTS_VALUE && delivered_us != kNoPts && at <= delivered_us) return true;
  catching_up = false;
  return false;
}

// La première image à ou après la cible arrive. La dernière écartée la couvre quand celle-ci est strictement
// après la cible : elle sort avant, avec pour durée l'écart de leurs pts (timed()). Sinon elle est jetée.
void VideoDecoder::State::release_covering(const AVFrame& next) {
  FramePtr previous = std::move(last_skipped);
  if (!previous || held || !skip_before) return;
  const std::int64_t pts =
      next.best_effort_timestamp == AV_NOPTS_VALUE ? next_pts_us : next.best_effort_timestamp;
  if (pts != kNoPts && pts > *skip_before) held = std::move(previous);
}

// La copie en mémoire hôte, le désentrelacement, puis l'horodatage.
std::expected<std::optional<DecodedFrame>, Refusal> VideoDecoder::State::deliver(FramePtr frame) {
  State& s = *this;
  // Les images d'un appareil partagé restent sur le GPU : leur propriétaire les lit en place.
  if (s.download && !s.hw.shared && frame->hw_frames_ctx != nullptr) {
    FramePtr host(av_frame_alloc());
    if (!host) return refuse(RefusalCode::DecodeFailed, "image : mémoire insuffisante");
    if (const int copied = av_hwframe_transfer_data(host.get(), frame.get(), 0); copied < 0)
      return refuse_av(RefusalCode::DecodeFailed, "copie de l'image matérielle en mémoire hôte", copied);
    if (const int props = av_frame_copy_props(host.get(), frame.get()); props < 0)
      return refuse_av(RefusalCode::DecodeFailed, "propriétés de l'image copiée", props);
    frame = std::move(host);
  }
  // P3.25 : une image entrelacée en mémoire hôte passe par bwdif, avec une image de retard.
  if (frame->hw_frames_ctx == nullptr &&
      (s.deinterlacer.active() || (frame->flags & AV_FRAME_FLAG_INTERLACED) != 0)) {
    if (!s.deinterlacer.active())
      s.log->info("images entrelacées : {}",
                  s.opened.inverse_telecine ? "télécinéma inversé (fieldmatch, decimate)"
                  : s.opened.field_match    ? "champs du même instant (fieldmatch, bwdif)"
                                            : "désentrelacement bwdif");
    if (auto pushed = s.deinterlacer.push(std::move(frame)); !pushed) return std::unexpected(pushed.error());
    auto out = s.deinterlacer.pull();
    if (!out) return std::unexpected(out.error());
    if (!*out) return std::nullopt;
    frame = std::move(*out);
  }
  return s.finish(std::move(frame));
}

void VideoDecoder::skip_before(std::optional<std::int64_t> target_us) {
  state_->skip_before = target_us;
  if (!target_us) state_->last_skipped.reset();
}

void VideoDecoder::flush() {
  State& s = *state_;
  s.skip_before.reset();
  s.last_skipped.reset();
  s.held.reset();
  avcodec_flush_buffers(s.codec.get());
  s.pending.reset();
  s.replay.clear();
  s.sent.clear();
  s.end_pending = false;
  s.ended = false;
  s.drained = false;
  s.next_pts_us = kNoPts;
  s.deinterlacer.reset();
  s.deinterlace_flushed = false;
  s.gop.clear();
  s.gop_bytes = 0;
  s.gop_valid = false;
  s.catching_up = false;
  s.delivered_us = kNoPts;
}

}  // namespace ajp::media
