#include "media/video_decoder.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <format>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/cpu.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mathematics.h>
}

#include "media/av_util.h"
#include "media/frame_data.h"
#include "media/hw_device.h"
#include "media/log.h"

namespace ajp::media {

namespace {

// Threads du décodage logiciel : la valeur demandée, ou un par cœur
// (affinité comprise) jusqu'à kAutoDecoderThreads. FFmpeg en prendrait
// jusqu'à 16, soit 15 images de retard avant la première.
int software_threads(int requested) {
  if (requested > 0) return requested;
  return std::clamp(av_cpu_count(), 1, kAutoDecoderThreads);
}

}  // namespace

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
  int extra_hw_frames = 0;  // de l'essai en cours
  bool direct = false;      // NVDEC sans copie de sécurité (DecoderOptions::direct_surfaces)
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
  std::shared_ptr<spdlog::logger> log = make_logger();

  // Données abîmées : l'image en cause est perdue, le décodage continue.
  void data_error(std::string_view where) {
    ++data_errors;
    log->warn("données invalides ({}) : image perdue, le décodage continue", where);
  }

  void fall_back(std::string_view why) {
    if (api == HwApi::Software) return;
    log->warn("{} : {} ; décodage logiciel", name(api), why);
    rejected.push_back(
        {api, Refusal{RefusalCode::HwDecodeUnavailable, std::format("{} : {}", name(api), why)}});
    api = HwApi::Software;
  }

  // Choisit le format matériel s'il est proposé. extra_hw_frames se pose ici,
  // pour ce seul format : sous NVDEC il n'augmente que ulNumDecodeSurfaces
  // (nvdec.c:416), le pool CUDA allouant ses images à la demande.
  static AVPixelFormat get_format(AVCodecContext* ctx, const AVPixelFormat* formats) {
    auto& self = *static_cast<State*>(ctx->opaque);
    if (self.hw.device) {
      for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == self.hw.format) {
          ctx->extra_hw_frames = self.extra_hw_frames;
          return *format;
        }
      }
      self.fall_back("format matériel non proposé par get_format (profil, taille ou pilote)");
      if (!self.software_allowed) return AV_PIX_FMT_NONE;
      self.reopen = !self.output && self.replayable;
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
    return err;
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
    replay.assign(sent.begin(), sent.end());
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

  std::expected<void, Refusal> open_codec(const AVCodec& decoder, const TrackInfo& track, int threads) {
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
    if (const int err = avcodec_open2(ctx.get(), &decoder, nullptr); err < 0) {
      return refuse_av(RefusalCode::DecoderUnavailable, std::format("{} : avcodec_open2", name(api)), err);
    }
    codec = std::move(ctx);
    return {};
  }

  std::expected<void, Refusal> try_attempt(HwApi attempt, const TrackInfo& track,
                                           const DecoderOptions& options) {
    const AVCodecID id = track.parameters->av().codec_id;
    api = attempt;
    hw = HwDevice{};
    direct = false;
    extra_hw_frames = options.extra_hw_frames;
    if (attempt == HwApi::Software) {
      const AVCodec* decoder = avcodec_find_decoder(id);
      if (decoder == nullptr) {
        return refuse(RefusalCode::DecoderUnavailable,
                      std::format("aucun décodeur {}", avcodec_get_name(id)));
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
        extra_hw_frames = std::max(extra_hw_frames, options.direct_surfaces);
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
  if (track.frame_rate.num > 0 && track.frame_rate.den > 0) {
    state->nominal_duration_us = av_rescale(1'000'000, track.frame_rate.den, track.frame_rate.num);
  }
  for (const HwApi attempt : options.attempts) {
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
      return refuse_av(RefusalCode::DecodeFailed, "envoi d'un paquet en attente", err);
    }
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
    s.drained = true;
    return std::nullopt;
  }
  if (err < 0) return refuse_av(RefusalCode::DecodeFailed, "réception d'une image", err);
  if (frame->hw_frames_ctx == nullptr) s.fall_back("le décodeur a rendu une image logicielle");
  if (!s.output) {
    s.output = true;
    s.sent.clear();
    s.sent.shrink_to_fit();
  }

  // Un trou d'horodatage se comble par l'image précédente plus sa durée. Sans image
  // précédente (début, ou après un flush), il reste inconnu : kNoPts, jamais un 0 inventé.
  const std::int64_t pts =
      frame->best_effort_timestamp == AV_NOPTS_VALUE ? s.next_pts_us : frame->best_effort_timestamp;
  const std::int64_t duration = frame->duration > 0 ? frame->duration : s.nominal_duration_us;
  s.next_pts_us = pts == kNoPts ? kNoPts : pts + duration;

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
  ReadyEvent ready;
  if (s.hw.stream && frame->hw_frames_ctx != nullptr) {
    auto recorded = s.hw.stream->record();
    if (!recorded) return std::unexpected(recorded.error());
    ready = std::move(*recorded);
  }
  return to_decoded_frame(std::move(frame), pts, duration, std::move(ready));
}

void VideoDecoder::flush() {
  State& s = *state_;
  avcodec_flush_buffers(s.codec.get());
  s.pending.reset();
  s.replay.clear();
  s.sent.clear();
  s.end_pending = false;
  s.ended = false;
  s.drained = false;
  s.next_pts_us = kNoPts;
}

}  // namespace ajp::media
