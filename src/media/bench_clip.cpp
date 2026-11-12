#include "media/bench_clip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <system_error>
#include <vector>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "media/av_util.h"

namespace ajp::media {
namespace {

struct Rgb {
  std::uint8_t r, g, b;
};

// Une teinte de la scène numéro `scene` : couleurs plates, comme un aplat d'animation.
Rgb palette(int scene, int index) {
  constexpr Rgb kColors[]{{232, 92, 120},  {80, 150, 230},  {250, 210, 90}, {96, 200, 140},
                          {150, 100, 220}, {245, 245, 240}, {30, 34, 52},   {255, 140, 60}};
  return kColors[static_cast<std::size_t>((scene * 3 + index) % 8)];
}

// Un grain d'image fixe par pixel et par image (hachage, sans état) : ±5 codes de luma, comme le grain
// d'une source réelle ; RIFE et le décodeur n'ont plus d'aplats parfaits.
int grain(int x, int y, int index) {
  std::uint32_t h = static_cast<std::uint32_t>(x) * 73856093U ^ static_cast<std::uint32_t>(y) * 19349663U ^
                    static_cast<std::uint32_t>(index) * 83492791U;
  h ^= h >> 13;
  h *= 0x5bd1e995U;
  h ^= h >> 15;
  return static_cast<int>(h % 11U) - 5;
}

void put(AVFrame& frame, int x, int y, Rgb c, int noise = 0) {
  // BT.709 limité, YUV 4:2:0 : Y partout, U et V au pixel pair.
  const float r = c.r;
  const float g = c.g;
  const float b = c.b;
  const auto luma = static_cast<std::uint8_t>(
      std::clamp(16.0F + 0.1826F * r + 0.6142F * g + 0.0620F * b + static_cast<float>(noise), 16.0F, 235.0F));
  frame.data[0][y * frame.linesize[0] + x] = luma;
  if (((x | y) & 1) == 0) {
    frame.data[1][(y / 2) * frame.linesize[1] + x / 2] = static_cast<std::uint8_t>(
        std::clamp(128.0F - 0.1006F * r - 0.3386F * g + 0.4392F * b, 16.0F, 240.0F));
    frame.data[2][(y / 2) * frame.linesize[2] + x / 2] = static_cast<std::uint8_t>(
        std::clamp(128.0F + 0.4392F * r - 0.3989F * g - 0.0403F * b, 16.0F, 240.0F));
  }
}

Rgb scaled(Rgb c, float k) {
  return {static_cast<std::uint8_t>(std::min(255.0F, static_cast<float>(c.r) * k)),
          static_cast<std::uint8_t>(std::min(255.0F, static_cast<float>(c.g) * k)),
          static_cast<std::uint8_t>(std::min(255.0F, static_cast<float>(c.b) * k))};
}

// Trois sortes de plans de deux secondes, en boucle : calme (le mode auto monte haut), combat rapide (il doit
// redescendre à temps), combat dans le noir (dégradés sombres, là où le FP8 et le débandage souffrent).
// Le passage d'un plan calme à un plan rapide est ce qui fait perdre des images à un contrôleur en retard.
void draw_frame(AVFrame& frame, int width, int height, int index, int fps) {
  const int scene = index / (fps * 2);  // une coupe toutes les deux secondes
  const int t = index % (fps * 2);      // l'instant dans le plan
  const int kind = scene % 3;           // 0 calme, 1 rapide, 2 rapide et sombre
  const bool fast = kind != 0;
  const float dark = kind == 2 ? 0.22F : 1.0F;
  // Le fond défile : 3 px par image au calme, 40 en combat (un travelling très vif en 1080p).
  const int pan = (t * (fast ? 40 : 3) + scene * 517) % 8192;
  const int band = std::max(1, height / 9);
  const float fh = static_cast<float>(height);
  for (int y = 0; y < height; ++y) {
    const float shade = dark * (0.70F + 0.30F * static_cast<float>(y) / fh);  // dégradé vertical
    for (int x = 0; x < width; ++x) {
      const int column = ((x + pan) / (width / 6)) % 4;
      Rgb color = scaled(palette(scene, column), shade);
      // Hachures fines (cheveux, herbe) : de la haute fréquence au contour franc, qui défile aussi.
      if (((x + y * 2 + pan / 3) % 14) < 2 && (y / band) % 2 == 0) color = scaled(palette(scene, 5), dark);
      put(frame, x, y, color, grain(x, y, index));
    }
  }
  const float phase = static_cast<float>(t) / static_cast<float>(fps * 2);
  // Deux personnages : un aplat arrondi à contour sombre ; en combat ils traversent l'écran en une seconde
  // (80 px par image), se croisent et rebondissent, avec occlusion de l'un par l'autre.
  for (int who = 0; who < 2; ++who) {
    const float travel = fast ? 2.0F * phase : 0.25F * phase;  // largeurs parcourues dans le plan
    float u = std::fmod(travel + 0.37F * static_cast<float>(who), 2.0F);
    if (u > 1.0F) u = 2.0F - u;  // aller-retour
    const float cx = static_cast<float>(width) * (who == 0 ? u : 1.0F - u);
    const float cy = fh * (who == 0 ? 0.45F : 0.60F) +
                     (fast ? 140.0F : 40.0F) *
                         std::sin(6.2832F * (phase * (fast ? 3.0F : 1.0F) + 0.5F * static_cast<float>(who)));
    const float rx = static_cast<float>(width) * (who == 0 ? 0.09F : 0.07F);
    const float ry = fh * (who == 0 ? 0.22F : 0.18F);
    const int x0 = std::max(0, static_cast<int>(cx - rx - 6));
    const int x1 = std::min(width - 1, static_cast<int>(cx + rx + 6));
    const int y0 = std::max(0, static_cast<int>(cy - ry - 6));
    const int y1 = std::min(height - 1, static_cast<int>(cy + ry + 6));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float dx = (static_cast<float>(x) - cx) / rx;
        const float dy = (static_cast<float>(y) - cy) / ry;
        const float d = dx * dx + dy * dy;
        if (d <= 1.0F)
          put(frame, x, y, scaled(palette(scene + who + 1, 1 + who), std::max(dark, 0.35F)),
              grain(x, y, index));
        else if (d <= 1.12F)
          put(frame, x, y, {20, 20, 28});
      }
    }
  }
  // Une lame qui tourne (un coup d'épée) : du mouvement qui n'est pas une translation, le plus dur pour RIFE.
  if (fast) {
    const float angle = 6.2832F * phase * 2.5F;
    const float ca = std::cos(angle);
    const float sa = std::sin(angle);
    const float ox = static_cast<float>(width) * 0.5F;
    const float oy = fh * 0.5F;
    const float length = fh * 0.42F;
    const int reach = static_cast<int>(length) + 4;
    for (int y = std::max(0, static_cast<int>(oy) - reach);
         y < std::min(height, static_cast<int>(oy) + reach); ++y) {
      for (int x = std::max(0, static_cast<int>(ox) - reach);
           x < std::min(width, static_cast<int>(ox) + reach); ++x) {
        const float px = static_cast<float>(x) - ox;
        const float py = static_cast<float>(y) - oy;
        const float along = px * ca + py * sa;
        const float across = -px * sa + py * ca;
        if (along >= 0.0F && along <= length && std::abs(across) <= 7.0F)
          put(frame, x, y, scaled({235, 240, 255}, std::max(dark * 1.6F, 0.6F)));
      }
    }
  }
}

std::expected<void, Refusal> write_clip_to(const std::filesystem::path& file, const BenchClipSpec& spec) {
  // MPEG-2 (P3.12) : le seul encodeur propre à FFmpeg que les décodeurs matériels lisent (NVDEC, D3D11VA,
  // VA-API) ; le MPEG-4 d'avant passait par le processeur, qui n'est pas le chemin de la lecture.
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MPEG2VIDEO);
  if (codec == nullptr) return refuse(RefusalCode::LibraryUnavailable, "encodeur MPEG-2 absent de FFmpeg");
  AVFormatContext* raw = nullptr;
  const std::string path = core::to_utf8(file);
  if (const int err = avformat_alloc_output_context2(&raw, nullptr, "matroska", path.c_str());
      err < 0 || raw == nullptr)
    return refuse_av(RefusalCode::LibraryUnavailable, "conteneur matroska", err);
  struct OutputFree {
    void operator()(AVFormatContext* c) const {
      if (c != nullptr && !(c->oformat->flags & AVFMT_NOFILE) && c->pb != nullptr) avio_closep(&c->pb);
      avformat_free_context(c);
    }
  };
  std::unique_ptr<AVFormatContext, OutputFree> out(raw);
  CodecContextPtr encoder(avcodec_alloc_context3(codec));
  if (!encoder) return refuse(RefusalCode::LibraryUnavailable, "contexte d'encodeur");
  encoder->width = spec.width;
  encoder->height = spec.height;
  encoder->time_base = AVRational{1, spec.fps};
  encoder->framerate = AVRational{spec.fps, 1};
  encoder->pix_fmt = AV_PIX_FMT_YUV420P;
  encoder->bit_rate = 30'000'000;  // le grain demande plus que les aplats
  encoder->gop_size = spec.fps;
  encoder->max_b_frames = 0;
  encoder->colorspace = AVCOL_SPC_BT709;
  encoder->color_primaries = AVCOL_PRI_BT709;
  encoder->color_trc = AVCOL_TRC_BT709;
  encoder->color_range = AVCOL_RANGE_MPEG;
  if (out->oformat->flags & AVFMT_GLOBALHEADER) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  if (const int err = avcodec_open2(encoder.get(), codec, nullptr); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "avcodec_open2", err);
  AVStream* stream = avformat_new_stream(out.get(), nullptr);
  if (stream == nullptr) return refuse(RefusalCode::LibraryUnavailable, "flux de sortie");
  stream->time_base = encoder->time_base;
  stream->avg_frame_rate = encoder->framerate;
  if (const int err = avcodec_parameters_from_context(stream->codecpar, encoder.get()); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "paramètres du flux", err);
  if (!(out->oformat->flags & AVFMT_NOFILE)) {
    if (const int err = avio_open(&out->pb, path.c_str(), AVIO_FLAG_WRITE); err < 0)
      return refuse_av(RefusalCode::FileUnwritable, path, err);
  }
  if (const int err = avformat_write_header(out.get(), nullptr); err < 0)
    return refuse_av(RefusalCode::FileUnwritable, "en-tête", err);

  FramePtr frame(av_frame_alloc());
  PacketPtr packet(av_packet_alloc());
  if (!frame || !packet) return refuse(RefusalCode::LibraryUnavailable, "mémoire FFmpeg");
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = spec.width;
  frame->height = spec.height;
  if (const int err = av_frame_get_buffer(frame.get(), 0); err < 0)
    return refuse_av(RefusalCode::LibraryUnavailable, "image", err);
  const auto drain = [&]() -> std::expected<void, Refusal> {
    while (true) {
      const int got = avcodec_receive_packet(encoder.get(), packet.get());
      if (got == AVERROR(EAGAIN) || got == AVERROR_EOF) return {};
      if (got < 0) return refuse_av(RefusalCode::DecodeFailed, "encodage", got);
      av_packet_rescale_ts(packet.get(), encoder->time_base, stream->time_base);
      packet->stream_index = stream->index;
      if (const int err = av_interleaved_write_frame(out.get(), packet.get()); err < 0)
        return refuse_av(RefusalCode::FileUnwritable, "écriture", err);
    }
  };
  const int total = spec.fps * spec.seconds;
  for (int i = 0; i < total; ++i) {
    if (const int err = av_frame_make_writable(frame.get()); err < 0)
      return refuse_av(RefusalCode::LibraryUnavailable, "image", err);
    draw_frame(*frame, spec.width, spec.height, i, spec.fps);
    frame->pts = i;
    if (const int err = avcodec_send_frame(encoder.get(), frame.get()); err < 0)
      return refuse_av(RefusalCode::DecodeFailed, "envoi", err);
    if (auto ok = drain(); !ok) return ok;
  }
  avcodec_send_frame(encoder.get(), nullptr);
  if (auto ok = drain(); !ok) return ok;
  if (const int err = av_write_trailer(out.get()); err < 0)
    return refuse_av(RefusalCode::FileUnwritable, "fin", err);
  return {};
}

}  // namespace

// Écrit à côté puis renomme (P2.20) : un clip interrompu (arrêt, disque plein) ne reste jamais sous son nom,
// où il servirait à toutes les calibrations suivantes.
std::expected<void, Refusal> write_bench_clip(const std::filesystem::path& file, const BenchClipSpec& spec) {
  std::filesystem::path partial = file;
  partial += ".partial";
  std::error_code error;
  auto written = write_clip_to(partial, spec);
  if (written && std::filesystem::file_size(partial, error) == 0 && !error)
    written = refuse(RefusalCode::FileUnwritable, "clip de calibration vide");
  // Sur le disque avant le renommage (C36-08) : une coupure juste après ne laisse pas un clip tronqué sous
  // son nom définitif, suivi par toutes les calibrations.
  if (written)
    if (const std::error_code synced = core::sync_file_retrying(partial))
      written = refuse(RefusalCode::FileUnwritable,
                       std::format("{} : {}", core::to_utf8(partial), synced.message()));
  if (written) std::filesystem::rename(partial, file, error);
  if (written && error)
    written =
        refuse(RefusalCode::FileUnwritable, std::format("{} : {}", core::to_utf8(file), error.message()));
  if (!written) std::filesystem::remove(partial, error);
  return written;
}

}  // namespace ajp::media
