// La mesure d'une image décodée par la pré-analyse des plans (measure_frame, media/shot_probe.h) : la grille
// de l'analyse lue en R'G'B', puis ses comptes (kernels/ref/frame_stats.h).
#include <algorithm>
#include <cmath>
#include <optional>
#include <span>

#include "kernels/ref/frame_stats.h"
#include "media/shot_probe.h"
#include "media/yuv_sample.h"

namespace ajp::media {
namespace {

// Les formats que la pré-analyse lit : ceux de yuv::layout_of (4:2:0), plus P016 (NVDEC en 12 bits) et
// Yuv444p16 (B42-10). Le décodeur logiciel de l'analyse ramène déjà le 4:2:2, le 4:4:4 et le 12 bits au 4:2:0
// (frame_data.cpp, to_chain_yuv). chroma_step : 2 en 4:2:0, 1 en 4:4:4.
struct ProbeLayout {
  yuv::Layout layout;
  int chroma_step = 2;
};

std::optional<ProbeLayout> probe_layout(PixelFormat format) {
  if (const std::optional<yuv::Layout> layout = yuv::layout_of(format)) return ProbeLayout{.layout = *layout};
  if (format == PixelFormat::P016)
    return ProbeLayout{.layout = {.bits = 16, .semi_planar = true, .high = true, .shift = 0}};
  if (format == PixelFormat::Yuv444p16)
    return ProbeLayout{.layout = {.bits = 16, .high = true}, .chroma_step = 1};
  return std::nullopt;
}

}  // namespace

std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame) {
  namespace fs = kernels::frame_stats;
  const std::optional<ProbeLayout> probe = probe_layout(frame.format);
  if (!probe) return refuse(RefusalCode::FormatUnsupported, "format de pixel non pris par la pré-analyse");
  const yuv::Layout* layout = &probe->layout;
  const int step = probe->chroma_step;
  if (frame.data == nullptr || frame.data->memory() != FrameMemory::Host)
    return refuse(RefusalCode::FormatUnsupported, "image décodée hors de la mémoire hôte");
  if (frame.width < 2 || frame.height < 2) return refuse(RefusalCode::FormatUnsupported, "image trop petite");
  const bool bt601 = frame.color.matrix == ColorMatrix::Bt601;
  const bool bt2020 = frame.color.matrix == ColorMatrix::Bt2020Ncl;
  const double kr = bt601 ? 0.299 : bt2020 ? 0.2627 : 0.2126;
  const double kb = bt601 ? 0.114 : bt2020 ? 0.0593 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const bool full = frame.color.range == ColorRange::Full;
  const double scale = std::ldexp(1.0, layout->bits - 8);
  const double y_off = full ? 0.0 : 16.0 * scale;
  const double y_range = (full ? 255.0 : 219.0) * scale;
  const double c_range = (full ? 255.0 : 224.0) * scale;
  const double c_mid = 128.0 * scale;
  const auto luma_plane = frame.data->host_plane(0);
  const std::size_t luma_stride = frame.data->host_stride(0);
  const auto u_plane = frame.data->host_plane(1);
  const std::size_t u_stride = frame.data->host_stride(1);
  std::span<const std::byte> v_plane;
  std::size_t v_stride = 0;
  if (!layout->semi_planar) {
    v_plane = frame.data->host_plane(2);
    v_stride = frame.data->host_stride(2);
  }
  // Les points de la grille de l'analyse (kernels/spec/frame_stats.h), convertis en R'G'B' : la
  // référence lit alors chaque point de cette image à sa place. Un point par case, pas une moyenne de bloc
  // (B42-11) : c'est le contrat de frame_stats.h, que suit aussi le noyau GPU de l'analyse à l'ingestion ;
  // les deux mesures restent comparables plan par plan.
  kernels::RgbImage rgb(fs::kGridW, fs::kGridH);
  for (int gy = 0; gy < fs::kGridH; ++gy) {
    const int sy = std::min((2 * gy + 1) * frame.height / (2 * fs::kGridH), frame.height - 1);
    for (int gx = 0; gx < fs::kGridW; ++gx) {
      const int sx = std::min((2 * gx + 1) * frame.width / (2 * fs::kGridW), frame.width - 1);
      const double yy = (yuv::sample(luma_plane, luma_stride, sx, sy, *layout) - y_off) / y_range;
      double cb = 0.0, cr = 0.0;
      if (layout->semi_planar) {
        cb = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout, 0, 2);
        cr = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout, 1, 2);
      } else {
        cb = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout);
        cr = yuv::sample(v_plane, v_stride, sx / step, sy / step, *layout);
      }
      cb = (cb - c_mid) / c_range;
      cr = (cr - c_mid) / c_range;
      // V se calcule avant de borner R et B (A42-07) : un canal hors plage le faussait.
      const double r = yy + 2.0 * (1.0 - kr) * cr;
      const double b = yy + 2.0 * (1.0 - kb) * cb;
      const double g = (yy - kr * r - kb * b) / kg;
      rgb.planes[0].at(gx, gy) = static_cast<float>(std::clamp(r, 0.0, 1.0));
      rgb.planes[1].at(gx, gy) = static_cast<float>(std::clamp(g, 0.0, 1.0));
      rgb.planes[2].at(gx, gy) = static_cast<float>(std::clamp(b, 0.0, 1.0));
    }
  }
  const kernels::FrameTransfer transfer = frame.color.transfer == Transfer::Pq ? kernels::FrameTransfer::Pq
                                          : frame.color.transfer == Transfer::Hlg
                                              ? kernels::FrameTransfer::Hlg
                                              : kernels::FrameTransfer::Sdr;
  auto counts = kernels::ref::frame_counts(rgb, transfer);
  if (!counts) return refuse(RefusalCode::FormatUnsupported, counts.error().detail);
  return shot_frame_of(frame.pts_us, kernels::ref::frame_stats_of(*counts), transfer);
}

}  // namespace ajp::media
