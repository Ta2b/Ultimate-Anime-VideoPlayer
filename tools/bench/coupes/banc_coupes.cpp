// Banc de la découpe en plans (media/shot_cutter.h), hors du lecteur : du RGB24 de 256 × 144 sur l'entrée
// standard (ffmpeg), mesuré comme la pré-analyse (la grille de kernels/spec/frame_stats.h, 128 × 72, par
// moyenne de 2 × 2), poussé dans ShotCutter. Écrit le numéro de la première image de chaque plan, une par
// ligne, sauf le premier.
//   banc_coupes [images par seconde, 24000/1001 par défaut] [sans-flux]
// Construit avec KIRARI_BANC_FLUX (une version de shot_cutter.h qui lit ShotFrame::motion), chaque image
// passe aussi par le flux optique de NVIDIA sur sa luma (media/probe_flow.h), comme dans la pré-analyse ;
// « sans-flux » l'éteint, pour vérifier qu'il ne change rien sans lui. banc_coupes.sh le construit pour deux
// versions de shot_cutter.cpp et compare leurs coupes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "kernels/ref/frame_stats.h"
#include "media/shot_cutter.h"
#ifdef KIRARI_BANC_FLUX
#include "media/probe_flow.h"
#endif

int main(int argc, char** argv) {
  using namespace kirari;
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);  // en mode texte, la lecture s'arrêtait au premier octet 0x1A
#endif
  const double fps = argc > 1 ? std::atof(argv[1]) : 24000.0 / 1001.0;
  constexpr int kW = kernels::frame_stats::kGridW;
  constexpr int kH = kernels::frame_stats::kGridH;
  constexpr int kInW = 2 * kW;
  constexpr int kInH = 2 * kH;
  static_assert(kInW == 256 && kInH == 144);
  std::vector<unsigned char> pixels(static_cast<std::size_t>(kInW * kInH * 3));
  media::ShotCutter cutter;
#ifdef KIRARI_BANC_FLUX
  std::unique_ptr<media::ProbeFlow> flow;
  if (!(argc > 2 && std::strcmp(argv[2], "sans-flux") == 0)) {
    auto opened = media::ProbeFlow::open();
    if (!opened) {
      std::fprintf(stderr, "flux optique refusé : %s\n", opened.error().c_str());
      return 2;
    }
    flow = std::move(*opened);
  }
  long long judged = 0;
#endif
  const auto print = [&](const std::optional<media::Shot>& shot) {
    if (shot && shot->start_us > 0)
      std::printf("%lld\n", std::llround(static_cast<double>(shot->start_us) * fps / 1e6));
  };
  const auto at = [&](int x, int y, int c) {
    return static_cast<float>(pixels.at(static_cast<std::size_t>(((y * kInW) + x) * 3 + c)));
  };
  long long count = 0;
  while (std::fread(pixels.data(), 1, pixels.size(), stdin) == pixels.size()) {
    kernels::RgbImage rgb(kW, kH);
    for (int y = 0; y < kH; ++y)
      for (int x = 0; x < kW; ++x)
        for (int c = 0; c < 3; ++c) {
          const float sum = at(2 * x, 2 * y, c) + at(2 * x + 1, 2 * y, c) + at(2 * x, 2 * y + 1, c) +
                            at(2 * x + 1, 2 * y + 1, c);
          rgb.planes.at(static_cast<std::size_t>(c)).at(x, y) = sum / (4.0F * 255.0F);
        }
    auto counts = kernels::ref::frame_counts(rgb, kernels::FrameTransfer::Sdr);
    if (!counts) return 1;
    const auto pts = std::llround(static_cast<double>(count) * 1e6 / fps);
    media::ShotFrame frame = media::shot_frame_of(pts, kernels::ref::frame_stats_of(*counts));
#ifdef KIRARI_BANC_FLUX
    if (flow) {
      media::ProbeLuma luma(static_cast<std::size_t>(kInW * kInH));
      for (int y = 0; y < kInH; ++y)
        for (int x = 0; x < kInW; ++x)  // la luma BT.709 du signal, comme le Y de la source
          luma[static_cast<std::size_t>(y * kInW + x)] = static_cast<std::uint8_t>(
              std::lround((0.2126F * at(x, y, 0)) + (0.7152F * at(x, y, 1)) + (0.0722F * at(x, y, 2))));
      frame.motion = flow->push(std::move(luma));
      if (auto failure = flow->take_failure()) {
        std::fprintf(stderr, "flux optique arrêté : %s\n", failure->c_str());
        return 2;
      }
      if (frame.motion) {
        ++judged;
        // Le détail par image, pour régler les seuils : image, part sans correspondance, part texturée.
        std::fprintf(stderr, "flux %lld %.3f %.3f\n", count, frame.motion->unmatched, frame.motion->textured);
      }
    }
#endif
    print(cutter.push(frame));
    ++count;
  }
  for (const media::Shot& shot : cutter.finish(std::llround(static_cast<double>(count) * 1e6 / fps)))
    print(shot);
#ifdef KIRARI_BANC_FLUX
  std::fprintf(stderr, "%lld images, %lld jugées par le flux\n", count, judged);
#else
  std::fprintf(stderr, "%lld images\n", count);
#endif
  return 0;
}
