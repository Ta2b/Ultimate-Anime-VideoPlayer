// Banc de la découpe en plans (media/shot_cutter.h), hors du lecteur : du RGB24 de 128 × 72 sur l'entrée
// standard (ffmpeg), mesuré comme la pré-analyse (la grille de kernels/spec/frame_stats.h), poussé dans
// ShotCutter. Écrit le numéro de la première image de chaque plan, une par ligne, sauf le premier.
//   banc_coupes [images par seconde, 24000/1001 par défaut]
// banc_coupes.sh le construit pour deux versions de shot_cutter.cpp et compare leurs coupes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include "kernels/ref/frame_stats.h"
#include "media/shot_cutter.h"

int main(int argc, char** argv) {
  using namespace ajp;
  const double fps = argc > 1 ? std::atof(argv[1]) : 24000.0 / 1001.0;
  constexpr int kW = kernels::frame_stats::kGridW;
  constexpr int kH = kernels::frame_stats::kGridH;
  std::vector<unsigned char> pixels(static_cast<std::size_t>(kW * kH * 3));
  media::ShotCutter cutter;
  const auto print = [&](const std::optional<media::Shot>& shot) {
    if (shot && shot->start_us > 0)
      std::printf("%lld\n", std::llround(static_cast<double>(shot->start_us) * fps / 1e6));
  };
  long long count = 0;
  while (std::fread(pixels.data(), 1, pixels.size(), stdin) == pixels.size()) {
    kernels::RgbImage rgb(kW, kH);
    for (int y = 0; y < kH; ++y)
      for (int x = 0; x < kW; ++x)
        for (int c = 0; c < 3; ++c)
          rgb.planes.at(static_cast<std::size_t>(c)).at(x, y) =
              static_cast<float>(pixels.at(static_cast<std::size_t>(((y * kW) + x) * 3 + c))) / 255.0F;
    auto counts = kernels::ref::frame_counts(rgb, kernels::FrameTransfer::Sdr);
    if (!counts) return 1;
    const auto pts = std::llround(static_cast<double>(count) * 1e6 / fps);
    print(cutter.push(media::shot_frame_of(pts, kernels::ref::frame_stats_of(*counts))));
    ++count;
  }
  for (const media::Shot& shot : cutter.finish(std::llround(static_cast<double>(count) * 1e6 / fps)))
    print(shot);
  std::fprintf(stderr, "%lld images\n", count);
  return 0;
}
