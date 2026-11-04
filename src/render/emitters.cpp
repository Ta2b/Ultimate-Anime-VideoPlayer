#include "render/emitters.h"

#include <algorithm>
#include <cmath>

namespace ajp::render {
namespace {

float smooth(float lo, float hi, float x) {
  const float t = std::clamp((x - lo) / (hi - lo), 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}

}  // namespace

std::vector<float> emitter_map_v2(const EmitterImage& image, const EmitterParams& params) {
  const int w = image.width;
  const int h = image.height;
  const auto n = static_cast<std::size_t>(std::max(w, 0)) * static_cast<std::size_t>(std::max(h, 0));
  if (w <= 0 || h <= 0 || image.nits.size() != n || image.chroma.size() != n) return {};
  const int bw = (w + kEmitterBlock - 1) / kEmitterBlock;
  const int bh = (h + kEmitterBlock - 1) / kEmitterBlock;
  // Les moyennes des blocs : luminance et saturation.
  std::vector<float> level(static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh), 0.0F);
  std::vector<float> sat(level.size(), 0.0F);
  std::vector<int> count(level.size(), 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const auto b = static_cast<std::size_t>(y / kEmitterBlock) * static_cast<std::size_t>(bw) +
                     static_cast<std::size_t>(x / kEmitterBlock);
      level[b] += std::max(image.nits[image.at(x, y)], 0.0F);
      sat[b] += std::clamp(image.chroma[image.at(x, y)], 0.0F, 1.0F);
      ++count[b];
    }
  }
  for (std::size_t b = 0; b < level.size(); ++b) {
    level[b] /= static_cast<float>(count[b]);
    sat[b] /= static_cast<float>(count[b]);
  }
  // La couronne de chaque bloc : la moyenne des blocs à une distance de 2 à 4 ; les voisins immédiats,
  // souvent la même lumière, n'y entrent pas.
  std::vector<float> ring(level.size(), 0.0F);
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      double sum = 0.0;
      int k = 0;
      for (int j = -kEmitterRingOut; j <= kEmitterRingOut; ++j) {
        for (int i = -kEmitterRingOut; i <= kEmitterRingOut; ++i) {
          if (std::max(std::abs(i), std::abs(j)) < kEmitterRingIn) continue;
          const int x = std::clamp(bx + i, 0, bw - 1);
          const int y = std::clamp(by + j, 0, bh - 1);
          sum +=
              level[static_cast<std::size_t>(y) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(x)];
          ++k;
        }
      }
      ring[static_cast<std::size_t>(by) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(bx)] =
          static_cast<float>(sum / k);
    }
  }
  const float white = std::max(params.white_nits, 1.0F);
  std::vector<float> out(n, 0.0F);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const auto b = static_cast<std::size_t>(y / kEmitterBlock) * static_cast<std::size_t>(bw) +
                     static_cast<std::size_t>(x / kEmitterBlock);
      const float pixel = std::max(image.nits[image.at(x, y)], 0.0F);
      const float block = std::max(level[b], 1e-3F);
      // Centre-couronne, en diaphragmes : une lumière domine son entourage d'au moins un diaphragme.
      const float stops = std::log2(block / std::max(ring[b], 1e-3F));
      const float iso = smooth(0.5F, 1.5F, stops);
      // Le niveau : proche du blanc, ou une lumière saturée moins claire (néon).
      const float white_gate = smooth(0.55F * white, 0.95F * white, block);
      const float colour_gate = smooth(0.30F * white, 0.60F * white, block) * smooth(0.45F, 0.75F, sat[b]);
      const float gate = std::max(white_gate, colour_gate);
      // La forme d'une étincelle : le pixel domine son bloc d'un diaphragme et demi, et approche le blanc.
      const float spark =
          smooth(1.0F, 1.5F, std::log2(std::max(pixel, 1e-3F) / block)) * smooth(0.6F * white, white, pixel);
      out[image.at(x, y)] = std::clamp(std::max(iso * gate, spark), 0.0F, 1.0F);
    }
  }
  return out;
}

}  // namespace ajp::render
