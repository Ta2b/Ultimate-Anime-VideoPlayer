#include "gpu/span_fp8/forward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/ref/half.h"

namespace ajp::gpu::span_fp8 {
namespace {

// Un tenseur de travail, canaux en dernier : (y, x, c).
struct Map {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::vector<float> values;

  Map(int h, int w, int c)
      : height(h),
        width(w),
        channels(c),
        values(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * static_cast<std::size_t>(c)) {}
  float* at(int y, int x) {
    return values.data() +
           (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels);
  }
  const float* at(int y, int x) const {
    return values.data() +
           (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels);
  }
};

// Une convolution du paquet en f32, rangée [sortie][position][entrée] sur les seules entrées réelles.
struct Dense {
  int outs = 0;
  int side = 1;  // 1 ou 3
  int inputs = 0;
  std::vector<float> weights;
  std::vector<float> bias;
};

// Le paquet range [sortie][ligne][colonne][groupe × stored] ; on garde les real premières entrées de chaque
// groupe et les outs premières sorties.
Dense dense(const Conv& conv, int outs, int side, int groups, int stored, int real) {
  Dense d{.outs = outs, .side = side, .inputs = groups * real};
  const int taps = side * side;
  d.weights.reserve(static_cast<std::size_t>(outs) * static_cast<std::size_t>(taps) *
                    static_cast<std::size_t>(d.inputs));
  for (int o = 0; o < outs; ++o)
    for (int t = 0; t < taps; ++t)
      for (int g = 0; g < groups; ++g)
        for (int c = 0; c < real; ++c) {
          const std::size_t from =
              (static_cast<std::size_t>(o * taps + t) * static_cast<std::size_t>(groups) +
               static_cast<std::size_t>(g)) *
                  static_cast<std::size_t>(stored) +
              static_cast<std::size_t>(c);
          d.weights.push_back(kernels::ref::half_to_float(conv.weights[from]));
        }
  d.bias.assign(conv.bias.begin(), conv.bias.begin() + outs);
  return d;
}

Map convolve(const Map& in, const Dense& d, bool relu) {
  Map out(in.height, in.width, d.outs);
  const int r = d.side / 2;
  const auto n = static_cast<std::size_t>(d.inputs);
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < in.width; ++x) {
      float* o = out.at(y, x);
      for (int oc = 0; oc < d.outs; ++oc) {
        float acc = d.bias[static_cast<std::size_t>(oc)];
        const float* w = d.weights.data() + static_cast<std::size_t>(oc * d.side * d.side) * n;
        for (int ky = 0; ky < d.side; ++ky)
          for (int kx = 0; kx < d.side; ++kx, w += n) {
            const int yy = y + ky - r, xx = x + kx - r;
            if (yy < 0 || xx < 0 || yy >= in.height || xx >= in.width) continue;
            const float* p = in.at(yy, xx);
            for (std::size_t c = 0; c < n; ++c) acc += w[c] * p[c];
          }
        o[oc] = relu ? std::max(acc, 0.0F) : acc;
      }
    }
  return out;
}

// Dégroupage 2×2 des trois plans : canal c·4 + dy·2 + dx.
Map unshuffle(std::span<const float> rgb, int height, int width) {
  Map out(height / 2, width / 2, kUnshuffled);
  for (int y = 0; y < out.height; ++y)
    for (int x = 0; x < out.width; ++x)
      for (int c = 0; c < 3; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            out.at(y, x)[c * 4 + dy * 2 + dx] =
                rgb[(static_cast<std::size_t>(c) * static_cast<std::size_t>(height) +
                     static_cast<std::size_t>(2 * y + dy)) *
                        static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(2 * x + dx)];
  return out;
}

// Fin de bloc : (x + c3)·tanh(att), att la convolution 1×1 de c3.
Map block_end(const Map& x, const Map& c3, const Map& att) {
  Map out(x.height, x.width, x.channels);
  for (std::size_t i = 0; i < out.values.size(); ++i)
    out.values[i] = (x.values[i] + c3.values[i]) * std::tanh(att.values[i]);
  return out;
}

Map concat(const std::vector<const Map*>& parts) {
  int channels = 0;
  for (const Map* p : parts) channels += p->channels;
  Map out(parts[0]->height, parts[0]->width, channels);
  for (int y = 0; y < out.height; ++y)
    for (int x = 0; x < out.width; ++x) {
      float* o = out.at(y, x);
      for (const Map* p : parts) o = std::copy_n(p->at(y, x), p->channels, o);
    }
  return out;
}

// DepthToSpace(4, CRD) : canal c·16 + by·4 + bx vers le pixel (4y + by, 4x + bx) du plan c.
std::vector<float> depth_to_space(const Map& pixel) {
  const int h = 4 * pixel.height, w = 4 * pixel.width;
  std::vector<float> out(std::size_t{3} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w));
  for (int y = 0; y < pixel.height; ++y)
    for (int x = 0; x < pixel.width; ++x)
      for (int c = 0; c < 3; ++c)
        for (int by = 0; by < 4; ++by)
          for (int bx = 0; bx < 4; ++bx)
            out[(static_cast<std::size_t>(c) * static_cast<std::size_t>(h) +
                 static_cast<std::size_t>(4 * y + by)) *
                    static_cast<std::size_t>(w) +
                static_cast<std::size_t>(4 * x + bx)] = pixel.at(y, x)[c * 16 + by * 4 + bx];
  return out;
}

}  // namespace

std::vector<float> forward(const Package& package, std::span<const float> rgb, int height, int width) {
  if (height <= 0 || width <= 0 || height % 2 != 0 || width % 2 != 0 ||
      rgb.size() != std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width) ||
      package.blocks.empty())
    return {};
  const int ch = package.real_channels;
  const Map first = convolve(unshuffle(rgb, height, width),
                             dense(package.first, ch, 3, 1, kUnshuffledPadded, kUnshuffled), false);
  Map x = first;
  Map first_block(0, 0, 0), last_relu(0, 0, 0);
  for (std::size_t b = 0; b < package.blocks.size(); ++b) {
    const Block& block = package.blocks[b];
    Map r1 = convolve(x, dense(block.conv[0], ch, 3, 1, kChannels, ch), true);
    const Map r2 = convolve(r1, dense(block.conv[1], ch, 3, 1, kChannels, ch), true);
    const Map c3 = convolve(r2, dense(block.conv[2], ch, 3, 1, kChannels, ch), false);
    const Map att = convolve(c3, dense(block.attention, ch, 1, 1, kChannels, ch), false);
    x = block_end(x, c3, att);
    if (b == 0) first_block = x;
    if (b + 1 == package.blocks.size()) last_relu = std::move(r1);
  }
  const Map cat = convolve(concat({&first, &x, &first_block, &last_relu}),
                           dense(package.concat, ch, 1, 4, kChannels, ch), false);
  return depth_to_space(convolve(cat, dense(package.pixel, kPixelChannels, 3, 1, kChannels, ch), false));
}

int exact_side(int side, int blocks) {
  return std::max(0, 4 * (side / 2 - (2 + 3 * blocks)));
}

}  // namespace ajp::gpu::span_fp8
