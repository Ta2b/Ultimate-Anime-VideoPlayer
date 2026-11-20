// Les calques et la plaque de décor rejoués sur processeur (calques.h).
#include "calques.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/ref/layers.h"
#include "kernels/ref/warp.h"

namespace carte_rejeu {
namespace {

using kirari::kernels::LayerParams;
using kirari::kernels::PlaneF;
using kirari::kernels::RgbImage;
namespace play = kirari::app::play;
namespace ref = kirari::kernels::ref;

PlaneF to_float(const kirari::kernels::Plane<std::uint8_t>& p) {
  PlaneF f(p.width, p.height);
  for (std::size_t i = 0; i < p.data.size(); ++i) f.data[i] = static_cast<float>(p.data[i]) / 255.0F;
  return f;
}

RgbImage gray(const PlaneF& p) {
  RgbImage image;
  image.planes = {p, p, p};
  return image;
}

// Lecture bilinéaire bornée à l'image, comme ref::compose_layers ; Catmull-Rom avec cubic.
float bilinear(const PlaneF& p, float x, float y) {
  x = std::clamp(x, 0.0F, static_cast<float>(p.width - 1));
  y = std::clamp(y, 0.0F, static_cast<float>(p.height - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, p.width - 1);
  const int y1 = std::min(y0 + 1, p.height - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = p.at(x0, y0) * (1.0F - fx) + p.at(x1, y0) * fx;
  const float bottom = p.at(x0, y1) * (1.0F - fx) + p.at(x1, y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

// Ce que la garde fait d'une composition : la part du décor qu'elle refuse, l'écart médian sur le cœur du
// décor (masque d'au moins kCore), et par pixel, pour les vues, ce qui est repris et ce qui est refusé.
struct Guard {
  double refused = -1.0;
  double median = -1.0;
  PlaneF taken;
  PlaneF denied;
};

constexpr float kCore = 0.5F;
// La lecture du lecteur par défaut (cubic_warp, settings.h) : Catmull-Rom.
constexpr bool kCubic = true;
constexpr int kBins = 2550;  // dixièmes de code

double median_of(const std::vector<std::uint64_t>& bins, std::uint64_t count) {
  std::uint64_t seen = 0;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    seen += bins[i];
    if (2 * seen >= count) return static_cast<double>(i) / 10.0;
  }
  return -1.0;
}

// Un point de la source i : sa valeur, s'il est dans l'image, et le masque de la source (la plaque).
struct Sample {
  float value = 0.0F;
  bool inside = false;
  float mask = 1.0F;
};

Sample sample(const LayerParams& p, std::size_t i, const PlaneF& source, float x, float y) {
  const auto& s = p.sources[i].shift;
  const float qx = x + (s[0] + s[1] * x + s[2] * y);
  const float qy = y + (s[3] + s[4] * x + s[5] * y);
  Sample out;
  out.inside = qx >= 0.0F && qy >= 0.0F && qx <= static_cast<float>(source.width - 1) &&
               qy <= static_cast<float>(source.height - 1);
  if (!out.inside) return out;
  if (i == 0 && p.source_grid.cols != 0) out.mask = ref::layer_mask(p.source_grid, qx, qy);
  out.value = p.cubic ? ref::catmull_rom_sample(source, qx, qy) : bilinear(source, qx, qy);
  return out;
}

// Les étapes 1 à 4 de spec/layers.h sur la luma : s0 et s1 les sources, here l'image sous la composition.
Guard guard_of(const LayerParams& p, const PlaneF& s0, const PlaneF& s1, const PlaneF& here, bool keep) {
  Guard g;
  if (keep) {
    g.taken = PlaneF(here.width, here.height);
    g.denied = PlaneF(here.width, here.height);
  }
  std::vector<std::uint64_t> bins(kBins + 1, 0);
  std::uint64_t core = 0;
  double weight = 0.0;
  double refused = 0.0;
  for (int y = 0; y < here.height; ++y) {
    for (int x = 0; x < here.width; ++x) {
      const auto fx = static_cast<float>(x);
      const auto fy = static_cast<float>(y);
      const float m = ref::layer_mask(p.grid, fx, fy);
      if (m <= 0.0F) continue;
      const Sample a = sample(p, 0, s0, fx, fy);
      const Sample b = p.count == 2 ? sample(p, 1, s1, fx, fy) : Sample{};
      const float w = (a.inside ? a.mask : 0.0F) + (b.inside ? 1.0F : 0.0F);
      if (w <= 0.0F) continue;  // rien à reprendre : l'image reste
      bool measured = true;
      float d = 0.0F;
      if (p.count == 2 && a.inside && b.inside) {
        d = std::fabs(a.value - b.value);
      } else if (p.count == 1) {
        d = std::fabs(a.value - here.at(x, y));
      } else {
        measured = false;  // une seule des deux dedans : reprise sans garde
      }
      const float k = 1.0F - std::clamp((d - p.guard_low) / (p.guard_high - p.guard_low), 0.0F, 1.0F);
      const float mw = m * (p.count == 1 ? a.mask : 1.0F);
      weight += mw;
      refused += mw * (1.0F - k);
      if (measured && m >= kCore && (p.count == 2 || a.mask >= kCore)) {
        ++bins[static_cast<std::size_t>(std::min(kBins, static_cast<int>(std::lround(d * 2550.0F))))];
        ++core;
      }
      if (keep) {
        g.taken.at(x, y) = mw * k;
        g.denied.at(x, y) = mw * (1.0F - k);
      }
    }
  }
  if (weight > 0.0) g.refused = refused / weight;
  if (core > 0) g.median = median_of(bins, core);
  return g;
}

std::uint8_t byte(float v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0F, 255.0F)));
}

// À gauche left, à droite base teintée : bleu repris, rouge refusé.
bool write_view(const std::string& path, const PlaneF& left, const PlaneF& base, const Guard& g) {
  constexpr std::array<float, 3> kTaken{40.0F, 110.0F, 255.0F};
  constexpr std::array<float, 3> kDenied{255.0F, 50.0F, 40.0F};
  const int w = left.width;
  const int h = left.height;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(2 * w) * static_cast<std::size_t>(h) * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(2 * w);
      const std::size_t l = (row + static_cast<std::size_t>(x)) * 3;
      const std::size_t r = (row + static_cast<std::size_t>(w + x)) * 3;
      const float at = 0.45F * g.taken.at(x, y);
      const float ad = 0.6F * g.denied.at(x, y);
      const float v = 255.0F * base.at(x, y);
      for (std::size_t c = 0; c < 3; ++c) {
        rgb[l + c] = byte(255.0F * left.at(x, y));
        rgb[r + c] = byte(v * (1.0F - at - ad) + kTaken[c] * at + kDenied[c] * ad);
      }
    }
  }
  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << 2 * w << ' ' << h << "\n255\n";
  out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  return static_cast<bool>(out);
}

// L'image composée (ref::compose_layers) sur under, avec la source 0 et la source 1.
std::optional<PlaneF> composed(const LayerParams& p, const PlaneF& under, const PlaneF& s0,
                               const PlaneF& s1) {
  RgbImage out = gray(under);
  if (!ref::compose_layers(out, gray(s0), gray(s1), p)) return std::nullopt;
  return std::move(out.planes[0]);
}

}  // namespace

LayerReplay::LayerReplay(int divisor, std::string views, int every, bool dump_grid)
    : divisor_(divisor), views_(std::move(views)), every_(every), dump_grid_(dump_grid) {}

bool LayerReplay::pair(const kirari::kernels::Plane<std::uint8_t>& a8,
                       const kirari::kernels::Plane<std::uint8_t>& b8,
                       const kirari::kernels::SceneMotion& motion,
                       const std::optional<play::NeighbourCamera>& late) {
  const int w = a8.width;
  const int h = a8.height;
  // Comme LayerStage::arrive : la caméra lk d'une paire d'avant quand celle de la paire manque.
  neighbours_.next();
  bool borrowed = false;
  const auto planned =
      play::plan_with_neighbour(motion, neighbours_.for_pair(), w * divisor_, h * divisor_, borrowed);
  if (!planned && planned.error() == play::LayerSkip::Cut) neighbours_.forget();
  if (const auto own = play::refined_camera(motion)) neighbours_.seen(*own);
  if (late) neighbours_.seen(*late);
  const PlaneF a = to_float(a8);
  const PlaneF b = to_float(b8);
  const bool view = !views_.empty() && every_ > 0 && index_ % every_ == 0;
  bool ok = true;
  Guard pair_guard;
  double speed = -1.0;
  if (planned) {
    const auto& c = planned->camera;
    if (dump_grid_) {
      std::cout << std::format("grille {} {} {} {}", index_, planned->cols, planned->rows, planned->tile_px);
      for (const auto value : c) std::cout << std::format(" {:.10f}", value);
      std::cout << ' ';
      for (const auto decor : planned->decor) std::cout << (decor != 0 ? '1' : '0');
      std::cout << '\n';
    }
    const double cx = (static_cast<double>(planned->work_width) - 1.0) / 2.0;
    const double cy = (static_cast<double>(planned->work_height) - 1.0) / 2.0;
    speed = std::hypot(c[0] + c[1] * cx + c[2] * cy - cx, c[3] + c[4] * cx + c[5] * cy - cy);
    if (const auto params = play::pair_params(*planned, w, h, 0.5, kCubic)) {
      PlaneF blend(w, h);
      for (std::size_t i = 0; i < blend.data.size(); ++i) blend.data[i] = 0.5F * (a.data[i] + b.data[i]);
      pair_guard = guard_of(*params, a, b, blend, view);
      if (view) {
        const auto image = composed(*params, blend, a, b);
        ok = image &&
             write_view(std::format("{}/calques_{:04d}.ppm", views_, index_), *image, blend, pair_guard);
      }
    }
  }
  const play::PlateStep step = plate_.step(planned, true);
  Guard plate_guard;
  if (step.action == play::PlateAction::Refresh) {
    key_ = b;
  } else if (step.action == play::PlateAction::Use) {
    if (const auto params = play::plate_params(plate_.key(), *planned, step.key_to_b, w, h, kCubic)) {
      plate_guard = guard_of(*params, key_, key_, b, view);
      if (view) {
        const auto image = composed(*params, b, key_, key_);
        ok = ok && image &&
             write_view(std::format("{}/plaque_{:04d}.ppm", views_, index_), *image, b, plate_guard);
      }
    }
  }
  const std::string_view reason = planned ? std::string_view("-") : play::name(planned.error());
  std::cout << std::format("calques {} {:.4f} {:.4f} {:.1f} {:.3f} {} {:.4f} {:.1f} {}\n",
                           planned ? (borrowed ? 2 : 1) : 0, planned ? planned->share : 0.0,
                           pair_guard.refused, pair_guard.median, speed, static_cast<int>(step.action),
                           plate_guard.refused, plate_guard.median, reason);
  ++index_;
  return ok;
}

}  // namespace carte_rejeu
