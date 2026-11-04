#include "kernels/ref/grade.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>

#include "kernels/ref/check.h"
#include "kernels/spec/grade_pixel.h"

namespace ajp::kernels::ref {

namespace g = grade;

namespace {

int clamp_index(int v, int n) {
  return std::clamp(v, 0, n - 1);
}

// Le code d'un pixel aux indices bornés.
struct ImageFetch {
  const RgbImage* image;
  g::V3 operator()(int x, int y) const {
    const int cx = clamp_index(x, image->width());
    const int cy = clamp_index(y, image->height());
    return {image->planes[0].at(cx, cy), image->planes[1].at(cx, cy), image->planes[2].at(cx, cy)};
  }
};

// Un texel d'une carte de `comps` composantes, aux indices bornés ; les composantes absentes valent 0.
struct MapFetch {
  const float* data;
  int w;
  int h;
  int comps;
  g::V3 operator()(int x, int y) const {
    const std::size_t at = (static_cast<std::size_t>(clamp_index(y, h)) * static_cast<std::size_t>(w) +
                            static_cast<std::size_t>(clamp_index(x, w))) *
                           static_cast<std::size_t>(comps);
    return {data[at], comps > 1 ? data[at + 1] : 0.0F, comps > 2 ? data[at + 2] : 0.0F};
  }
};

// La lecture trilinéaire de la pile, comme la texture LINEAR du hook : pos = (c (n − 1) + 0,5) / n.
struct LookFetch {
  const GradeLook* look;
  g::V3 operator()(g::V3 c) const {
    const int n = look->size;
    if (n < 2) return c;
    const auto texel = [this, n](int r, int gr, int b) {
      const std::size_t at =
          ((static_cast<std::size_t>(b) * static_cast<std::size_t>(n) + static_cast<std::size_t>(gr)) *
               static_cast<std::size_t>(n) +
           static_cast<std::size_t>(r)) *
          3;
      return g::V3{look->rgb[at], look->rgb[at + 1], look->rgb[at + 2]};
    };
    const float scale = static_cast<float>(n - 1);
    const float xr = g::gclamp(c.r, 0.0F, 1.0F) * scale;
    const float xg = g::gclamp(c.g, 0.0F, 1.0F) * scale;
    const float xb = g::gclamp(c.b, 0.0F, 1.0F) * scale;
    const int r0 = std::min(static_cast<int>(xr), n - 2);
    const int g0 = std::min(static_cast<int>(xg), n - 2);
    const int b0 = std::min(static_cast<int>(xb), n - 2);
    const float tr = xr - static_cast<float>(r0);
    const float tg = xg - static_cast<float>(g0);
    const float tb = xb - static_cast<float>(b0);
    const g::V3 c00 = g::vmix(texel(r0, g0, b0), texel(r0 + 1, g0, b0), tr);
    const g::V3 c10 = g::vmix(texel(r0, g0 + 1, b0), texel(r0 + 1, g0 + 1, b0), tr);
    const g::V3 c01 = g::vmix(texel(r0, g0, b0 + 1), texel(r0 + 1, g0, b0 + 1), tr);
    const g::V3 c11 = g::vmix(texel(r0, g0 + 1, b0 + 1), texel(r0 + 1, g0 + 1, b0 + 1), tr);
    return g::vmix(g::vmix(c00, c10, tg), g::vmix(c01, c11, tg), tb);
  }
};

Result<void> check_look(const GradeLook& look) {
  if (look.size == 0 && look.rgb.empty()) return {};
  const auto n = static_cast<std::size_t>(look.size);
  if (look.size < 2 || look.size > 65 || look.rgb.size() != n * n * n * 3)
    return fail(ErrorCode::SizeMismatch,
                std::format("look : {} valeurs pour {} points par axe", look.rgb.size(), look.size));
  return {};
}

Result<void> check_image(const RgbImage& image, std::string_view what) {
  for (const PlaneF& plane : image.planes) {
    if (auto ok = check_plane(plane, what); !ok) return ok;
    if (plane.width != image.width() || plane.height != image.height())
      return fail(ErrorCode::SizeMismatch, std::format("{} : plans de tailles différentes", what));
  }
  if (image.width() < 2 || image.height() < 2)
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("{} : {}x{} sous 2x2", what, image.width(), image.height()));
  return {};
}

Rgb to_rgb(g::V3 v) {
  return {.r = v.r, .g = v.g, .b = v.b};
}

}  // namespace

GradeInput sdr_input(float white_nits, float gamma) {
  GradeInput input;
  for (auto& table : input.nits) {
    table.resize(g::kInputTablePoints);
    for (int i = 0; i < g::kInputTablePoints; ++i)
      table[static_cast<std::size_t>(i)] =
          white_nits * std::pow(static_cast<float>(i) / static_cast<float>(g::kInputTablePoints - 1), gamma);
  }
  input.matrix = {0.6274040F, 0.3292820F, 0.0433136F, 0.0690970F, 0.9195400F,
                  0.0113612F, 0.0163916F, 0.0880132F, 0.8955950F};
  return input;
}

Result<RgbImage> apply_input(const RgbImage& sdr, const GradeInput& input) {
  if (auto ok = check_image(sdr, "apply_input"); !ok) return std::unexpected(ok.error());
  for (const auto& table : input.nits)
    if (table.size() != static_cast<std::size_t>(g::kInputTablePoints))
      return fail(ErrorCode::SizeMismatch, std::format("apply_input : table de {} points", table.size()));
  std::vector<float> nits;
  nits.reserve(3 * static_cast<std::size_t>(g::kInputTablePoints));
  for (const auto& table : input.nits) nits.insert(nits.end(), table.begin(), table.end());
  RgbImage out(sdr.width(), sdr.height());
  for (int y = 0; y < sdr.height(); ++y)
    for (int x = 0; x < sdr.width(); ++x) {
      const g::V3 pq = g::input_code({sdr.planes[0].at(x, y), sdr.planes[1].at(x, y), sdr.planes[2].at(x, y)},
                                     nits.data(), input.matrix.data());
      out.planes[0].at(x, y) = pq.r;
      out.planes[1].at(x, y) = pq.g;
      out.planes[2].at(x, y) = pq.b;
    }
  return out;
}

Result<GradeMaps> grade_maps(const RgbImage& pq, const GradeUniforms& u, const GradeLook& look,
                             std::span<const float> previous, float dt_s) {
  if (auto ok = check_image(pq, "grade_maps"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_look(look); !ok) return std::unexpected(ok.error());
  const int w = pq.width();
  const int h = pq.height();
  GradeMaps maps;
  maps.local_w = g::map_size(w, g::kLocalDiv);
  maps.local_h = g::map_size(h, g::kLocalDiv);
  maps.area_w = g::map_size(maps.local_w, g::kAreaDiv);
  maps.area_h = g::map_size(maps.local_h, g::kAreaDiv);
  const auto local_n = static_cast<std::size_t>(maps.local_w) * static_cast<std::size_t>(maps.local_h);
  const auto area_n = static_cast<std::size_t>(maps.area_w) * static_cast<std::size_t>(maps.area_h);
  if (!previous.empty() && previous.size() != area_n)
    return fail(ErrorCode::SizeMismatch,
                std::format("grade_maps : historique de {} texels pour {}", previous.size(), area_n));

  const ImageFetch fetch{&pq};
  maps.local.resize(local_n * 3);
  for (int ly = 0; ly < maps.local_h; ++ly)
    for (int lx = 0; lx < maps.local_w; ++lx) {
      const g::V3 v = g::local_texel(fetch, w, h, lx, ly, maps.local_w, maps.local_h, u);
      const std::size_t at = (static_cast<std::size_t>(ly) * static_cast<std::size_t>(maps.local_w) +
                              static_cast<std::size_t>(lx)) *
                             3;
      maps.local[at] = v.r;
      maps.local[at + 1] = v.g;
      maps.local[at + 2] = v.b;
    }

  // Zone : moyenne 4 × 4 aux indices bornés, flou horizontal puis vertical (même ordre de sommes que le
  // shader), puis le lissage.
  const MapFetch local{maps.local.data(), maps.local_w, maps.local_h, 3};
  const int aw = maps.area_w;
  const int ah = maps.area_h;
  std::vector<float> mean(area_n);
  for (int qy = 0; qy < ah; ++qy)
    for (int qx = 0; qx < aw; ++qx)
      mean[static_cast<std::size_t>(qy) * static_cast<std::size_t>(aw) + static_cast<std::size_t>(qx)] =
          g::area_mean(local, maps.local_w, maps.local_h, qx, qy, aw, ah);
  const auto at_mean = [&](int x, int y) {
    return mean[static_cast<std::size_t>(clamp_index(y, ah)) * static_cast<std::size_t>(aw) +
                static_cast<std::size_t>(clamp_index(x, aw))];
  };
  std::vector<float> horizontal(area_n);
  for (int y = 0; y < ah; ++y)
    for (int x = 0; x < aw; ++x) {
      float acc = at_mean(x, y) * g::kBlurWeights[0];
      for (int k = 1; k <= g::kBlurRadius; ++k)
        acc += (at_mean(x - k, y) + at_mean(x + k, y)) * g::kBlurWeights[static_cast<std::size_t>(k)];
      horizontal[static_cast<std::size_t>(y) * static_cast<std::size_t>(aw) + static_cast<std::size_t>(x)] =
          acc;
    }
  const auto at_h = [&](int x, int y) {
    return horizontal[static_cast<std::size_t>(clamp_index(y, ah)) * static_cast<std::size_t>(aw) +
                      static_cast<std::size_t>(x)];
  };
  maps.area.resize(area_n);
  maps.history.assign(area_n, 0.0F);
  for (int y = 0; y < ah; ++y)
    for (int x = 0; x < aw; ++x) {
      float value = at_h(x, y) * g::kBlurWeights[0];
      for (int k = 1; k <= g::kBlurRadius; ++k)
        value += (at_h(x, y - k) + at_h(x, y + k)) * g::kBlurWeights[static_cast<std::size_t>(k)];
      const std::size_t idx =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(aw) + static_cast<std::size_t>(x);
      if (idx < static_cast<std::size_t>(g::kHistoryTexels)) {
        const float kept = previous.empty() ? 0.0F : previous[idx];
        value = g::area_smooth(value, kept, dt_s);
        maps.history[idx] = value + 1.0F;
      }
      maps.area[idx] = value;
    }

  // Max local : le pixel le plus clair du bloc, gradé avec ses cartes et le look.
  maps.localmax.assign(local_n, 0.0F);
  if (u.black_hi > u.black_lo) {
    const MapFetch area{maps.area.data(), aw, ah, 1};
    const LookFetch lut{&look};
    for (int ly = 0; ly < maps.local_h; ++ly)
      for (int lx = 0; lx < maps.local_w; ++lx) {
        int px = 0;
        int py = 0;
        g::localmax_pick(fetch, w, h, lx, ly, maps.local_w, maps.local_h, &px, &py);
        const float s = (static_cast<float>(px) + 0.5F) / static_cast<float>(w);
        const float t = (static_cast<float>(py) + 0.5F) / static_cast<float>(h);
        g::V3 e = g::grade_code(fetch(px, py), g::skin_area(fetch, px, py, u),
                                g::bilinear(local, maps.local_w, maps.local_h, s, t).r,
                                g::bilinear(area, aw, ah, s, t).r, u);
        e = g::look_code(e, lut, u);
        maps.localmax[static_cast<std::size_t>(ly) * static_cast<std::size_t>(maps.local_w) +
                      static_cast<std::size_t>(lx)] = g::max3(e);
      }
  }
  return maps;
}

Rgb grade_pixel(const Rgb& code, float skin, float local, float area, float black_local,
                const GradeUniforms& u, const GradeLook& look) {
  const LookFetch lut{&look};
  return to_rgb(g::grade_pixel(g::V3{code.r, code.g, code.b}, skin, local, area, black_local, u, lut));
}

Result<RgbImage> grade_image(const RgbImage& pq, const GradeMaps& maps, const GradeUniforms& u,
                             const GradeLook& look) {
  if (auto ok = check_image(pq, "grade_image"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_look(look); !ok) return std::unexpected(ok.error());
  const int w = pq.width();
  const int h = pq.height();
  if (maps.local_w != g::map_size(w, g::kLocalDiv) || maps.local_h != g::map_size(h, g::kLocalDiv) ||
      maps.local.size() !=
          static_cast<std::size_t>(maps.local_w) * static_cast<std::size_t>(maps.local_h) * 3 ||
      maps.localmax.size() * 3 != maps.local.size() ||
      maps.area.size() != static_cast<std::size_t>(maps.area_w) * static_cast<std::size_t>(maps.area_h))
    return fail(ErrorCode::SizeMismatch,
                std::format("grade_image : cartes d'une autre image que {}x{}", w, h));
  const ImageFetch fetch{&pq};
  const MapFetch local{maps.local.data(), maps.local_w, maps.local_h, 3};
  const MapFetch area{maps.area.data(), maps.area_w, maps.area_h, 1};
  const MapFetch localmax{maps.localmax.data(), maps.local_w, maps.local_h, 1};
  const LookFetch lut{&look};
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float s = (static_cast<float>(x) + 0.5F) / static_cast<float>(w);
      const float t = (static_cast<float>(y) + 0.5F) / static_cast<float>(h);
      const g::V3 e = g::grade_pixel(fetch(x, y), g::skin_area(fetch, x, y, u),
                                     g::bilinear(local, maps.local_w, maps.local_h, s, t).r,
                                     g::bilinear(area, maps.area_w, maps.area_h, s, t).r,
                                     g::bilinear(localmax, maps.local_w, maps.local_h, s, t).r, u, lut);
      out.planes[0].at(x, y) = e.r;
      out.planes[1].at(x, y) = e.g;
      out.planes[2].at(x, y) = e.b;
    }
  return out;
}

}  // namespace ajp::kernels::ref
