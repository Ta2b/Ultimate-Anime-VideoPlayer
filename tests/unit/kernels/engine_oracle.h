// Oracle de parité : transcription littérale, en C++ scalaire, des noyaux de
// animejanai-inference-rtx@118c272 :
//   src/resample.h:31-129 (spline36, bilinear, compute, chroma_shifts, make_csp),
//   src/kernels.cu:87-97 (mirr, quant), 101-165 (k_uv_h, k_v_f32, k_pre_combine),
//   169-219 (k_rife_uv_h, k_rife_pre_combine), 221-313 (k_post_matrix, k_h_f32, k_uv_v_store,
//   k_post444), 334-386 (k_pre444, k_pre444_pad), 632-644 (k_rife_consts),
//   700-724 (k_scd_diff, terme par pixel), 927-949 (k_block_diff), 963-1096 (warp).
// Les boucles gardent l'ordre, les types et les indices du CUDA ; un fil devient une itération,
// la chroma reste entrelacée comme en NV12 et P010, les tenseurs fp16 sont des float.
// La référence de src/kernels/ref/, restructurée, doit lui être égale au bit près
// (parity_test.cpp). Ne pas « améliorer » ce fichier : il fixe le comportement du moteur.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace oracle {

using u16 = std::uint16_t;

inline std::size_t sz(int i) {
  return static_cast<std::size_t>(i);
}

// ---- src/resample.h ----

inline double spline36(double x) {
  x = std::fabs(x);
  if (x < 1.0) return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
  if (x < 2.0) {
    x -= 1.0;
    return ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x;
  }
  if (x < 3.0) {
    x -= 2.0;
    return ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x;
  }
  return 0.0;
}

inline double bilinear(double x) {
  x = std::fabs(x);
  return x < 1.0 ? 1.0 - x : 0.0;
}

struct pass {
  int taps = 0, src = 0, dst = 0;
  std::vector<int> start;
  std::vector<float> wt;
};

inline pass compute(int src, int dst, double shift, bool bilinear_filter) {
  const double scale = static_cast<double>(dst) / src;
  const double step = scale < 1.0 ? scale : 1.0;
  double (*f)(double) = bilinear_filter ? bilinear : spline36;
  const double support = bilinear_filter ? 1.0 : 3.0;
  int taps = static_cast<int>(std::ceil(support / step)) * 2;
  if (taps < 1) taps = 1;
  pass w;
  w.taps = taps;
  w.src = src;
  w.dst = dst;
  w.start.resize(sz(dst));
  w.wt.resize(sz(dst) * sz(taps));
  std::vector<double> row(sz(taps));
  for (int i = 0; i < dst; i++) {
    const double pos = (i + 0.5) / scale + shift;
    const double begin = std::floor(pos - taps / 2.0 + 0.5) + 0.5;
    double sum = 0.0;
    for (int j = 0; j < taps; j++) {
      row[sz(j)] = f((begin + j - pos) * step);
      sum += row[sz(j)];
    }
    w.start[sz(i)] = static_cast<int>(begin - 0.5);
    for (int j = 0; j < taps; j++) w.wt[sz(i) * sz(taps) + sz(j)] = static_cast<float>(row[sz(j)] / sum);
  }
  return w;
}

enum { SITING_LEFT = 1, SITING_CENTER = 2, SITING_TOPLEFT = 3 };

inline void chroma_shifts(int siting, bool up, double* sx, double* sy) {
  if (up) {
    *sx = siting == SITING_CENTER ? 0.0 : 0.25;
    *sy = siting == SITING_TOPLEFT ? 0.25 : 0.0;
  } else {
    *sx = siting == SITING_CENTER ? 0.0 : -0.5;
    *sy = siting == SITING_TOPLEFT ? -0.5 : 0.0;
  }
}

struct csp {
  float kr, kb, yoff, yscale, coff, cscale;
};

enum { FMT_NV12 = 1, FMT_P010 = 2, FMT_YUV444P16 = 3 };
enum { MATRIX_BT601 = 1, MATRIX_BT709 = 2, MATRIX_BT2020 = 3 };

inline csp make_csp(int format, int matrix, bool full) {
  csp c = {};
  switch (matrix) {
    case MATRIX_BT601:
      c.kr = 0.299f;
      c.kb = 0.114f;
      break;
    case MATRIX_BT2020:
      c.kr = 0.2627f;
      c.kb = 0.0593f;
      break;
    default:
      c.kr = 0.2126f;
      c.kb = 0.0722f;
      break;
  }
  const float m = format == FMT_P010 || format == FMT_YUV444P16 ? 256.0f : 1.0f;
  const float maxraw = format == FMT_P010 ? 65472.0f : format == FMT_YUV444P16 ? 65535.0f : 255.0f;
  if (full) {
    c.yoff = 0.0f;
    c.yscale = maxraw;
    c.coff = m == 256.0f ? 32768.0f : 128.0f;
    c.cscale = maxraw;
  } else {
    c.yoff = 16.0f * m;
    c.yscale = 219.0f * m;
    c.coff = 128.0f * m;
    c.cscale = 224.0f * m;
  }
  return c;
}

// ---- src/kernels.cu ----

inline int mirr(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  return std::min(std::max(i, 0), n - 1);
}

inline float quant(float v, float qdiv, float qmax) {
  return std::fmin(std::fmax(std::rint(v / qdiv), 0.0f), qmax) * qdiv;
}

// Image 4:2:0 en mémoire : luma w × h, chroma entrelacée (w/2 couples) × h/2.
struct frame420 {
  int w = 0, h = 0;
  std::vector<u16> y, uv;
};

inline void combine(float Y, float U, float V, const csp& c, float* r, float* g, float* b) {
  const float kg = 1.0f - c.kr - c.kb;
  *r = Y + 2.0f * (1.0f - c.kr) * V;
  *b = Y + 2.0f * (1.0f - c.kb) * U;
  *g = Y - (2.0f * c.kb * (1.0f - c.kb) * U + 2.0f * c.kr * (1.0f - c.kr) * V) / kg;
}

// k_v_f32 : passe verticale de N plans float.
inline std::vector<float> v_f32(const std::vector<float>& src, int w, int planes, const pass& py) {
  std::vector<float> dst(sz(planes) * sz(w) * sz(py.dst));
  for (int z = 0; z < planes; z++)
    for (int y = 0; y < py.dst; y++)
      for (int x = 0; x < w; x++) {
        const float* sp = src.data() + sz(z) * sz(w) * sz(py.src);
        const float* wt = py.wt.data() + sz(y) * sz(py.taps);
        const int s0 = py.start[sz(y)];
        float a = 0.0f;
        for (int j = 0; j < py.taps; j++) a += wt[j] * sp[sz(mirr(s0 + j, py.src)) * sz(w) + sz(x)];
        dst[sz(z) * sz(w) * sz(py.dst) + sz(y) * sz(w) + sz(x)] = a;
      }
  return dst;
}

// aji_run_pre : k_uv_h, k_v_f32, k_pre_combine. Sortie NCHW avant __float2half.
inline std::vector<float> run_pre(const frame420& in, int siting, bool bilinear_filter, const csp& c) {
  const int w = in.w, h = in.h, cw = w >> 1, ch = h >> 1;
  double sx = 0.0, sy = 0.0;
  chroma_shifts(siting, true, &sx, &sy);
  const pass px = compute(cw, w, sx, bilinear_filter), py = compute(ch, h, sy, bilinear_filter);
  std::vector<float> tmp0(2 * sz(w) * sz(ch));
  for (int y = 0; y < ch; y++)
    for (int x = 0; x < px.dst; x++) {
      const u16* row = in.uv.data() + sz(y) * sz(2 * cw);
      const float* wt = px.wt.data() + sz(x) * sz(px.taps);
      const int s0 = px.start[sz(x)];
      float u = 0.0f, v = 0.0f;
      for (int j = 0; j < px.taps; j++) {
        const int s = mirr(s0 + j, px.src);
        u += wt[j] * static_cast<float>(row[2 * s]);
        v += wt[j] * static_cast<float>(row[2 * s + 1]);
      }
      const std::size_t plane = sz(px.dst) * sz(ch), idx = sz(y) * sz(px.dst) + sz(x);
      tmp0[idx] = u;
      tmp0[plane + idx] = v;
    }
  const std::vector<float> tmp1 = v_f32(tmp0, w, 2, py);
  std::vector<float> dst(3 * sz(w) * sz(h));
  const std::size_t plane = sz(w) * sz(h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const std::size_t idx = sz(y) * sz(w) + sz(x);
      const float Y = (static_cast<float>(in.y[idx]) - c.yoff) / c.yscale;
      const float U = (tmp1[idx] - c.coff) / c.cscale;
      const float V = (tmp1[plane + idx] - c.coff) / c.cscale;
      combine(Y, U, V, c, &dst[idx], &dst[plane + idx], &dst[2 * plane + idx]);
    }
  return dst;
}

// aji_run_post : k_post_matrix, k_h_f32, k_uv_v_store. Entrée NCHW déjà relue de fp16.
inline frame420 run_post(const std::vector<float>& src, int w, int h, bool p010, int siting,
                         bool bilinear_filter, const csp& c) {
  const int cw = w >> 1, ch = h >> 1;
  const float qdiv = p010 ? 64.0f : 1.0f, qmax = p010 ? 1023.0f : 255.0f;
  double sx = 0.0, sy = 0.0;
  chroma_shifts(siting, false, &sx, &sy);
  const pass px = compute(w, cw, sx, bilinear_filter), py = compute(h, ch, sy, bilinear_filter);
  frame420 out{.w = w, .h = h, .y = std::vector<u16>(sz(w) * sz(h)), .uv = std::vector<u16>(sz(w) * sz(ch))};
  std::vector<float> tmp0(2 * sz(w) * sz(h)), tmp1(2 * sz(cw) * sz(h));
  const std::size_t plane = sz(w) * sz(h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const std::size_t idx = sz(y) * sz(w) + sz(x);
      const float r = src[idx], g = src[plane + idx], b = src[2 * plane + idx];
      const float Y = c.kr * r + (1.0f - c.kr - c.kb) * g + c.kb * b;
      out.y[idx] = static_cast<u16>(quant(Y * c.yscale + c.yoff, qdiv, qmax));
      tmp0[idx] = (b - Y) / (2.0f * (1.0f - c.kb));
      tmp0[plane + idx] = (r - Y) / (2.0f * (1.0f - c.kr));
    }
  for (int z = 0; z < 2; z++)  // k_h_f32
    for (int y = 0; y < h; y++)
      for (int x = 0; x < px.dst; x++) {
        const float* sp = tmp0.data() + sz(z) * sz(px.src) * sz(h) + sz(y) * sz(px.src);
        const float* wt = px.wt.data() + sz(x) * sz(px.taps);
        float a = 0.0f;
        for (int j = 0; j < px.taps; j++) a += wt[j] * sp[mirr(px.start[sz(x)] + j, px.src)];
        tmp1[sz(z) * sz(px.dst) * sz(h) + sz(y) * sz(px.dst) + sz(x)] = a;
      }
  for (int y = 0; y < py.dst; y++)  // k_uv_v_store
    for (int x = 0; x < cw; x++) {
      const std::size_t cplane = sz(cw) * sz(py.src);
      const float* wt = py.wt.data() + sz(y) * sz(py.taps);
      float u = 0.0f, v = 0.0f;
      for (int j = 0; j < py.taps; j++) {
        const std::size_t row = sz(mirr(py.start[sz(y)] + j, py.src)) * sz(cw) + sz(x);
        u += wt[j] * tmp1[row];
        v += wt[j] * tmp1[cplane + row];
      }
      u16* uvrow = out.uv.data() + sz(y) * sz(2 * cw);
      uvrow[2 * x] = static_cast<u16>(quant(u * c.cscale + c.coff, qdiv, qmax));
      uvrow[2 * x + 1] = static_cast<u16>(quant(v * c.cscale + c.coff, qdiv, qmax));
    }
  return out;
}

// aji_run_rife_pre420 : la même ingestion sur l'image virtuellement remplie de noir studio.
inline std::vector<float> run_rife_pre420(const frame420& in, int pw, int ph, int left, int top, bool p010,
                                          const csp& c) {
  const int w = pw, h = ph, ch = h >> 1, cw = in.w / 2, source_ch = in.h / 2;
  double sx = 0.0, sy = 0.0;
  chroma_shifts(SITING_LEFT, true, &sx, &sy);
  const pass px = compute(w >> 1, w, sx, true), py = compute(ch, h, sy, true);
  const u16 cpad = p010 ? 32768 : 128, ypad = p010 ? 4096 : 16;
  std::vector<float> tmp0(2 * sz(w) * sz(ch));
  for (int y = 0; y < ch; y++)
    for (int x = 0; x < px.dst; x++) {
      const int syy = y - top / 2;
      const u16* row = syy >= 0 && syy < source_ch ? in.uv.data() + sz(syy) * sz(2 * cw) : nullptr;
      const float* wt = px.wt.data() + sz(x) * sz(px.taps);
      float u = 0.0f, v = 0.0f;
      for (int j = 0; j < px.taps; j++) {
        const int s = mirr(px.start[sz(x)] + j, px.src) - left / 2;
        const bool inside = row != nullptr && s >= 0 && s < cw;
        u += wt[j] * static_cast<float>(inside ? row[2 * s] : cpad);
        v += wt[j] * static_cast<float>(inside ? row[2 * s + 1] : cpad);
      }
      const std::size_t plane = sz(px.dst) * sz(ch), idx = sz(y) * sz(px.dst) + sz(x);
      tmp0[idx] = u;
      tmp0[plane + idx] = v;
    }
  const std::vector<float> tmp1 = v_f32(tmp0, w, 2, py);
  std::vector<float> dst(3 * sz(w) * sz(h));
  const std::size_t plane = sz(w) * sz(h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const int sxx = x - left, syy = y - top;
      const bool inside = sxx >= 0 && syy >= 0 && sxx < in.w && syy < in.h;
      const u16 yy = inside ? in.y[sz(syy) * sz(in.w) + sz(sxx)] : ypad;
      const std::size_t idx = sz(y) * sz(w) + sz(x);
      const float Y = (static_cast<float>(yy) - c.yoff) / c.yscale;
      const float U = (tmp1[idx] - c.coff) / c.cscale;
      const float V = (tmp1[plane + idx] - c.coff) / c.cscale;
      combine(Y, U, V, c, &dst[idx], &dst[plane + idx], &dst[2 * plane + idx]);
    }
  return dst;
}

// k_pre444_pad (left = top = 0 et pw = w : k_pre444). Plans Y, Cb, Cr de w × h.
inline std::vector<float> run_pre444_pad(const std::vector<u16>& yp, const std::vector<u16>& cbp,
                                         const std::vector<u16>& crp, int w, int h, int pw, int ph, int left,
                                         int top, const csp& c) {
  std::vector<float> dst(3 * sz(pw) * sz(ph));
  const std::size_t plane = sz(pw) * sz(ph);
  for (int y = 0; y < ph; y++)
    for (int x = 0; x < pw; x++) {
      const int sx = x - left, sy = y - top;
      const bool inside = sx >= 0 && sy >= 0 && sx < w && sy < h;
      const std::size_t at = inside ? sz(sy) * sz(w) + sz(sx) : 0;
      const unsigned yy = inside ? yp[at] : 16u * 256u;
      const unsigned uu = inside ? cbp[at] : 128u * 256u;
      const unsigned vv = inside ? crp[at] : 128u * 256u;
      const float Y = (static_cast<float>(yy) - c.yoff) / c.yscale;
      const float U = (static_cast<float>(uu) - c.coff) / c.cscale;
      const float V = (static_cast<float>(vv) - c.coff) / c.cscale;
      const std::size_t idx = sz(y) * sz(pw) + sz(x);
      combine(Y, U, V, c, &dst[idx], &dst[plane + idx], &dst[2 * plane + idx]);
    }
  return dst;
}

// k_post444 (YUV444P16 : qdiv 1, qmax 65535). Sortie Y, Cb, Cr à la suite.
inline std::vector<u16> run_post444(const std::vector<float>& src, int w, int h, const csp& c) {
  const std::size_t plane = sz(w) * sz(h);
  std::vector<u16> out(3 * plane);
  for (std::size_t idx = 0; idx < plane; idx++) {
    const float r = src[idx], g = src[plane + idx], b = src[2 * plane + idx];
    const float Y = c.kr * r + (1.0f - c.kr - c.kb) * g + c.kb * b;
    const float U = (b - Y) / (2.0f * (1.0f - c.kb));
    const float V = (r - Y) / (2.0f * (1.0f - c.kr));
    out[idx] = static_cast<u16>(quant(Y * c.yscale + c.yoff, 1.0f, 65535.0f));
    out[plane + idx] = static_cast<u16>(quant(U * c.cscale + c.coff, 1.0f, 65535.0f));
    out[2 * plane + idx] = static_cast<u16>(quant(V * c.cscale + c.coff, 1.0f, 65535.0f));
  }
  return out;
}

// k_block_diff : bpp 1 pour NV12, 2 pour les conteneurs 16 bits.
inline unsigned block_diff(const std::vector<u16>& ya, const std::vector<u16>& yb, int w, int h, int bpp) {
  unsigned dmax = 0;
  for (int y0 = 0; y0 < h; y0 += 32)
    for (int x0 = 0; x0 < w; x0 += 32) {
      const int x1 = std::min(x0 + 32, w), y1 = std::min(y0 + 32, h);
      unsigned int sum = 0;
      for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
          const std::size_t at = sz(y) * sz(w) + sz(x);
          sum += static_cast<unsigned>(std::abs(static_cast<int>(ya[at]) - static_cast<int>(yb[at])));
        }
      const auto n = static_cast<unsigned>((x1 - x0) * (y1 - y0));
      dmax = std::max(dmax, bpp == 1 ? (sum * 256u) / n : sum / n);  // atomicMax
    }
  return dmax;
}

// k_scd_diff : chaque terme |a − b| × norm en float ; la somme, ici en double.
inline double scd_sum(const std::vector<u16>& ya, const std::vector<u16>& yb, float norm) {
  double sum = 0.0;
  for (std::size_t i = 0; i < ya.size(); ++i) {
    const float a = ya[i], b = yb[i];
    sum += static_cast<double>(std::fabs(a - b) * norm);
  }
  return sum;
}

// sample_window ; sample_plane est la fenêtre (0, 0, w, h) d'un plan de pas w.
inline float sample_window(const float* p, int stride, int x0w, int y0w, int w, int h, float x, float y) {
  x = std::fmin(std::fmax(x, 0.0f), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0f), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
  const float* r0 = p + sz(y0w + y0) * sz(stride) + sz(x0w);
  const float* r1 = p + sz(y0w + y1) * sz(stride) + sz(x0w);
  const float top_row = r0[x0] * (1.0f - fx) + r0[x1] * fx;
  const float bottom = r1[x0] * (1.0f - fx) + r1[x1] * fx;
  return top_row * (1.0f - fy) + bottom * fy;
}

// k_warp_blend_mid ; à t = 0,5, sa branche est exactement k_warp_blend.
// a, b : NCHW w × h ; flow : 4 plans fpw × fph ; mask : 1 plan ; fenêtre (fx0, fy0, fw, fh).
inline std::vector<float> warp_blend_mid(const std::vector<float>& a, const std::vector<float>& b,
                                         const std::vector<float>& flow, const std::vector<float>& mask,
                                         int w, int h, int fpw, int fph, int fx0, int fy0, int fw, int fh,
                                         float t) {
  const float law_a = 0.285f, law_b = 2.591f, law_c = 0.170f;
  std::vector<float> dst(3 * sz(w) * sz(h));
  const std::size_t fplane = sz(fpw) * sz(fph);
  const float *f0x = flow.data(), *f0y = f0x + fplane, *f1x = f0y + fplane, *f1y = f1x + fplane;
  const auto win = [&](const float* p, float u, float v) {
    return sample_window(p, fpw, fx0, fy0, fw, fh, u, v);
  };
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const float sx = static_cast<float>(w) / static_cast<float>(fw),
                  sy = static_cast<float>(h) / static_cast<float>(fh);
      const float u = (static_cast<float>(x) + 0.5f) / sx - 0.5f,
                  v = (static_cast<float>(y) + 0.5f) / sy - 0.5f;
      float d0x = 0, d0y = 0, d1x = 0, d1y = 0, m = 0;
      if (t == 0.5f) {
        d0x = win(f0x, u, v);
        d0y = win(f0y, u, v);
        d1x = win(f1x, u, v);
        d1y = win(f1y, u, v);
        m = win(mask.data(), u, v);
      } else {
        const float a0x = win(f0x, u, v), a0y = win(f0y, u, v), a1x = win(f1x, u, v), a1y = win(f1y, u, v);
        const float u2 = u - ((1.0f - t) * a0x + t * a1x), v2 = v - ((1.0f - t) * a0y + t * a1y);
        const float tx = win(f1x, u2, v2) - win(f0x, u2, v2);
        const float ty = win(f1y, u2, v2) - win(f0y, u2, v2);
        d0x = -t * tx;
        d0y = -t * ty;
        d1x = (1.0f - t) * tx;
        d1y = (1.0f - t) * ty;
        const float mm = std::fmin(std::fmax(win(mask.data(), u2, v2), 1e-4f), 1.0f - 1e-4f);
        const float z = law_a * std::log(mm / (1.0f - mm)) + law_b * std::log((1.0f - t) / t) + law_c;
        m = 1.0f / (1.0f + std::exp(-z));
      }
      const float ax = static_cast<float>(x) + sx * d0x, ay = static_cast<float>(y) + sy * d0y;
      const float bx = static_cast<float>(x) + sx * d1x, by = static_cast<float>(y) + sy * d1y;
      const std::size_t plane = sz(w) * sz(h), at = sz(y) * sz(w) + sz(x);
      for (int ch = 0; ch < 3; ch++) {
        const float va = sample_window(a.data() + sz(ch) * plane, w, 0, 0, w, h, ax, ay);
        const float vb = sample_window(b.data() + sz(ch) * plane, w, 0, 0, w, h, bx, by);
        dst[sz(ch) * plane + at] = va * m + vb * (1.0f - m);
      }
    }
  return dst;
}

// k_rife_consts : grille et multiplicateurs, avant __float2half. Plans meshx, meshy, mulw, mulh.
inline std::vector<float> rife_consts(int w, int h) {
  const std::size_t plane = sz(w) * sz(h);
  std::vector<float> out(4 * plane);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const std::size_t idx = sz(y) * sz(w) + sz(x);
      out[idx] = 2.0f * static_cast<float>(x) / static_cast<float>(w - 1) - 1.0f;
      out[plane + idx] = 2.0f * static_cast<float>(y) / static_cast<float>(h - 1) - 1.0f;
      out[2 * plane + idx] = 2.0f / static_cast<float>(w - 1);
      out[3 * plane + idx] = 2.0f / static_cast<float>(h - 1);
    }
  return out;
}

}  // namespace oracle
