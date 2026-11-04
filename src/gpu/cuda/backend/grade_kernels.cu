#include "gpu/cuda/backend/grade_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "kernels/spec/grade_pixel.h"

namespace ajp::gpu::cuda {

namespace g = kernels::grade;

namespace {

// Les poids de la gaussienne (kernels::grade::kBlurWeights, un std::array que le code de l'appareil ne lit
// pas), en mémoire constante ; leur égalité est vérifiée à la compilation de l'hôte.
__constant__ float kBlur[8] = {0.1346F, 0.1273F, 0.1078F, 0.0816F, 0.0553F, 0.0336F, 0.0182F, 0.0088F};
#if !defined(__CUDA_ARCH__)
static_assert(g::kBlurWeights[0] == 0.1346F && g::kBlurWeights[1] == 0.1273F &&
                  g::kBlurWeights[2] == 0.1078F && g::kBlurWeights[3] == 0.0816F &&
                  g::kBlurWeights[4] == 0.0553F && g::kBlurWeights[5] == 0.0336F &&
                  g::kBlurWeights[6] == 0.0182F && g::kBlurWeights[7] == 0.0088F,
              "kBlur suit kernels::grade::kBlurWeights");
#endif
static_assert(g::kBlurRadius == 7, "la tuile partagée suppose un rayon de 7");

constexpr int kTile = 16;                 // 16 × 16 fils, comme le shader de zone
constexpr int kHalo = g::kBlurRadius;     // 7
constexpr int kSpan = kTile + 2 * kHalo;  // 30

__host__ __device__ inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Le code PQ d'un pixel de l'emplacement : relu, puis l'entrée de libplacebo (indices bornés à l'image).
struct SurfaceFetch {
  RingSurface surface;
  int width;
  int height;
  const float* nits;
  const float* matrix;
  __host__ __device__ g::V3 operator()(int x, int y) const {
#if defined(__CUDA_ARCH__)
    const float3 c = load_ring(surface, clampi(x, 0, width - 1), clampi(y, 0, height - 1));
    return g::input_code({c.x, c.y, c.z}, nits, matrix);
#else
    (void)x;
    (void)y;
    return g::v3(0.0F);
#endif
  }
};

// Un texel d'une carte de `comps` composantes, aux indices bornés.
struct MapFetch {
  const float* data;
  int w;
  int h;
  int comps;
  __host__ __device__ g::V3 operator()(int x, int y) const {
    const int at = (clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1)) * comps;
    return {data[at], comps > 1 ? data[at + 1] : 0.0F, comps > 2 ? data[at + 2] : 0.0F};
  }
};

// La pile de LUT : la lecture linéaire de la texture, au centre des texels des bords comme le hook.
struct LookTex {
  cudaTextureObject_t object;
  int size;
  __host__ __device__ g::V3 operator()(g::V3 c) const {
#if defined(__CUDA_ARCH__)
    if (object == 0 || size < 2) return c;
    const float n = static_cast<float>(size);
    const float4 v = tex3D<float4>(object, (c.r * (n - 1.0F) + 0.5F) / n, (c.g * (n - 1.0F) + 0.5F) / n,
                                   (c.b * (n - 1.0F) + 0.5F) / n);
    return {v.x, v.y, v.z};
#else
    return c;
#endif
  }
};

// Les paramètres d'un lancement, passés par valeur (la matrice d'entrée dans le bloc des paramètres).
struct Launch {
  GradeFrame frame;
  GradeMapSizes sizes;
};

__device__ inline SurfaceFetch device_fetch(const Launch& p) {
  return {p.frame.surface, p.frame.width, p.frame.height, p.frame.input_nits, p.frame.input_matrix};
}

__global__ void __launch_bounds__(kPixelThreads) grade_local_kernel(Launch p) {
  const int lx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int ly = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (lx >= p.sizes.local_w || ly >= p.sizes.local_h) return;
  const SurfaceFetch fetch = device_fetch(p);
  const g::V3 v = g::local_texel(fetch, p.frame.width, p.frame.height, lx, ly, p.sizes.local_w,
                                 p.sizes.local_h, p.frame.uniforms);
  float* out = p.frame.local + (ly * p.sizes.local_w + lx) * 3;
  out[0] = v.r;
  out[1] = v.g;
  out[2] = v.b;
}

// La carte locale quand le bloc tombe sur les pixels (largeur et hauteur multiples de 8) : chaque lecture
// bilinéaire de local_texel est alors au coin de 4 pixels, soit leur moyenne. Un fil par pixel (lectures
// voisines entre fils, au lieu de 64 lectures en série par texel), la tuile de 16 × 16 en mémoire partagée :
// 2 × 2 texels par bloc, sommés dans le même ordre que local_texel.
__global__ void __launch_bounds__(kTile* kTile) grade_local_tiled_kernel(Launch p) {
  __shared__ float code[kTile][kTile][3];
  __shared__ float quad[kTile / 2][kTile / 2][3];
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int x = static_cast<int>(blockIdx.x) * kTile + tx;
  const int y = static_cast<int>(blockIdx.y) * kTile + ty;
  const g::V3 e = x < p.frame.width && y < p.frame.height ? device_fetch(p)(x, y) : g::v3(0.0F);
  code[ty][tx][0] = e.r;
  code[ty][tx][1] = e.g;
  code[ty][tx][2] = e.b;
  __syncthreads();
  if (tx < kTile / 2 && ty < kTile / 2) {
    const auto at = [&](int i, int j) { return g::V3{code[j][i][0], code[j][i][1], code[j][i][2]}; };
    const int qx = 2 * tx;
    const int qy = 2 * ty;
    const g::V3 top = g::vmix(at(qx, qy), at(qx + 1, qy), 0.5F);
    const g::V3 bottom = g::vmix(at(qx, qy + 1), at(qx + 1, qy + 1), 0.5F);
    const g::V3 v = g::local_sample(g::vmix(top, bottom, 0.5F), p.frame.uniforms);
    quad[ty][tx][0] = v.r;
    quad[ty][tx][1] = v.g;
    quad[ty][tx][2] = v.b;
  }
  __syncthreads();
  if (tx >= 2 || ty >= 2) return;
  const int lx = static_cast<int>(blockIdx.x) * 2 + tx;
  const int ly = static_cast<int>(blockIdx.y) * 2 + ty;
  if (lx >= p.sizes.local_w || ly >= p.sizes.local_h) return;
  g::V3 acc = g::v3(0.0F);
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
      const float* q = quad[ty * 4 + j][tx * 4 + i];
      acc = acc + g::V3{q[0], q[1], q[2]};
    }
  acc = acc / 16.0F;
  float* out = p.frame.local + (ly * p.sizes.local_w + lx) * 3;
  out[0] = acc.r;
  out[1] = acc.g;
  out[2] = acc.b;
}

__global__ void __launch_bounds__(kPixelThreads) grade_skin_kernel(Launch p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.frame.width || y >= p.frame.height) return;
  p.frame.skin[y * p.frame.width + x] = g::skin_area(device_fetch(p), x, y, p.frame.uniforms);
}

// 2-area.hook : la moyenne 4 × 4 de la carte locale sur une tuile de 30 × 30 (halo de 7, indices bornés), le
// flou horizontal puis vertical en mémoire partagée, puis le lissage.
__global__ void __launch_bounds__(kTile* kTile) grade_area_kernel(Launch p, float dt_s, bool reset) {
  __shared__ float a0[kSpan][kSpan];
  __shared__ float ah[kSpan][kTile];
  const int aw = p.sizes.area_w;
  const int ah_ = p.sizes.area_h;
  const MapFetch local{p.frame.local, p.sizes.local_w, p.sizes.local_h, 3};
  const int ox = static_cast<int>(blockIdx.x) * kTile - kHalo;
  const int oy = static_cast<int>(blockIdx.y) * kTile - kHalo;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int i = tid; i < kSpan * kSpan; i += kTile * kTile) {
    const int qx = clampi(ox + i % kSpan, 0, aw - 1);
    const int qy = clampi(oy + i / kSpan, 0, ah_ - 1);
    a0[i / kSpan][i % kSpan] = g::area_mean(local, p.sizes.local_w, p.sizes.local_h, qx, qy, aw, ah_);
  }
  __syncthreads();
  for (int i = tid; i < kSpan * kTile; i += kTile * kTile) {
    const int y = i / kTile;
    const int x = i % kTile + kHalo;
    float acc = a0[y][x] * kBlur[0];
    for (int k = 1; k <= kHalo; ++k) acc += (a0[y][x - k] + a0[y][x + k]) * kBlur[k];
    ah[y][x - kHalo] = acc;
  }
  __syncthreads();
  const int px = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
  const int py = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
  if (px >= aw || py >= ah_) return;
  const int ly = static_cast<int>(threadIdx.y) + kHalo;
  const int lx = static_cast<int>(threadIdx.x);
  float value = ah[ly][lx] * kBlur[0];
  for (int k = 1; k <= kHalo; ++k) value += (ah[ly - k][lx] + ah[ly + k][lx]) * kBlur[k];
  const int idx = py * aw + px;
  if (idx < g::kHistoryTexels) {
    const float kept = reset ? 0.0F : p.frame.history[idx];
    value = g::area_smooth(value, kept, dt_s, p.frame.smooth_jump);
    p.frame.history[idx] = value + 1.0F;
  }
  p.frame.area[idx] = value;
}

__global__ void __launch_bounds__(kPixelThreads) grade_localmax_kernel(Launch p) {
  const int lx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int ly = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (lx >= p.sizes.local_w || ly >= p.sizes.local_h) return;
  const kernels::GradeUniforms& u = p.frame.uniforms;
  const SurfaceFetch fetch = device_fetch(p);
  const int w = p.frame.width;
  const int h = p.frame.height;
  int px = 0;
  int py = 0;
  g::localmax_pick(fetch, w, h, lx, ly, p.sizes.local_w, p.sizes.local_h, &px, &py);
  const float s = (static_cast<float>(px) + 0.5F) / static_cast<float>(w);
  const float t = (static_cast<float>(py) + 0.5F) / static_cast<float>(h);
  const MapFetch local{p.frame.local, p.sizes.local_w, p.sizes.local_h, 3};
  const MapFetch area{p.frame.area, p.sizes.area_w, p.sizes.area_h, 1};
  const float skin = u.skin_on > 0.5F ? p.frame.skin[py * w + px] : 0.0F;
  g::V3 e = g::grade_code(fetch(px, py), skin, g::bilinear(local, p.sizes.local_w, p.sizes.local_h, s, t).r,
                          g::bilinear(area, p.sizes.area_w, p.sizes.area_h, s, t).r, u);
  e = g::look_code(e, LookTex{p.frame.look, p.frame.look_size}, u);
  p.frame.localmax[ly * p.sizes.local_w + lx] = g::max3(e);
}

__global__ void __launch_bounds__(kPixelThreads) grade_pixel_kernel(Launch p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int w = p.frame.width;
  const int h = p.frame.height;
  if (x >= w || y >= h) return;
  const kernels::GradeUniforms& u = p.frame.uniforms;
  const float s = (static_cast<float>(x) + 0.5F) / static_cast<float>(w);
  const float t = (static_cast<float>(y) + 0.5F) / static_cast<float>(h);
  const MapFetch local{p.frame.local, p.sizes.local_w, p.sizes.local_h, 3};
  const MapFetch area{p.frame.area, p.sizes.area_w, p.sizes.area_h, 1};
  const float black = u.black_hi > u.black_lo
                          ? g::bilinear(MapFetch{p.frame.localmax, p.sizes.local_w, p.sizes.local_h, 1},
                                        p.sizes.local_w, p.sizes.local_h, s, t)
                                .r
                          : 0.0F;
  const float skin = u.skin_on > 0.5F ? p.frame.skin[y * w + x] : 0.0F;
  // Seul ce pixel est relu ici : les voisins (masque peau) l'ont été avant, l'écriture en place est sûre.
  const g::V3 e = g::grade_pixel(device_fetch(p)(x, y), skin,
                                 g::bilinear(local, p.sizes.local_w, p.sizes.local_h, s, t).r,
                                 g::bilinear(area, p.sizes.area_w, p.sizes.area_h, s, t).r, black, u,
                                 LookTex{p.frame.look, p.frame.look_size});
  store_ring(p.frame.surface, x, y, e.r, e.g, e.b);
}

__global__ void __launch_bounds__(kPixelThreads) grade_lut_kernel(GradeFrame f) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= f.width || y >= f.height) return;
  const SurfaceFetch fetch{f.surface, f.width, f.height, f.input_nits, f.input_matrix};
  const g::V3 e =
      g::curve_lut_code(fetch(x, y), LookTex{f.curve, f.curve_size}, LookTex{f.curve_fade, f.curve_size},
                        f.curve_fade != 0 ? f.curve_mix : 0.0F);
  store_ring(f.surface, x, y, e.r, e.g, e.b);
}

dim3 grid_of(int w, int h) {
  return dim3(static_cast<unsigned>((w + kTile - 1) / kTile), static_cast<unsigned>((h + kTile - 1) / kTile));
}

bool valid(const GradeFrame& f) {
  return f.surface.object != 0 && f.width >= 2 && f.height >= 2 && f.input_nits != nullptr &&
         f.local != nullptr && f.area != nullptr && f.history != nullptr && f.localmax != nullptr &&
         (f.uniforms.skin_on < 0.5F || f.skin != nullptr) &&
         (f.uniforms.look_mix <= 0.0F || (f.look != 0 && f.look_size >= 2));
}

}  // namespace

GradeMapSizes grade_map_sizes(int width, int height) {
  GradeMapSizes s;
  s.local_w = g::map_size(width, g::kLocalDiv);
  s.local_h = g::map_size(height, g::kLocalDiv);
  s.area_w = g::map_size(s.local_w, g::kAreaDiv);
  s.area_h = g::map_size(s.local_h, g::kAreaDiv);
  return s;
}

cudaError_t launch_grade_local(const GradeFrame& frame, cudaStream_t stream) {
  if (!valid(frame)) return cudaErrorInvalidValue;
  const Launch p{frame, grade_map_sizes(frame.width, frame.height)};
  const dim3 block(kTile, kTile);
  const bool tiled = frame.width % g::kLocalDiv == 0 && frame.height % g::kLocalDiv == 0 &&
                     p.sizes.local_w * g::kLocalDiv == frame.width &&
                     p.sizes.local_h * g::kLocalDiv == frame.height;
  if (tiled)
    grade_local_tiled_kernel<<<grid_of(frame.width, frame.height), block, 0, stream>>>(p);
  else
    grade_local_kernel<<<grid_of(p.sizes.local_w, p.sizes.local_h), block, 0, stream>>>(p);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  if (frame.uniforms.skin_on > 0.5F)
    grade_skin_kernel<<<grid_of(frame.width, frame.height), block, 0, stream>>>(p);
  return cudaGetLastError();
}

cudaError_t launch_grade_area(const GradeFrame& frame, float dt_s, bool reset, cudaStream_t stream) {
  if (!valid(frame)) return cudaErrorInvalidValue;
  const Launch p{frame, grade_map_sizes(frame.width, frame.height)};
  grade_area_kernel<<<grid_of(p.sizes.area_w, p.sizes.area_h), dim3(kTile, kTile), 0, stream>>>(p, dt_s,
                                                                                                reset);
  return cudaGetLastError();
}

cudaError_t launch_grade_slot(const GradeFrame& frame, cudaStream_t stream) {
  if (!valid(frame)) return cudaErrorInvalidValue;
  const Launch p{frame, grade_map_sizes(frame.width, frame.height)};
  const dim3 block(kTile, kTile);
  if (frame.uniforms.black_hi > frame.uniforms.black_lo) {
    grade_localmax_kernel<<<grid_of(p.sizes.local_w, p.sizes.local_h), block, 0, stream>>>(p);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  grade_pixel_kernel<<<grid_of(frame.width, frame.height), block, 0, stream>>>(p);
  return cudaGetLastError();
}

cudaError_t launch_grade_lut_slot(const GradeFrame& frame, cudaStream_t stream) {
  if (frame.surface.object == 0 || frame.width < 1 || frame.height < 1 || frame.input_nits == nullptr ||
      frame.curve == 0 || frame.curve_size < 2)
    return cudaErrorInvalidValue;
  grade_lut_kernel<<<grid_of(frame.width, frame.height), dim3(kTile, kTile), 0, stream>>>(frame);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
