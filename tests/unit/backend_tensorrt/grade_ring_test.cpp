// Le grade exact dans l'anneau (gpu/cuda/backend/grade_kernels.h, chantier 2.8) contre la référence
// kernels/ref/grade.cpp : même entrée SDR, mêmes réglages, |ΔPQ| ≤ 2⁻¹¹ (max) et ≤ 2⁻¹² (p99,9) ; et le banc
// caché [.grade_ring] (1440p et 2160p).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/cuda/backend/grade_kernels.h"
#include "kernels/ref/grade.h"
#include "kernels/spec/grade_pixel.h"

using namespace ajp;
using ajp::test::has_cuda;
namespace kref = ajp::kernels::ref;

namespace {

// Pans sombre et clair, rampe orange, rouge et bleu saturés, une peau claire, un éclat, en codes SDR.
kernels::RgbImage scene(int w, int h) {
  kernels::RgbImage image(w, h);
  std::uint32_t noise = 777U;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float r = x < w / 2 ? 0.25F : 0.85F;
      float g = r;
      float b = r;
      if (y >= h / 2) {
        r = static_cast<float>(x + 1) / static_cast<float>(w);
        g = 0.75F * r;
        b = 0.2F * r;
      }
      if (y >= 3 * h / 4 && x < w / 4) {
        r = 0.7F;
        g = 0.12F;
        b = 0.08F;
      }
      if (y >= 3 * h / 4 && x >= 3 * w / 4) {
        r = 0.15F;
        g = 0.3F;
        b = 0.75F;
      }
      if (y < h / 4 && x >= w / 4 && x < w / 2) {
        r = 1.0F;
        g = 224.0F / 255.0F;
        b = 200.0F / 255.0F;
      }
      if (x >= 40 && x < 43 && y >= 30 && y < 33) r = g = b = 1.0F;
      noise = noise * 1664525U + 1013904223U;
      const float grain = 0.01F * (static_cast<float>(noise >> 24U) / 255.0F - 0.5F);
      image.planes[0].at(x, y) = std::clamp(r + grain, 0.0F, 1.0F);
      image.planes[1].at(x, y) = std::clamp(g + grain, 0.0F, 1.0F);
      image.planes[2].at(x, y) = std::clamp(b + grain, 0.0F, 1.0F);
    }
  return image;
}

// Une LUT chaude de 17 points (R + 8 %, B − 10 %), R le plus rapide.
std::vector<float> warm_lut(int n) {
  std::vector<float> lut;
  for (int b = 0; b < n; ++b)
    for (int g = 0; g < n; ++g)
      for (int r = 0; r < n; ++r) {
        const auto v = [n](int i) { return static_cast<float>(i) / static_cast<float>(n - 1); };
        lut.push_back(std::min(1.0F, v(r) * 1.08F));
        lut.push_back(v(g));
        lut.push_back(v(b) * 0.9F);
      }
  return lut;
}

struct Case {
  const char* name;
  kernels::GradeUniforms uniforms;
  bool look = false;
};

std::vector<Case> cases() {
  std::vector<Case> out;
  out.push_back({"defaut", {}});
  kernels::GradeUniforms strong;
  strong.paper = 500.0F;
  strong.in_max = 1000.0F;
  strong.shadows = 0.0F;
  strong.area_gain = 3.0F;
  strong.spec_gain = 3.0F;
  strong.guard = 0.5F;
  strong.gamut_p3 = 0.0F;
  out.push_back({"eclats", strong});
  kernels::GradeUniforms vivid = strong;
  vivid.vibrance = 0.6F;
  vivid.punch = 0.4F;
  vivid.chroma = 1.2F;
  vivid.gamut_p3 = 1.0F;
  vivid.expand = 0.5F;
  out.push_back({"vibrance-punch-etendu", vivid});
  kernels::GradeUniforms truehdr;
  truehdr.emulate = 1.0F;
  truehdr.adapt = 1.0F;
  truehdr.ad_on = 1.0F;
  truehdr.sh_on = 1.0F;
  truehdr.sh_t = 0.6F;
  truehdr.sh_glow = 0.3F;
  truehdr.sh_series = 1.0F;
  truehdr.fs_apl = 60.0F;
  truehdr.fs_bright = 0.1F;
  truehdr.abl = 0.5F;
  truehdr.skin_on = 1.0F;
  out.push_back({"truehdr-peau", truehdr});
  kernels::GradeUniforms finish = strong;
  finish.look_mix = 1.0F;
  finish.black_lo = 0.002F;
  finish.black_hi = 0.02F;
  finish.kwin_on = 1.0F;
  finish.kwin_ref = 200.0F;
  finish.kwin_peak = 800.0F;
  out.push_back({"look-noir-kwin", finish, true});
  return out;
}

// Les tampons de l'appareil d'une image de l'anneau, RGBA16.
struct DeviceFrame {
  cudaArray_t array = nullptr;
  cudaSurfaceObject_t surface = 0;
  cudaArray_t lut_array = nullptr;
  cudaTextureObject_t lut = 0;
  float* table = nullptr;
  float* local = nullptr;
  float* area = nullptr;
  float* history = nullptr;
  float* localmax = nullptr;
  float* skin = nullptr;
  int w = 0;
  int h = 0;
  gpu::cuda::GradeMapSizes sizes;

  DeviceFrame(int width, int height, const kref::GradeInput& input, const std::vector<float>& lut_rgb, int n)
      : w(width), h(height), sizes(gpu::cuda::grade_map_sizes(width, height)) {
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<ushort4>();
    REQUIRE(cudaMallocArray(&array, &format, static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                            cudaArraySurfaceLoadStore) == cudaSuccess);
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    REQUIRE(cudaCreateSurfaceObject(&surface, &resource) == cudaSuccess);
    std::vector<float> nits;
    for (const auto& t : input.nits) nits.insert(nits.end(), t.begin(), t.end());
    REQUIRE(cudaMalloc(&table, nits.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(table, nits.data(), nits.size() * sizeof(float), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    const auto local_n = static_cast<std::size_t>(sizes.local_w) * static_cast<std::size_t>(sizes.local_h);
    const auto area_n = static_cast<std::size_t>(sizes.area_w) * static_cast<std::size_t>(sizes.area_h);
    REQUIRE(cudaMalloc(&local, local_n * 3 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&localmax, local_n * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&area, area_n * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&history, area_n * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(history, 0, area_n * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&skin, static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * sizeof(float)) ==
            cudaSuccess);
    // La pile en texture 3D float4, filtre linéaire, coordonnées normalisées.
    std::vector<float> rgba;
    for (std::size_t i = 0; i < lut_rgb.size(); i += 3)
      rgba.insert(rgba.end(), {lut_rgb[i], lut_rgb[i + 1], lut_rgb[i + 2], 1.0F});
    const cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
    const auto un = static_cast<std::size_t>(n);
    REQUIRE(cudaMalloc3DArray(&lut_array, &f4, make_cudaExtent(un, un, un)) == cudaSuccess);
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(rgba.data(), un * sizeof(float4), un, un);
    copy.dstArray = lut_array;
    copy.extent = make_cudaExtent(un, un, un);
    copy.kind = cudaMemcpyHostToDevice;
    REQUIRE(cudaMemcpy3D(&copy) == cudaSuccess);
    cudaResourceDesc lut_resource{};
    lut_resource.resType = cudaResourceTypeArray;
    lut_resource.res.array.array = lut_array;
    cudaTextureDesc tex{};
    tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 1;
    REQUIRE(cudaCreateTextureObject(&lut, &lut_resource, &tex, nullptr) == cudaSuccess);
  }
  DeviceFrame(const DeviceFrame&) = delete;
  DeviceFrame& operator=(const DeviceFrame&) = delete;
  ~DeviceFrame() {
    cudaDestroyTextureObject(lut);
    cudaFreeArray(lut_array);
    cudaDestroySurfaceObject(surface);
    cudaFreeArray(array);
    for (float* p : {table, local, area, history, localmax, skin}) cudaFree(p);
  }

  void upload(const std::vector<std::uint16_t>& rgba16) {
    REQUIRE(cudaMemcpy2DToArray(array, 0, 0, rgba16.data(), static_cast<std::size_t>(w) * 8,
                                static_cast<std::size_t>(w) * 8, static_cast<std::size_t>(h),
                                cudaMemcpyHostToDevice) == cudaSuccess);
  }
  std::vector<std::uint16_t> download() const {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    REQUIRE(cudaMemcpy2DFromArray(out.data(), static_cast<std::size_t>(w) * 8, array, 0, 0,
                                  static_cast<std::size_t>(w) * 8, static_cast<std::size_t>(h),
                                  cudaMemcpyDeviceToHost) == cudaSuccess);
    return out;
  }
  std::vector<float> history_host() const {
    std::vector<float> out(static_cast<std::size_t>(sizes.area_w) * static_cast<std::size_t>(sizes.area_h));
    REQUIRE(cudaMemcpy(out.data(), history, out.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return out;
  }

  gpu::cuda::GradeFrame frame(const kernels::GradeUniforms& u, const kref::GradeInput& input, bool look,
                              int lut_size) const {
    gpu::cuda::GradeFrame f;
    f.surface = {.object = surface, .packing = gpu::cuda::RingPacking::Rgba16, .dither = false};
    f.width = w;
    f.height = h;
    f.input_nits = table;
    std::copy(input.matrix.begin(), input.matrix.end(), f.input_matrix);
    f.look = look ? lut : 0;
    f.look_size = look ? lut_size : 0;
    f.uniforms = u;
    f.local = local;
    f.area = area;
    f.history = history;
    f.localmax = localmax;
    f.skin = skin;
    return f;
  }
};

std::vector<std::uint16_t> to_rgba16(const kernels::RgbImage& image) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(image.width()) *
                                 static_cast<std::size_t>(image.height()) * 4);
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x) {
      const std::size_t at = (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width()) +
                              static_cast<std::size_t>(x)) *
                             4;
      for (std::size_t c = 0; c < 3; ++c)
        out[at + c] = static_cast<std::uint16_t>(std::lround(image.planes[c].at(x, y) * 65535.0F));
      out[at + 3] = 65535;
    }
  return out;
}

// Les codes que l'emplacement porte : ceux que les noyaux relisent.
kernels::RgbImage from_rgba16(const std::vector<std::uint16_t>& codes, int w, int h) {
  kernels::RgbImage out(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (std::size_t c = 0; c < 3; ++c)
        out.planes[c].at(x, y) =
            static_cast<float>(codes[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                                      static_cast<std::size_t>(x)) *
                                         4 +
                                     c]) /
            65535.0F;
  return out;
}

}  // namespace

TEST_CASE("grade de l'anneau contre la référence", "[backend_tensorrt][cuda][grade_ring]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int n = 17;
  // 250 × 140 : ni 8 ni 32 ne la divisent, les cartes arrondies comme libplacebo ; 256 × 144 : multiple de 8,
  // la carte locale par tuiles (grade_local_tiled_kernel).
  for (const auto [w, h] : {std::array{250, 140}, std::array{256, 144}}) {
    INFO(w << "x" << h);
    const kref::GradeInput input = kref::sdr_input();
    const std::vector<float> lut = warm_lut(n);
    const std::vector<std::uint16_t> source = to_rgba16(scene(w, h));
    const kernels::RgbImage codes = from_rgba16(source, w, h);
    const auto pq = kref::apply_input(codes, input);
    REQUIRE(pq);
    DeviceFrame dev(w, h, input, lut, n);
    for (const Case& c : cases()) {
      INFO(c.name);
      const kref::GradeLook look = c.look ? kref::GradeLook{.size = n, .rgb = lut} : kref::GradeLook{};
      std::vector<float> previous;
      // Deux images de suite : la première sans historique (époque neuve), la seconde lissée par la première.
      for (int frame = 0; frame < 2; ++frame) {
        INFO("image " << frame);
        dev.upload(source);
        const gpu::cuda::GradeFrame f = dev.frame(c.uniforms, input, c.look, n);
        const float dt = 1.0F / 24.0F;
        REQUIRE(gpu::cuda::launch_grade_local(f, nullptr) == cudaSuccess);
        REQUIRE(gpu::cuda::launch_grade_area(f, dt, frame == 0, nullptr) == cudaSuccess);
        REQUIRE(gpu::cuda::launch_grade_slot(f, nullptr) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const std::vector<std::uint16_t> got = dev.download();

        const auto maps = kref::grade_maps(*pq, c.uniforms, look, previous, dt);
        REQUIRE(maps);
        const auto want = kref::grade_image(*pq, *maps, c.uniforms, look);
        REQUIRE(want);
        previous = maps->history;
        const std::vector<float> history = dev.history_host();
        float history_worst = 0.0F;
        for (std::size_t i = 0; i < history.size(); ++i)
          history_worst = std::max(history_worst, std::abs(history[i] - maps->history[i]));
        CHECK(history_worst <= 1e-4F);

        std::vector<float> deltas;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            for (std::size_t ch = 0; ch < 3; ++ch) {
              const float v =
                  static_cast<float>(
                      got[(static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)) * 4 + ch]) /
                  65535.0F;
              deltas.push_back(std::abs(v - std::clamp(want->planes[ch].at(x, y), 0.0F, 1.0F)));
            }
        std::ranges::sort(deltas);
        const float p999 = deltas[static_cast<std::size_t>(0.999 * static_cast<double>(deltas.size() - 1))];
        INFO("|ΔPQ| p99,9 = " << p999 * 65535.0F << " codes 16 bits, max = " << deltas.back() * 65535.0F);
        CHECK(p999 <= std::ldexp(1.0F, -12));
        CHECK(deltas.back() <= std::ldexp(1.0F, -11));
      }
    }
  }

  // La lecture trilinéaire d'une LUT n³ (R le plus rapide) comme la texture du noyau : c × (n − 1).
  struct HostLut {
    const std::vector<float>* rgb;
    int n;
    kernels::grade::V3 operator()(kernels::grade::V3 c) const {
      const std::array<float, 3> at{c.r, c.g, c.b};
      std::array<int, 3> lo{};
      std::array<float, 3> t{};
      for (std::size_t k = 0; k < 3; ++k) {
        const float x = std::clamp(at[k], 0.0F, 1.0F) * static_cast<float>(n - 1);
        lo[k] = std::min(static_cast<int>(x), n - 2);
        t[k] = x - static_cast<float>(lo[k]);
      }
      std::array<float, 3> out{};
      for (int corner = 0; corner < 8; ++corner) {
        const int d[3] = {corner & 1, (corner >> 1) & 1, (corner >> 2) & 1};
        float w = 1.0F;
        for (std::size_t k = 0; k < 3; ++k) w *= d[k] != 0 ? t[k] : 1.0F - t[k];
        const auto un = [](int v) { return static_cast<std::size_t>(v); };
        const std::size_t i = ((un(lo[2] + d[2]) * un(n) + un(lo[1] + d[1])) * un(n) + un(lo[0] + d[0])) * 3;
        for (std::size_t ch = 0; ch < 3; ++ch) out[ch] += w * (*rgb)[i + ch];
      }
      return {out[0], out[1], out[2]};
    }
  }
}

TEST_CASE("grade de l'anneau v2 : la LUT de la courbe et le fondu contre la référence",
          "[backend_tensorrt][cuda][grade_ring]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 250;
  constexpr int h = 140;
  constexpr int n = 33;
  const kref::GradeInput input = kref::sdr_input();
  const std::vector<float> curve = warm_lut(n);
  std::vector<float> fade(curve.size());
  for (std::size_t i = 0; i < fade.size(); ++i) fade[i] = 0.8F * curve[i];  // la fin du fondu, plus sombre
  const std::vector<std::uint16_t> source = to_rgba16(scene(w, h));
  const auto pq = kref::apply_input(from_rgba16(source, w, h), input);
  REQUIRE(pq);
  DeviceFrame dev(w, h, input, curve, n);
  DeviceFrame end(w, h, input, fade, n);
  for (const float mix : {0.0F, 0.3F, 1.0F}) {
    INFO("mélange " << mix);
    dev.upload(source);
    gpu::cuda::GradeFrame f = dev.frame(kernels::GradeUniforms{}, input, false, n);
    f.curve = dev.lut;
    f.curve_fade = mix > 0.0F ? end.lut : 0;
    f.curve_size = n;
    f.curve_mix = mix;
    REQUIRE(gpu::cuda::launch_grade_lut_slot(f, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<std::uint16_t> got = dev.download();
    const HostLut a{&curve, n};
    const HostLut b{&fade, n};
    std::vector<float> deltas;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const kernels::grade::V3 e{pq->planes[0].at(x, y), pq->planes[1].at(x, y), pq->planes[2].at(x, y)};
        const kernels::grade::V3 want = kernels::grade::curve_lut_code(e, a, b, mix);
        const float wanted[3] = {want.r, want.g, want.b};
        for (std::size_t ch = 0; ch < 3; ++ch) {
          const float v = static_cast<float>(
                              got[(static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)) * 4 + ch]) /
                          65535.0F;
          deltas.push_back(std::abs(v - std::clamp(wanted[ch], 0.0F, 1.0F)));
        }
      }
    std::ranges::sort(deltas);
    const float p999 = deltas[static_cast<std::size_t>(0.999 * static_cast<double>(deltas.size() - 1))];
    INFO("|ΔPQ| p99,9 = " << p999 * 65535.0F << " codes 16 bits, max = " << deltas.back() * 65535.0F);
    CHECK(p999 <= std::ldexp(1.0F, -12));
  }
  gpu::cuda::GradeFrame empty;
  CHECK(gpu::cuda::launch_grade_lut_slot(empty, nullptr) == cudaErrorInvalidValue);
}

TEST_CASE("grade de l'anneau : entrées refusées", "[backend_tensorrt][cuda][grade_ring]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  gpu::cuda::GradeFrame f;
  CHECK(gpu::cuda::launch_grade_local(f, nullptr) == cudaErrorInvalidValue);
  CHECK(gpu::cuda::launch_grade_area(f, 0.04F, true, nullptr) == cudaErrorInvalidValue);
  CHECK(gpu::cuda::launch_grade_slot(f, nullptr) == cudaErrorInvalidValue);
}

TEST_CASE("grade de l'anneau : coût en 1440p et 2160p", "[.grade_ring][backend_tensorrt][cuda]") {
  // Critères du chantier 2.8 : ≤ 0,30 ms en 1440p, ≤ 0,60 ms en 2160p, réglages par défaut et vrai noir
  // allumé. Les registres et l'absence de débordement se lisent au journal de ptxas (-Xptxas -v,
  // CMakeLists.txt).
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const kref::GradeInput input = kref::sdr_input();
  const std::vector<float> lut = warm_lut(17);
  for (const auto [w, h, limit] : {std::array{2560, 1440, 30}, std::array{3840, 2160, 60}}) {
    DeviceFrame dev(w, h, input, lut, 17);
    dev.upload(to_rgba16(scene(w, h)));
    kernels::GradeUniforms u;
    u.black_lo = 0.002F;
    u.black_hi = 0.02F;
    const gpu::cuda::GradeFrame f = dev.frame(u, input, false, 0);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    REQUIRE(cudaEventCreate(&start) == cudaSuccess);
    REQUIRE(cudaEventCreate(&stop) == cudaSuccess);
    constexpr int kRuns = 200;
    for (int i = 0; i < 20; ++i) {
      REQUIRE(gpu::cuda::launch_grade_local(f, nullptr) == cudaSuccess);
      REQUIRE(gpu::cuda::launch_grade_area(f, 1.0F / 24.0F, false, nullptr) == cudaSuccess);
      REQUIRE(gpu::cuda::launch_grade_slot(f, nullptr) == cudaSuccess);
    }
    REQUIRE(cudaEventRecord(start, nullptr) == cudaSuccess);
    for (int i = 0; i < kRuns; ++i) {
      REQUIRE(gpu::cuda::launch_grade_local(f, nullptr) == cudaSuccess);
      REQUIRE(gpu::cuda::launch_grade_area(f, 1.0F / 24.0F, false, nullptr) == cudaSuccess);
      REQUIRE(gpu::cuda::launch_grade_slot(f, nullptr) == cudaSuccess);
    }
    REQUIRE(cudaEventRecord(stop, nullptr) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(stop) == cudaSuccess);
    float ms = 0.0F;
    REQUIRE(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess);
    const float per = ms / kRuns;
    // Le détail par noyau, pour savoir où va le temps (carte locale, zone, pixel).
    const auto timed = [&](auto&& launch) {
      REQUIRE(cudaEventRecord(start, nullptr) == cudaSuccess);
      for (int i = 0; i < kRuns; ++i) REQUIRE(launch() == cudaSuccess);
      REQUIRE(cudaEventRecord(stop, nullptr) == cudaSuccess);
      REQUIRE(cudaEventSynchronize(stop) == cudaSuccess);
      float total = 0.0F;
      REQUIRE(cudaEventElapsedTime(&total, start, stop) == cudaSuccess);
      return total / kRuns;
    };
    const float local_ms = timed([&] { return gpu::cuda::launch_grade_local(f, nullptr); });
    const float area_ms =
        timed([&] { return gpu::cuda::launch_grade_area(f, 1.0F / 24.0F, false, nullptr); });
    const float slot_ms = timed([&] { return gpu::cuda::launch_grade_slot(f, nullptr); });
    WARN(w << "x" << h << " : " << per << " ms par image (locale " << local_ms << ", zone " << area_ms
           << ", pixel et vrai noir " << slot_ms << ")");
    CHECK(per <= static_cast<float>(limit) / 100.0F);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
}
