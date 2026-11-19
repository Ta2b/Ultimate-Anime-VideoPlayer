#pragma once
// Les bords du SPAN FP8 (conv_1, conv_cat, to_pixel, tous en FP16) par les noyaux du SPAN FP16 natif
// (docs/span-native16.md, « Le FP8 ») : le dégroupage fondu dans le chargement de conv_1, qui écrit en une
// passe l'entrée E4M3 du premier bloc et la tranche 0 de la concaténation ; conv_cat qui lit les quatre
// tranches ; to_pixel avec DepthToSpace et l'anneau. Les blocs restent les noyaux CUTLASS FP8. Pris par
// span_fp8_body.cuh quand SpanFp8Weights::native_edges (--span-native).
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/cuda/backend/span_fp8_layout.cuh"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_native16_kernels.cuh"

namespace kirari::gpu::cuda {
namespace {

template <int C>
class NativeEdges {
  using M = span_native::Mode;
  using S = span_native::Src;
  static constexpr int kPixel = span_native::kPixel;
  static constexpr int kImageCh = span_native::kImageCh;

 public:
  void size(int height, int width) {
    fh_ = height / 2;
    fw_ = width / 2;
    grid_ = dim3(static_cast<unsigned>((fw_ + span_native::kTW - 1) / span_native::kTW),
                 static_cast<unsigned>((fh_ + span_native::kTH - 1) / span_native::kTH));
  }

  // Les poids de conv_1 ([C][9][16]), conv_cat ([C][4C]) et to_pixel ([48][9][C]), rangés pour les mma.
  cudaError_t upload(const SpanFp8Weights& weights) {
    const auto put = [](Device& w, Device& b, const SpanFp8ConvView& view, int cout, int taps, int cin) {
      const std::vector<std::uint32_t> packed = span_native::pack(view.weights, cout, taps, cin);
      cudaError_t error = w.upload(packed.data(), packed.size() * sizeof(std::uint32_t));
      return error != cudaSuccess ? error
                                  : b.upload(view.bias, static_cast<std::size_t>(cout) * sizeof(float));
    };
    cudaError_t error = put(first_w_, first_b_, weights.first, C, 9, kImageCh);
    if (error == cudaSuccess) error = put(cat_w_, cat_b_, weights.concat, C, 1, 4 * C);
    if (error == cudaSuccess) error = put(pixel_w_, pixel_b_, weights.pixel, kPixel, 9, C);
    return error;
  }

  // conv_1 depuis l'image NCHW : x0 (E4M3, pas C, échelle inv_s8) et la tranche 0 de la concaténation (pas
  // ld).
  void first(cudaStream_t stream, const void* input, std::uint8_t* x0, float inv_s8, __half* slice0,
             int ld) const {
    span_native::Params p = params(first_w_, first_b_, slice0);
    p.in[0] = static_cast<const __half*>(input);
    p.out_ld = ld;
    p.out8 = x0;
    p.inv_s8 = inv_s8;
    constexpr int kSmem = span_native::smem_bytes(kImageCh, C, 9);
    span_native::conv_kernel<kImageCh, C, 9, 1, S::kImage, M::kPlain, false, span_native::Acc::kF32>
        <<<grid_, span_native::kThreads, kSmem, stream>>>(p);
  }

  // conv_cat : les quatre tranches de la concaténation (pas ld) vers out (C canaux). f16 : sommes en f16,
  // une somme débordée notée dans overflow (le mot du FP8, nul sans acc16), comme cat16_.
  void cat(cudaStream_t stream, const __half* concat, int ld, __half* out, bool f16, int* overflow) const {
    span_native::Params p = params(cat_w_, cat_b_, out);
    for (int s = 0; s < 4; ++s) p.in[s] = concat + s * C;
    p.in_ld = ld;
    p.overflow = overflow;
    constexpr int kSmem = span_native::smem_bytes(C, C, 1);
    if (f16)
      span_native::conv_kernel<C, C, 1, 4, S::kNhwc, M::kPlain, false, span_native::Acc::kF16>
          <<<grid_, span_native::kThreads, kSmem, stream>>>(p);
    else
      span_native::conv_kernel<C, C, 1, 4, S::kNhwc, M::kPlain, false, span_native::Acc::kF32>
          <<<grid_, span_native::kThreads, kSmem, stream>>>(p);
  }

  // to_pixel : sortie NCHW FP16 (ou nulle) et/ou l'emplacement de l'anneau ; f16 et overflow comme cat.
  void pixel(cudaStream_t stream, const __half* in, void* output, const SpanFp8Ring& ring, bool f16,
             int* overflow) const {
    span_native::Params p = params(pixel_w_, pixel_b_, static_cast<__half*>(output));
    p.in[0] = in;
    p.overflow = overflow;
    p.surface = ring.surface;
    p.surf_w = ring.width;
    p.surf_h = ring.height;
    constexpr int kSmem = span_native::smem_bytes(C, kPixel, 9);
    if (f16)
      span_native::conv_kernel<C, kPixel, 9, 1, S::kNhwc, M::kPixel, false, span_native::Acc::kF16>
          <<<grid_, span_native::kThreads, kSmem, stream>>>(p);
    else
      span_native::conv_kernel<C, kPixel, 9, 1, S::kNhwc, M::kPixel, false, span_native::Acc::kF32>
          <<<grid_, span_native::kThreads, kSmem, stream>>>(p);
  }

 private:
  span_native::Params params(const Device& w, const Device& b, __half* out) const {
    span_native::Params p;
    p.out = out;
    p.w = w.as<uint4>();
    p.bias = b.as<float>();
    p.fh = fh_;
    p.fw = fw_;
    return p;
  }

  int fh_ = 0, fw_ = 0;
  dim3 grid_;
  Device first_w_, first_b_, cat_w_, cat_b_, pixel_w_, pixel_b_;
};

}  // namespace
}  // namespace kirari::gpu::cuda
