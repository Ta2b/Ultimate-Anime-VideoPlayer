#include "gpu/cuda/compute/transfer.h"

#include <cmath>
#include <cstddef>
#include <format>

#include "gpu/cuda/compute/kernels/launch.h"
#include "kernels/spec/color.h"

namespace ajp::gpu::cuda {

Result<dev::TransferParams> make_transfer(const TransferSpec& spec) {
  dev::TransferParams p{.decode = spec.decode, .encode = spec.encode, .gain = spec.gain};
  if (!std::isfinite(spec.gain))
    return refuse(RefusalCode::FieldOutOfRange, std::format("gain {} non fini", spec.gain));
  if (spec.hlg_ootf_peak) {
    const float peak = *spec.hlg_ootf_peak;
    if (!(peak >= kernels::hlg::kMinPeak && peak <= kernels::hlg::kMaxPeak)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("pic HLG {} cd/m² hors de [{}, {}]", peak, kernels::hlg::kMinPeak,
                                kernels::hlg::kMaxPeak));
    }
    p.hlg_ootf = true;
    p.hlg_peak = peak;
    p.hlg_gamma = kernels::hlg::kGammaAt1000 + kernels::hlg::kGammaSlope * std::log10(peak / 1000.0F);
    const kernels::Coefficients k = kernels::coefficients(kernels::Matrix::Bt2020Ncl);
    p.hlg_kr = k.kr;
    p.hlg_kb = k.kb;
  }
  if (spec.decode == Curve::Bt1886 || spec.encode == Curve::Bt1886) {
    const float white = spec.bt1886_white;
    const float black = spec.bt1886_black;
    if (!(black >= 0.0F && white > black && std::isfinite(white))) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("BT.1886 : blanc {} et noir {} invalides", white, black));
    }
    const float wg = std::pow(white, 1.0F / kernels::bt1886::kGamma);
    const float bg = std::pow(black, 1.0F / kernels::bt1886::kGamma);
    p.bt1886_white = white;
    p.bt1886_black = black;
    p.bt1886_a = std::pow(wg - bg, kernels::bt1886::kGamma);
    p.bt1886_b = bg / (wg - bg);
  }
  if (spec.primaries) {
    p.convert = true;
    for (int i = 0; i < 9; ++i)
      p.matrix[i] = static_cast<float>(spec.primaries->m[static_cast<std::size_t>(i)]);
  }
  return p;
}

Result<> transfer(const Tensor& in, const dev::TransferParams& params, const Tensor& out,
                  cudaStream_t stream) {
  if (!kernels::valid_plane_dimension(in.width) || !kernels::valid_plane_dimension(in.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("transfert : {}x{} hors de [1, {}]", in.width,
                                                         in.height, kernels::kMaxDimension));
  }
  if (auto ok = check_tensor(in, in.width, in.height, 3, "transfert"); !ok) return ok;
  if (auto ok = check_tensor(out, in.width, in.height, 3, "transfert"); !ok) return ok;
  return cuda_check(dev::apply_transfer(in, params, out, stream), "lancement de apply_transfer");
}

}  // namespace ajp::gpu::cuda
