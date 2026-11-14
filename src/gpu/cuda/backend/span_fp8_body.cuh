#pragma once
// Le gabarit du réseau SPAN FP8 (span_fp8_body.h) : Ch canaux (64, ou 48 sans rembourrage), l'algorithme
// des convolutions 3×3 à Ch canaux d'entrée (BlockAlg : kOptimized, K par position du filtre, ou
// kFewChannels, K mis à plat) et Fp16 : les blocs en FP16 sans quantification (SPAN FP16 en noyaux
// CUTLASS, lot 2 du fil SPAN), activations FP16 et poids du paquet tels quels, échelles ignorées. Le graphe
// des noyaux est celui du banc tools/bench/fp8/cutlass/span_net.cu ; voir docs/span-fp8-cutlass.md. Les poids
// que reçoit upload sont déjà à la largeur Ch (span_fp8_net.cu).
#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "gpu/cuda/backend/span_fp8_body.h"
#include "gpu/cuda/backend/span_fp8_kernels.cuh"
#include "gpu/cuda/backend/span_fp8_layout.cuh"
#include "gpu/span_fp8/quantize.h"

namespace ajp::gpu::cuda {
namespace {

namespace sk = span_fp8_kernels;
using Alg = cutlass::conv::IteratorAlgorithm;

// Un produit de deux E4M3 au maximum (448 × 448) dépasse le plus grand f16 (65504) : à accumulateur f16, le
// maximum de chaque canal de poids est rangé à 448 / 16 = 28. E4M3 étant flottant, la précision relative ne
// change pas ; seuls les poids sous 1/1792 du maximum de leur canal passent en dénormaux. La porte du reçu
// juge la variante (keep_acc16) : des sommes qui débordent font échouer sa validation, et le paquet garde le
// f32.
constexpr float kAcc16Headroom = 16.F;
constexpr int kPadded = 16;  // conv_1 lit 16 canaux (12 + 4 nuls)
constexpr int kPixel = 48;

// Poids FP8 d'un bloc, rangés [sortie][ligne][colonne][entrée] : échelle par canal de sortie (étape 10.3,
// gpu/span_fp8/quantize.h) « maximum du canal / 448 » (1 pour un canal nul), arrondi au plus proche, saturé.
// headroom > 1 : l'échelle est multipliée d'autant (maximum du canal rangé à 448 / headroom), pour les blocs
// à accumulateur f16.
inline std::vector<float> quantize_fp8(const std::uint16_t* half_bits, std::size_t count, std::size_t outs,
                                       std::vector<std::uint8_t>& out, float headroom) {
  const std::size_t per = count / outs;
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    __half_raw raw;
    raw.x = half_bits[i];
    values[i] = __half2float(__half(raw));
  }
  // Les échelles de la référence hôte testée (quantize.h), pas une copie (A02-09).
  std::vector<float> scales = span_fp8::channel_scales(values, outs);
  for (float& scale : scales) scale *= headroom;
  out.resize(values.size());
  for (std::size_t c = 0; c < outs; ++c) {
    for (std::size_t i = c * per; i < (c + 1) * per; ++i)
      out[i] =
          static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(values[i] / scales[c], __NV_SATFINITE, __NV_E4M3));
  }
  return scales;
}

constexpr std::int64_t kAlign = 256;
constexpr std::int64_t align_up(std::int64_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

template <int Ch, Alg BlockAlg, bool Fp16>
class Body final : public SpanFp8Body {
  using E4 = cutlass::float_e4m3_t;
  using F16 = cutlass::half_t;
  static constexpr int kBlocks = SpanFp8Weights::kBlocks;
  static constexpr int kConcat = 4 * Ch;
  static constexpr int kScaleCount = SpanFp8Weights::kScales;

  // Blocs : entrée FP8 (ou FP16), Ch canaux. Hors blocs : entrée FP16, accumulateur f32.
  template <class Acc>
  using CB = std::conditional_t<Fp16, sk::Cfg<F16, 128, 32, 32, 4, Acc, BlockAlg, Ch>,
                                sk::Cfg<E4, 128, 64, 32, 3, Acc, BlockAlg, Ch>>;
  using EB = typename CB<float>::E;          // élément des activations des blocs
  static constexpr int kAct = Fp16 ? 2 : 1;  // octets par canal d'activation des blocs
  using C8 = CB<float>;
  using L8 = sk::Launcher<C8, 0, true, Fp16>;
  using LT = sk::Launcher<C8, 1, false, Fp16>;
  using C16A = sk::Cfg<F16, 128, 32, 32, 4, float, Alg::kAnalytic, Ch>;
  using C16 = sk::Cfg<F16, 128, 32, 32, 4, float, Alg::kOptimized, Ch>;
  using C16P = sk::Cfg<F16, 128, 32, 32, 4, float, BlockAlg>;  // to_pixel : Ch canaux, kPixel colonnes
  using LC1 = sk::Launcher<C16A, 0, false, Fp16>;              // sa sortie out8 entre au bloc 1
  using LCAT = sk::Launcher<C16, 0, false>;
  using LPIX = sk::Launcher<C16P, 2, false>;  // to_pixel, DepthToSpace fondu (étape 3.4a)
  // Étape 3.5a : les mêmes, accumulateur f16 (mma m16n8k16 .f16.f16.f16.f16), choisis par le paquet.
  using C16H = sk::Cfg<F16, 128, 32, 32, 4, F16, Alg::kOptimized, Ch>;
  using C16PH = sk::Cfg<F16, 128, 32, 32, 4, F16, BlockAlg>;
  using LCATH = sk::Launcher<C16H, 0, false>;
  using LPIXH = sk::Launcher<C16PH, 2, false>;
  // Les blocs FP8 à accumulateur f16 (mma m16n8k32 .f16.e4m3.e4m3.f16) avec le même drapeau : sur les
  // GeForce Ada, l'accumulation f32 des mma FP8 tourne à la moitié du débit de l'accumulation f16 (piste N3
  // de la deuxième passe TensorRT). Les poids y sont rangés kAcc16Headroom fois plus bas.
  using C8H = CB<F16>;
  using L8H = sk::Launcher<C8H, 0, true, Fp16>;
  using LTH = sk::Launcher<C8H, 1, false, Fp16>;

 public:
  Body(int height, int width)
      : height_(height), width_(width), h_(height / 2), w_(width / 2), npix_(std::int64_t{h_} * w_) {}

  bool upload(const SpanFp8Weights& weights, SpanFp8Status& status, std::string& error) override;
  std::int64_t workspace_bytes() const override {
    // concaténation FP16 de 4·Ch canaux + quatre tampons d'activation de Ch canaux (FP8, ou FP16) ;
    // dégroupage et sortie de conv_cat recouvrent des tampons alors inactifs. to_pixel écrit droit dans la
    // sortie.
    const std::int64_t rows = tiled();
    return 2 * kAlign + align_up(rows * kConcat * 2) + align_up(rows * 4 * Ch * kAct);
  }
  void bind(void* base) override;
  bool launch_body(cudaStream_t stream, const void* input, std::string& error) override;
  bool launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring, std::string& error) override;

 private:
  struct BlockWeights {
    std::array<Device, 3> w, b;
    Device att_w, att_b;
    std::array<Device, 3> alpha;  // Ch flottants chacun : échelle de l'activation × échelle du canal
  };

  // Les tampons à un multiple de 384 lignes, que les tuiles M de 128 et de 96 (Cfg::kTM, 48 colonnes)
  // divisent : la dernière tuile reste dans son tampon même si une garde manquait (étape 3.1).
  std::int64_t tiled() const { return (npix_ + 383) / 384 * 384; }
  // L'échelle de l'activation qui entre dans la convolution i du bloc b (s_in, s_r1, s_r2).
  float act_scale(std::size_t b, std::size_t i) const {
    if (i > 0) return scales_[i + 3 * b];
    return b == 0 ? scales_[0] : scales_[1 + 3 * (b - 1) + 2];
  }
  bool upload_block(std::size_t b, const SpanFp8Weights& weights, SpanFp8Status& status, std::string& error);
  // L'échelle par colonne de la convolution i d'un bloc ; nulle en FP16 (alpha de KP, 1).
  const float* col_scale(const BlockWeights& bw, std::size_t i) const {
    return Fp16 ? nullptr : bw.alpha[i].template as<float>();
  }
  void bind_block(std::size_t b, std::uint8_t* const (&x)[2], std::uint8_t* r1, std::uint8_t* r2,
                  __half* cat_in);
  bool failed(cudaStream_t stream, std::string& error) const;

  int height_, width_, h_, w_;
  std::int64_t npix_;
  Device c1_w_, c1_b_, cat_w_, cat_b_, pix_w_, pix_b_;
  std::array<BlockWeights, kBlocks> blocks_;
  int used_ = kBlocks;  // blocs du modèle (étape 3.2c)
  std::array<float, kScaleCount> scales_{};
  bool acc16_ = false;

  // Lancements : leurs paramètres portent les adresses de la zone de travail (bind).
  LC1 c1_;
  std::array<L8, kBlocks> la_, lb_;
  std::array<LT, kBlocks> lc_;
  LCAT cat_;
  LPIX pix_;
  LCATH cat16_;
  LPIXH pix16_;
  std::array<L8H, kBlocks> la16_, lb16_;  // blocs à accumulateur f16 (acc16)
  std::array<LTH, kBlocks> lc16_;
  void* workspace_ = nullptr;
  __half* unshuffled_ = nullptr;  // recouvre r1 (posé par bind)
  cudaError_t attribute_ = cudaSuccess;
};

template <int Ch, Alg BlockAlg, bool Fp16>
bool Body<Ch, BlockAlg, Fp16>::upload(const SpanFp8Weights& weights, SpanFp8Status& status,
                                      std::string& error) {
  used_ = weights.blocks;
  acc16_ = weights.acc16;
  scales_.fill(1.F);
  if (!Fp16)
    for (std::size_t i = 0; i < static_cast<std::size_t>(1 + 3 * used_); ++i) scales_[i] = weights.scales[i];

  // false : l'envoi a échoué, status et error le disent.
  const auto sent = [&](cudaError_t result, const char* what) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    status = result == cudaErrorMemoryAllocation ? SpanFp8Status::OutOfMemory : SpanFp8Status::Failed;
    error = std::string(what) + " : " + cudaGetErrorString(result);
    return false;
  };
  const auto put_half = [&](Device& to, const SpanFp8ConvView& conv, std::size_t count) {
    return sent(to.upload(conv.weights, count * sizeof(std::uint16_t)), "poids");
  };
  const auto put_bias = [&](Device& to, const SpanFp8ConvView& conv, std::size_t count) {
    return sent(to.upload(conv.bias, count * sizeof(float)), "biais");
  };
  if (!put_half(c1_w_, weights.first, std::size_t{Ch} * 9 * kPadded) || !put_bias(c1_b_, weights.first, Ch) ||
      !put_half(cat_w_, weights.concat, std::size_t{Ch} * kConcat) || !put_bias(cat_b_, weights.concat, Ch) ||
      !put_half(pix_w_, weights.pixel, std::size_t{kPixel} * 9 * Ch) ||
      !put_bias(pix_b_, weights.pixel, kPixel))
    return false;
  for (std::size_t b = 0; b < static_cast<std::size_t>(used_); ++b)
    if (!upload_block(b, weights, status, error)) return false;
  status = SpanFp8Status::Ok;
  return true;
}

// Les poids du bloc b : FP8 à l'échelle de leur canal (ou FP16 tels quels), biais, attention.
template <int Ch, Alg BlockAlg, bool Fp16>
bool Body<Ch, BlockAlg, Fp16>::upload_block(std::size_t b, const SpanFp8Weights& weights,
                                            SpanFp8Status& status, std::string& error) {
  const auto sent = [&](cudaError_t result, const char* what) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    status = result == cudaErrorMemoryAllocation ? SpanFp8Status::OutOfMemory : SpanFp8Status::Failed;
    error = std::string(what) + " : " + cudaGetErrorString(result);
    return false;
  };
  constexpr std::size_t kBlockWeights = std::size_t{Ch} * 9 * Ch;
  BlockWeights& bw = blocks_[b];
  for (std::size_t i = 0; i < 3; ++i) {
    const SpanFp8ConvView& conv = weights.conv[b][i];
    if (!sent(bw.b[i].upload(conv.bias, Ch * sizeof(float)), "biais")) return false;
    if (Fp16) {
      if (!sent(bw.w[i].upload(conv.weights, kBlockWeights * sizeof(std::uint16_t)), "poids")) return false;
      continue;
    }
    std::vector<std::uint8_t> q;
    const std::vector<float> w_scale =
        quantize_fp8(conv.weights, kBlockWeights, Ch, q, acc16_ ? kAcc16Headroom : 1.F);
    std::array<float, Ch> alpha{};
    for (std::size_t c = 0; c < alpha.size(); ++c) alpha[c] = act_scale(b, i) * w_scale[c];
    if (!sent(bw.w[i].upload(q.data(), q.size()), "poids FP8") ||
        !sent(bw.alpha[i].upload(alpha.data(), sizeof(alpha)), "échelles des poids FP8"))
      return false;
  }
  const SpanFp8ConvView& att = weights.attention[b];
  return sent(bw.att_w.upload(att.weights, std::size_t{Ch} * Ch * sizeof(std::uint16_t)), "poids") &&
         sent(bw.att_b.upload(att.bias, Ch * sizeof(float)), "biais");
}

template <int Ch, Alg BlockAlg, bool Fp16>
void Body<Ch, BlockAlg, Fp16>::bind_block(std::size_t b, std::uint8_t* const (&x)[2], std::uint8_t* r1,
                                          std::uint8_t* r2, __half* cat_in) {
  const bool last = b + 1 == static_cast<std::size_t>(used_);
  const std::uint8_t* xin = x[b % 2];
  std::uint8_t* xout = x[(b + 1) % 2];
  BlockWeights& bw = blocks_[b];
  const auto init = [&](auto& plain, auto& half, const std::uint8_t* in, std::size_t i, const sk::KP& k) {
    const auto* in8 = reinterpret_cast<const EB*>(in);
    if (acc16_)
      half.init(h_, w_, 3, 1, Ch, Ch, in8, bw.w[i].template as<EB>(), k);
    else
      plain.init(h_, w_, 3, 1, Ch, Ch, in8, bw.w[i].template as<EB>(), k);
  };
  {  // c1 + Relu -> FP8 ; pour le dernier bloc, aussi FP16 dans la tranche 3 (son relu1)
    sk::KP k;
    k.alpha_col = col_scale(bw, 0);
    k.bias = bw.b[0].template as<float>();
    k.out8 = r1;
    k.inv_s8 = 1.F / scales_[1 + 3 * b];
    if (last) {
      k.out16 = cat_in + 3 * Ch;
      k.ld16 = kConcat;
    }
    init(la_[b], la16_[b], xin, 0, k);
  }
  {
    sk::KP k;
    k.alpha_col = col_scale(bw, 1);
    k.bias = bw.b[1].template as<float>();
    k.out8 = r2;
    k.inv_s8 = 1.F / scales_[2 + 3 * b];
    init(lb_[b], lb16_[b], r1, 1, k);
  }
  {  // fin de bloc : c3, attention 1x1, tanh, saut, produit
    sk::KP k;
    k.skip = xin;
    k.skip_scale = act_scale(b, 0);
    k.alpha_col = col_scale(bw, 2);
    k.bias = bw.b[2].template as<float>();
    k.w1 = bw.att_w.template as<__half>();
    k.b1 = bw.att_b.template as<float>();
    if (!last) {
      k.out8 = xout;
      k.inv_s8 = 1.F / scales_[3 + 3 * b];
    }
    if (b == 0) {  // sortie du bloc 1 : tranche 2
      k.out16 = cat_in + 2 * Ch;
      k.ld16 = kConcat;
    }
    if (last) {  // sortie du dernier bloc : tranche 1
      k.out16 = cat_in + Ch;
      k.ld16 = kConcat;
    }
    init(lc_[b], lc16_[b], r2, 2, k);
  }
}

template <int Ch, Alg BlockAlg, bool Fp16>
void Body<Ch, BlockAlg, Fp16>::bind(void* base) {
  workspace_ = base;
  auto* bytes = static_cast<std::uint8_t*>(base);
  const std::int64_t rows = tiled();
  __half* cat_in = reinterpret_cast<__half*>(bytes);
  std::uint8_t* region = bytes + align_up(rows * kConcat * 2);
  const std::int64_t act = rows * Ch * kAct;  // un tampon d'activation des blocs
  std::uint8_t* const x[2] = {region, region + act};
  std::uint8_t* r1 = region + 2 * act;
  std::uint8_t* r2 = region + 3 * act;
  // Le dégroupage (16 canaux FP16, 32 o) et la sortie de conv_cat (Ch canaux FP16, 2·Ch o) tiennent dans r1
  // et r2 qui se suivent (2·Ch·kAct o par ligne).
  unshuffled_ = reinterpret_cast<__half*>(r1);
  auto* cat_out = reinterpret_cast<__half*>(r1);

  {  // conv_1 : sorties FP8 (entrée du bloc 1) et FP16 (tranche 0 de la concaténation)
    sk::KP k;
    k.bias = c1_b_.as<float>();
    k.out8 = x[0];
    k.inv_s8 = 1.F / scales_[0];
    k.out16 = cat_in;
    k.ld16 = kConcat;
    c1_.init(h_, w_, 3, 1, kPadded, Ch, reinterpret_cast<const F16*>(unshuffled_), c1_w_.as<F16>(), k);
  }
  for (std::size_t b = 0; b < static_cast<std::size_t>(used_); ++b) bind_block(b, x, r1, r2, cat_in);
  {
    sk::KP k;
    k.bias = cat_b_.as<float>();
    k.out16 = cat_out;
    k.ld16 = Ch;
    sk::KP p;  // sortie et surface posées à chaque lancement (launch_tail)
    p.bias = pix_b_.as<float>();
    const auto* cat_src = reinterpret_cast<const F16*>(cat_in);
    const auto* pix_src = reinterpret_cast<const F16*>(cat_out);
    if (acc16_) {
      cat16_.init(h_, w_, 1, 0, kConcat, Ch, cat_src, cat_w_.as<F16>(), k);
      pix16_.init(h_, w_, 3, 1, Ch, kPixel, pix_src, pix_w_.as<F16>(), p);
    } else {
      cat_.init(h_, w_, 1, 0, kConcat, Ch, cat_src, cat_w_.as<F16>(), k);
      pix_.init(h_, w_, 3, 1, Ch, kPixel, pix_src, pix_w_.as<F16>(), p);
    }
  }
  attribute_ = c1_.attribute;
  for (std::size_t b = 0; b < static_cast<std::size_t>(used_); ++b)
    for (const cudaError_t e : {la_[b].attribute, lb_[b].attribute, lc_[b].attribute, la16_[b].attribute,
                                lb16_[b].attribute, lc16_[b].attribute})
      if (attribute_ == cudaSuccess) attribute_ = e;
  for (const cudaError_t e : {cat_.attribute, pix_.attribute, cat16_.attribute, pix16_.attribute})
    if (attribute_ == cudaSuccess) attribute_ = e;
}

// Vrai si un lancement de stream a échoué, avec le motif dans error.
template <int Ch, Alg BlockAlg, bool Fp16>
bool Body<Ch, BlockAlg, Fp16>::failed(cudaStream_t /*stream*/, std::string& error) const {
  const cudaError_t result = cudaGetLastError();
  if (result == cudaSuccess) return false;
  error = std::string("lancement des noyaux : ") + cudaGetErrorString(result);
  return true;
}

template <int Ch, Alg BlockAlg, bool Fp16>
bool Body<Ch, BlockAlg, Fp16>::launch_body(cudaStream_t stream, const void* input, std::string& error) {
  if (workspace_ == nullptr) {
    error = "zone de travail absente";
    return false;
  }
  if (attribute_ != cudaSuccess) {
    error = std::string("mémoire partagée des noyaux refusée : ") + cudaGetErrorString(attribute_);
    return false;
  }
  static_cast<void>(cudaGetLastError());  // une erreur d'un appel d'avant n'est pas celle de ces lancements
  const int grid = static_cast<int>((npix_ + 255) / 256);
  unshuffle_kernel<<<grid, 256, 0, stream>>>(static_cast<const __half*>(input), unshuffled_, height_, width_);
  c1_.launch(stream);
  for (std::size_t b = 0; b < static_cast<std::size_t>(used_); ++b) {
    if (acc16_) {
      la16_[b].launch(stream);
      lb16_[b].launch(stream);
      lc16_[b].launch(stream);
    } else {
      la_[b].launch(stream);
      lb_[b].launch(stream);
      lc_[b].launch(stream);
    }
  }
  if (acc16_)
    cat16_.launch(stream);
  else
    cat_.launch(stream);
  return !failed(stream, error);
}

template <int Ch, Alg BlockAlg, bool Fp16>
bool Body<Ch, BlockAlg, Fp16>::launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring,
                                           std::string& error) {
  if (workspace_ == nullptr) {
    error = "zone de travail absente";
    return false;
  }
  if (output == nullptr && ring.surface.object == 0) {
    error = "ni sortie ni surface";
    return false;
  }
  const auto tail = [&](auto& pix) {
    pix.kp.out16 = static_cast<__half*>(output);
    pix.kp.surface = ring.surface;
    pix.kp.surf_w = ring.width;
    pix.kp.surf_h = ring.height;
    pix.launch(stream);
  };
  if (acc16_)
    tail(pix16_);
  else
    tail(pix_);
  return !failed(stream, error);
}

}  // namespace
}  // namespace ajp::gpu::cuda
