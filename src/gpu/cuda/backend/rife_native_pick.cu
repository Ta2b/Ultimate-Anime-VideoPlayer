// Les noyaux du RIFE natif que le réglage au chargement peut choisir (rife_native_plan.h) : une instance de
// conv_kernel par forme (tranche, tuile, lignes par warp, poids à l'avance) et par sommes. Les formes d'une
// ligne par warp (tuile de 4 × 32 pixels) et celles des poids à l'avance n'existent qu'en sommes f16 et
// mixtes, en pas 1 (les poids à l'avance : tranche de 32 ou 64 seulement) : le f32 ne sert qu'aux têtes et au
// repli, et la taille du module reste bornée.
#include <algorithm>
#include <cstddef>
#include <vector>

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::rife_native {
namespace {

template <int Kc, int CT, int KH, int S, int Rows, In I, Out O, Acc A, bool Ahead>
void launch_one(dim3 grid, int smem, cudaStream_t stream, const Params& p) {
  conv_kernel<Kc, CT, KH, S, Rows, I, O, A, Ahead>
      <<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

template <int Kc, int KH, int S, int Rows, In I, Out O, Acc A, bool Ahead = false>
Launch by_tile(int ct) {
  if (ct == 16) return launch_one<Kc, 16, KH, S, Rows, I, O, A, Ahead>;
  if (ct == 32) return launch_one<Kc, 32, KH, S, Rows, I, O, A, Ahead>;
  if constexpr (A == Acc::kF16) {
    if (ct == 64) return launch_one<Kc, 64, KH, S, Rows, I, O, A, Ahead>;
  }
  return nullptr;
}

// Pas 1 (3 × 3, ou 2 × 2 d'une phase de transposée) : deux lignes par warp, ou une hors f32.
template <int Kc, int KH, Out O, Acc A, bool Ahead = false>
Launch by_rows(int rows, int ct) {
  if (rows == 2) return by_tile<Kc, KH, 1, 2, In::kNhwc, O, A, Ahead>(ct);
  if constexpr (A != Acc::kF32) {
    if (rows == 1) return by_tile<Kc, KH, 1, 1, In::kNhwc, O, A, Ahead>(ct);
  }
  return nullptr;
}

// Les poids à l'avance : 3 × 3 de pas 1 (les ResConv), tranche de 32 ou 64, hors f32.
template <Acc A>
Launch ahead_of(Kind kind, int kc, int ct, int rows) {
  if constexpr (A == Acc::kF32) {
    return nullptr;
  } else {
    if (kind != Kind::kSame) return nullptr;
    if (kc == 32) return by_rows<32, 3, Out::kNhwc, A, true>(rows, ct);
    if (kc == 64) return by_rows<64, 3, Out::kNhwc, A, true>(rows, ct);
    return nullptr;
  }
}

template <Acc A>
Launch pick(Kind kind, int kc, int ct, int rows, bool ahead) {
  if (ahead) return ahead_of<A>(kind, kc, ct, rows);
  switch (kind) {
    case Kind::kSame:
      if (kc == 16) return by_rows<16, 3, Out::kNhwc, A>(rows, ct);
      if (kc == 32) return by_rows<32, 3, Out::kNhwc, A>(rows, ct);
      if (kc == 64) return by_rows<64, 3, Out::kNhwc, A>(rows, ct);
      return nullptr;
    case Kind::kHalf:
      if (rows != 1) return nullptr;
      if (kc == 16) return by_tile<16, 3, 2, 1, In::kNhwc, Out::kNhwc, A>(ct);
      if (kc == 32) return by_tile<32, 3, 2, 1, In::kNhwc, Out::kNhwc, A>(ct);
      return nullptr;
    case Kind::kImage:
      return kc == 16 && rows == 1 ? by_tile<16, 3, 2, 1, In::kImage, Out::kNhwc, A>(ct) : nullptr;
    case Kind::kPhase:
      if (kc == 16) return by_rows<16, 2, Out::kPhase, A>(rows, ct);
      if (kc == 32) return by_rows<32, 2, Out::kPhase, A>(rows, ct);
      if (kc == 64) return by_rows<64, 2, Out::kPhase, A>(rows, ct);
      return nullptr;
    case Kind::kPlanes4:
      return kc == 16 && ct == 16 && rows == 2
                 ? launch_one<16, 16, 3, 1, 2, In::kNhwc, Out::kPlanes4, A, false>
                 : nullptr;
  }
  return nullptr;
}

}  // namespace

Launch pick_for(RifeNativeSums sums, Kind kind, int kc, int ct, int rows, bool ahead) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return pick<Acc::kF16>(kind, kc, ct, rows, ahead);
    case RifeNativeSums::kMixed:
      return pick<Acc::kMixed>(kind, kc, ct, rows, ahead);
    case RifeNativeSums::kF32:
      return pick<Acc::kF32>(kind, kc, ct, rows, ahead);
  }
  return nullptr;
}

std::vector<Config> configs(RifeNativeSums sums, Kind kind, const Extent& extent, int sms) {
  std::vector<Config> out;
  const int taps = kernel_size(kind) * kernel_size(kind);
  const bool split = kind != Kind::kImage && kind != Kind::kPlanes4;
  for (const int kc : {64, 32, 16})
    for (const int ct : {64, 32, 16})
      for (const int rows : {2, 1})
        for (const bool ahead : {false, true}) {
          Config cfg{.kc = kc, .ct = ct, .rows = rows, .ksplit = 1, .ahead = ahead};
          if (extent.cin % kc != 0 || extent.cout % ct != 0 ||
              pick_for(sums, kind, kc, ct, rows, ahead) == nullptr)
            continue;
          out.push_back(cfg);
          const int th = kWarps * rows;
          const long blocks = static_cast<long>((extent.ow + kTW - 1) / kTW) * ((extent.oh + th - 1) / th) *
                              extent.phases * (extent.cout / ct);
          const long want = 2L * sms;
          if (!split || blocks >= want) continue;
          cfg.ksplit = static_cast<int>(
              std::min((want + blocks - 1) / blocks, static_cast<long>(extent.cin / kc) * taps));
          if (cfg.ksplit > 1) out.push_back(cfg);
        }
  return out;
}

const char* sums_name(RifeNativeSums sums) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return "f16";
    case RifeNativeSums::kMixed:
      return "mixtes";
    case RifeNativeSums::kF32:
      return "f32";
  }
  return "?";
}

void launch_reduce(bool phase, int width, cudaStream_t stream, const Params& p) {
  const long count = static_cast<long>(p.oh) * p.ow * (width / 8);
  const auto blocks = static_cast<unsigned>((count + 255) / 256);
  if (phase)
    reduce_kernel<Out::kPhase><<<blocks, 256, 0, stream>>>(p, width);
  else
    reduce_kernel<Out::kNhwc><<<blocks, 256, 0, stream>>>(p, width);
}

}  // namespace kirari::gpu::cuda::rife_native
