// Les poids des noyaux de rife_native_kernels.cuh, rangés pour leurs fragments B (hôte) : RIFE natif et
// TemporalFix natif.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

namespace kirari::gpu::cuda::rife_native {

// Poids FP16 [n][taps][cin] (n : les canaux d'un paquet blockIdx.z) rangés dans l'ordre des fragments B de
// mma m16n8k16 : pas k (prise, tranche de 16 entrées), paire de tuiles n8, lane (g = lane / 4, t = lane % 4)
// ; 4 mots par lane : b0 et b1 de la tuile 2·np, puis de la tuile 2·np + 1. b0 = entrées 2t et 2t + 1 de la
// sortie 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9 ; la plus petite entrée dans la moitié basse du mot.
inline void pack(const std::vector<__half>& w, int ct, int taps, int cin, std::vector<std::uint32_t>& out) {
  const int kcs = cin / 16, nps = ct / 16;
  const auto bits = [&](int n, int tap, int c) -> std::uint32_t {
    const __half h = w[(static_cast<std::size_t>(n) * taps + tap) * cin + c];
    return __half_as_ushort(h);
  };
  for (int ks = 0; ks < taps * kcs; ++ks) {
    const int tap = ks / kcs, kc = ks % kcs;
    for (int np = 0; np < nps; ++np)
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, k0 = kc * 16 + 2 * t;
        for (int j = 0; j < 2; ++j) {
          const int n = (2 * np + j) * 8 + g;
          out.push_back(bits(n, tap, k0) | (bits(n, tap, k0 + 1) << 16U));
          out.push_back(bits(n, tap, k0 + 8) | (bits(n, tap, k0 + 9) << 16U));
        }
      }
  }
}

}  // namespace kirari::gpu::cuda::rife_native
