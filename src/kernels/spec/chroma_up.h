// La chroma 4:2:0 portée à la grille de luma par Spline36, pour une image de dimensions paires (rapport 2
// exact) : la même table que compute_weights(n / 2, n, shift, Spline36) (ref/resample.cpp), réduite à ses
// deux phases. La sortie i = 2m + parité a pour centre m + parité / 2 + 0,25 + shift dans la grille de la
// chroma : seuls deux jeux de six poids existent par axe, un par parité.
//
// Partagé entre l'hôte et l'appareil (audits A03-01) : l'hôte calcule les deux phases en double
// (chroma_up_phase) et les passe au noyau, qui n'applique que chroma_up_at ; le test sur processeur compare
// l'ensemble à ref::resample_chroma, le test GPU compare le noyau à cette même référence. Application :
// somme float de j = 0 à 5, depuis 0, bords en miroir (mirror_index de resample.h, recopié ici pour
// l'appareil) ; passe horizontale sur les codes bruts, puis verticale, intermédiaire float.
#pragma once

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep
#include "kernels/spec/resample.h"

// nvcc n'appelle depuis l'appareil qu'une fonction marquée pour lui. chroma_up_at n'y est que pour
// l'appareil : son fetch est une lambda de noyau, que nvcc refuse d'appeler depuis une fonction d'hôte.
#if defined(__CUDACC__)
#define AJP_SPEC_HOST_DEVICE __host__ __device__
#define AJP_SPEC_DEVICE __device__
#else
#define AJP_SPEC_HOST_DEVICE
#define AJP_SPEC_DEVICE
#endif

namespace ajp::kernels {

inline constexpr int kChromaUpTaps = 6;

// Les prises d'une parité : la première est l'échantillon m + offset de la chroma, m = i / 2.
struct ChromaUpPhase {
  int offset = 0;
  float w[kChromaUpTaps] = {};
};

// La phase de la parité (0 ou 1) pour le décalage de chroma shift (chroma_shift_up). Hôte seulement.
constexpr ChromaUpPhase chroma_up_phase(int parity, double shift) {
  const double pos = 0.5 * parity + 0.25 + shift;  // le centre de la sortie, pour m = 0
  const double low = pos - kChromaUpTaps / 2.0 + 0.5;
  int start = static_cast<int>(low);
  if (static_cast<double>(start) > low) --start;  // floor
  const double begin = start + 0.5;
  double row[kChromaUpTaps] = {};
  double sum = 0.0;
  for (int j = 0; j < kChromaUpTaps; ++j) {
    row[j] = spline36(begin + j - pos);
    sum += row[j];
  }
  ChromaUpPhase phase;
  phase.offset = start;
  for (int j = 0; j < kChromaUpTaps; ++j) phase.w[j] = static_cast<float>(row[j] / sum);
  return phase;
}

// mirror_index (resample.h), pour l'hôte et l'appareil : −1 → 0, n → n − 1, puis borné.
AJP_SPEC_HOST_DEVICE constexpr int chroma_up_mirror(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  if (i < 0) return 0;
  return i >= n ? n - 1 : i;
}

// La sortie i (≥ 0) d'un axe de n échantillons de chroma : fetch(k) rend l'échantillon k, phase est celle de
// la parité de i.
template <typename Fetch>
AJP_SPEC_DEVICE constexpr float chroma_up_at(const ChromaUpPhase& phase, int i, int n, const Fetch& fetch) {
  const int start = i / 2 + phase.offset;
  float a = 0.0F;
  for (int j = 0; j < kChromaUpTaps; ++j) a += phase.w[j] * fetch(chroma_up_mirror(start + j, n));
  return a;
}

}  // namespace ajp::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
