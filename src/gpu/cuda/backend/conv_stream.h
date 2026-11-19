#pragma once
// Le calendrier de la forme en flux des convolutions natives (conv_stream_kernels.cuh), sans CUDA : le noyau
// et ses tests (tests/unit/gpu/conv_stream_test.cpp) prennent les mêmes fonctions.
//
// Un bloc descend une colonne de 32 pixels de large, bande après bande (4 · rows lignes de sortie chacune),
// toute l'entrée (une seule tranche de canaux) dans un anneau de lignes en mémoire partagée. La fenêtre d'une
// bande, th + 2 lignes d'entrée (le bord compris), y occupe th + 2 emplacements ; les th autres reçoivent,
// par cp.async pendant les mma de la bande, les th nouvelles lignes de la suivante. Les deux lignes du bord
// sont déjà là : aucune ligne n'est lue deux fois. Quand ses mma sont finis, les th premières lignes de la
// fenêtre ne servent plus : la sortie de la bande s'y range (biais, arrondi FP16), puis part en mémoire
// globale, avant que le chargement de la bande d'après ne les reprenne.
#include <algorithm>
#include <vector>

namespace kirari::gpu::cuda::conv_stream {

inline constexpr int kWarps = 4;       // warps d'un bloc, une ligne de sortie chacun par rangée
inline constexpr int kTW = 32;         // colonnes de sortie d'un bloc
inline constexpr int kCols = kTW + 2;  // colonnes d'une ligne de l'anneau, le bord compris

// Lignes de sortie d'une bande pour rows lignes par warp ; lignes de l'anneau : la fenêtre (th + 2) et les
// nouvelles lignes de la bande suivante (th).
constexpr int band_rows(int rows) {
  return kWarps * rows;
}
constexpr int ring_rows(int th) {
  return 2 * th + 2;
}

// L'emplacement de la ligne row de la fenêtre de la bande band (row 0 : la ligne au-dessus de la bande, row
// th + 1 : celle au-dessous) ; la ligne row de la bande band + 1 est la ligne row + th de la bande band.
constexpr int window_slot(int band, int row, int th) {
  return (band * th + row) % ring_rows(th);
}

// Octets de mémoire partagée de l'anneau, pour une tranche de kc canaux au pas kc + 8 (celui des noyaux du
// RIFE natif : ldmatrix sans conflit de banques).
constexpr int smem_bytes(int kc, int rows) {
  return ring_rows(band_rows(rows)) * kCols * (kc + 8) * 2;
}
// La sortie d'une ligne (kTW pixels de ct canaux, au pas ct + 8) tient dans un emplacement.
constexpr bool stages_in_slot(int kc, int ct) {
  return kTW * (ct + 8) <= kCols * (kc + 8);
}

// Les bandes [first, first + count) de la colonne de blocs by, quand strips colonnes se partagent tiles
// bandes : count est nul pour une colonne de trop.
struct Strip {
  int first = 0;
  int count = 0;
};
constexpr Strip strip_of(int tiles, int strips, int by) {
  const int per = (tiles + strips - 1) / strips;
  const int first = by * per;
  return {.first = first, .count = std::max(0, std::min(per, tiles - first))};
}
// Les colonnes de blocs (gridDim.y) pour bands bandes au plus par bloc.
constexpr int strips_for(int tiles, int bands) {
  return (tiles + bands - 1) / bands;
}

// Les bandes par bloc à essayer au réglage : une, deux ou quatre vagues de deux blocs par multiprocesseur
// (la mémoire partagée de l'anneau en laisse deux), sur une grille de tiles_x × tiles_y bandes et z tuiles de
// sortie ; deux bandes au moins (une seule ne recouvre rien).
inline std::vector<int> band_counts(int tiles_x, int tiles_y, int z, int sms) {
  std::vector<int> out;
  const long columns = static_cast<long>(tiles_x) * z;
  for (const int waves : {1, 2, 4}) {
    const long strips = std::max(1L, (2L * waves * sms + columns - 1) / columns);
    const int bands = std::max(2, static_cast<int>((tiles_y + strips - 1) / strips));
    if (bands <= tiles_y && std::ranges::find(out, bands) == out.end()) out.push_back(bands);
  }
  return out;
}

}  // namespace kirari::gpu::cuda::conv_stream
