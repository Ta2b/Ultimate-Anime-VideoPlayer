#pragma once
// La planification commune des petits réseaux natifs de GMFSS (--gmfss-native, docs/gmfss-natif.md), faite
// par gmfss_conv_plan.cu : chaque convolution d'un réseau aux types de prepare/gmfss_grid.h planifiée une
// fois, ses poids rangés dans l'ordre des fragments B du mma, et les tenseurs placés dans une zone par durée
// de vie : deux qui vivent ensemble ne se recouvrent jamais (le résiduel n'est donc jamais la sortie), deux
// qui se suivent partagent la mémoire. Les noyaux sont ceux du RIFE natif (rife_native_pick.cu) ; chaque
// convolution y prend, pour chacune des trois sommes, sa forme (tranche, tuile, lignes par warp, K découpé,
// poids à l'avance), ou la forme en flux (conv_stream_kernels.cuh), réglée au banc du RIFE natif
// (rife_native_layer.h), avec son garde-fou NaN et ses tolérances. Les convolutions se rangent en morceaux,
// dont les sommes se choisissent à part (gmfss_mix.h) ; la tête : les convolutions qui écrivent une sortie du
// réseau. Pour la paire (gmfss_pair_native_net.cu) : des tenseurs externes, donnés au lancement ; une image
// en trois plans lue par In::kImage en pas 2 ; la zone donnée au lancement, que deux réseaux partagent. Lu
// par nvcc seulement (C++20).
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/gmfss_native_parts.h"
#include "gpu/cuda/backend/rife_native_layer.h"
#include "prepare/gmfss_grid.h"

namespace kirari::gpu::cuda::gmfss_plan {

namespace rn = rife_native;

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
}
inline unsigned blocks_for(long count) {
  return static_cast<unsigned>((count + 255) / 256);
}

// Les sommes réglées pour chaque convolution, dans l'ordre de Layer::shapes (celui de RifeNativeSums).
inline constexpr std::array<RifeNativeSums, 3> kSums{RifeNativeSums::kF16, RifeNativeSums::kMixed,
                                                     RifeNativeSums::kF32};
constexpr std::size_t sums_index(RifeNativeSums sums) {
  return static_cast<std::size_t>(sums);
}

// La forme d'une convolution pour des sommes : la plus rapide au banc (ms, son temps, sur une bande de ses
// lignes pour une grande couche), ou la seule qu'un noyau fasse (ms nul : pas mesurée).
struct Shape {
  rn::Config cfg;
  rn::Launch launch = nullptr;
  dim3 grid;
  int smem = 0;
  float ms = 0.F;
  int rejected = 0;    // formes écartées au banc : sortie différente de celle de la forme de référence
  std::string stream;  // comparaison au banc de la forme en flux ou à halo partagé, si essayée
};

// Une convolution : kImage (l'image en trois plans, pas 2), kSame, kHalf, kPhase (transposée par phases) ou
// kShuffle (3 × 3 et PixelShuffle(2) : la phase (py, px) écrite au pixel (2y + py, 2x + px)).
struct Layer {
  rn::Kind kind = rn::Kind::kSame;
  int cout = 16, phases = 1;  // cout : canaux de sortie d'une phase, complétés à 16
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int in_ld = 0, out_ld = 0, cin = 0, skip_ld = 0;
  int from = 0, to = 0, skip = -1;
  float in_slope = 1.F;
  bool activation = false;
  bool blocked8 = false;   // entrée [groupes C8][pixels][8]
  bool ring_next = false;  // l’étape suivante est le consommateur unique de cet anneau
  bool join_next = false;  // les quatre convolutions des deux branches se terminent par leur addition
  bool rgb = false;        // dernière convolution 64 → 3 du GridNet, sans résiduel
  bool head = false;       // écrit une sortie du réseau (Io::kept)
  int piece = 0;
  RifeNativeSums sums = RifeNativeSums::kF32;  // les sommes lancées
  std::array<Shape, kSums.size()> shapes;
  std::array<rn::Buffer, 3> packs;  // poids rangés pour les tuiles de 16, 32 et 64 sorties
  rn::Buffer bias, slope;
  const Shape& shape() const { return shapes[sums_index(sums)]; }
};

struct Placed {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::int64_t offset = 0;
  std::int64_t bytes = 0;
  int first = 0;      // première écriture (-1 : écrit avant la première convolution)
  int last = 0;       // dernière lecture (le nombre de convolutions : lu après la dernière)
  int external = -1;  // rang parmi les tenseurs donnés au lancement ; -1 : dans la zone
};

// Ce que le réseau prend et rend hors de ses convolutions. external : les tenseurs dont la mémoire est donnée
// au lancement (dans l'ordre de bound) ; image : l'un d'eux, trois plans FP16 (-1 : aucun) ; written : ceux
// de la zone qu'on remplit avant la première convolution ; kept : ceux qu'on lit après la dernière, les
// sorties du réseau.
struct Io {
  std::vector<int> external;
  int image = -1;
  std::vector<int> written;
  std::vector<int> kept;
  int rgb = -1;              // image finale facultative, dans kept
  bool features_c8 = false;  // GridNet : les trois entrées de caractéristiques
};

// Le banc du réglage et sa voie, où partent aussi les poids ; tuning : le compte, que le plan complète.
struct Tuner {
  rn::Bench& bench;
  cudaStream_t stream = nullptr;
  GmfssTuning& tuning;
};

// Les lignes lancées : toutes celles de la couche, ou au banc une bande de ses premières lignes de sortie, et
// ih, les lignes d'entrée qu'elles lisent (au-delà, le bord de zéros d'une image plus basse).
struct Rows {
  int ih = 0;
  int oh = 0;
};

// Ce que le plan (gmfss_conv_plan.cu) et son réglage au banc (gmfss_conv_tune.cu) partagent : une sortie
// par phases (transposée, PixelShuffle), le paquet de poids d'une tuile, les sommes partielles d'une forme à
// K découpé, les paramètres d'un lancement sur rows de from vers to (skip : le résiduel, nul sans ; poids w
// rangés pour la tuile de la forme), et le lancement (avec la fin d'une forme à K découpé).
bool phased(rn::Kind kind);
std::size_t pack_index(int ct);
std::int64_t partial_of(const Layer& layer, const rn::Config& cfg, int oh);
rn::Params params_of(const Layer& layer, Rows rows, const Shape& shape, const __half* from, __half* to,
                     const __half* skip, float* partial, const void* w);
void enqueue(const Layer& layer, const Shape& shape, cudaStream_t stream, const rn::Params& p);

// Un réseau planifié pour une taille : height × width au niveau 0 (kGmfssFullLevel au double).
class ConvNet {
 public:
  ConvNet(int height, int width);

  // Les tenseurs placés, puis chaque convolution planifiée et réglée au banc pour chacune des sommes de
  // kSums. pieces : le morceau de chaque convolution (vide : un seul). Faux et error rempli si une
  // convolution sort de ce que les noyaux font, ou sur une erreur CUDA (envoi des poids, banc).
  bool plan(const prepare::GmfssGrid& grid, const Io& io, std::span<const int> pieces, const Tuner& tuner,
            std::string& error);

  // Octets de la zone des tenseurs, et des sommes partielles de la plus grande forme réglée à K découpé.
  std::int64_t zone_bytes() const { return zone_bytes_; }
  std::int64_t partial_bytes() const { return partial_bytes_; }
  const Placed& tensor(int id) const { return tensors_[static_cast<std::size_t>(id)]; }
  // Un tenseur de la zone qui commence à zone.
  static __half* at(void* zone, const Placed& t) {
    return reinterpret_cast<__half*>(static_cast<std::uint8_t*>(zone) + t.offset);
  }

  bool has_head(int piece) const;
  bool planar_head_ready() const;
  bool join_ready() const;
  // Au chargement seulement, stream vidé entre les deux placements.
  void use_join(bool enabled);
  // Les sommes de chaque convolution : celles de son morceau (precisions[morceau]), la tête à part.
  void use(std::span<const RifeNativePrecision> precisions);
  // Les poids des tuiles que les sommes posées n'emploient pas : libérés.
  void trim();

  // Les convolutions sur stream ; bound : la mémoire des tenseurs externes, dans l'ordre d'Io::external ;
  // piece positif : celles de ce morceau seulement, sur ce que la zone garde (pour son temps).
  // Avec rgb, essaie d'écrire directement en trois plans. Renvoie vrai si ces plans ont été écrits.
  // planar_head désigne les trois images de l'entrée x0 quand la fusion de sa lecture est validée.
  bool run(cudaStream_t stream, void* zone, float* partial, std::span<const void* const> bound,
           int piece = -1, __half* rgb = nullptr,
           const std::array<const __half*, 3>* planar_head = nullptr) const;

 private:
  void lifetimes(const prepare::GmfssGrid& grid, const Io& io);
  void place(const prepare::GmfssGrid& grid, const Io& io);
  void pack();
  bool layout(const prepare::GmfssGridStep& step, const Io& io, Layer& layer, std::string& error) const;
  bool build(const prepare::GmfssGridStep& step, const Tuner& tuner, Layer& layer, std::string& error);
  bool tune(const Tuner& tuner, std::size_t s, Layer& layer, std::string& error);
  const Layer* twin(const Layer& layer) const;
  bool joins_at(std::size_t index) const;
  __half* address(int id, void* zone, std::span<const void* const> bound) const;

  int height_;
  int width_;
  int sms_ = 1;
  std::vector<Placed> tensors_;
  std::vector<Placed> join_tensors_;
  bool join_on_ = false;
  int planar_head_ = -1;
  std::vector<Layer> layers_;
  std::int64_t zone_bytes_ = 0;
  std::int64_t partial_bytes_ = 0;
};

}  // namespace kirari::gpu::cuda::gmfss_plan
