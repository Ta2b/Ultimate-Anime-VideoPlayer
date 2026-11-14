// Le SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md), vu de l'extérieur : aucun en-tête CUTLASS
// ici, ils restent dans span_fp8_net.cu. Dégroupage, conv_1 (FP16), huit blocs SPAB en FP8 E4M3,
// concaténation des quatre tenseurs, conv_cat, to_pixel (FP16, DepthToSpace fondu à son épilogue), lancés
// l'un après l'autre sur le flux donné. Entrée 1×3×H×W et sortie 1×3×2H×2W en FP16 NCHW, comme le SPAN en
// ONNX.
//
// Cette interface n'emploie ni std::expected ni <format> : nvcc (EDG) ne les lit pas, et le .cu qui
// l'implémente est compilé par lui. span_fp8_engine.cpp la convertit en Refusal.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/ring_surface.h"

namespace ajp::gpu::cuda {

// Une convolution vue sans copie : poids FP16 (motifs binaires) rangés [sortie][ligne][colonne][entrée],
// biais f32.
struct SpanFp8ConvView {
  const std::uint16_t* weights = nullptr;
  const float* bias = nullptr;
};

// Comment calculer un modèle à 48 canaux (kKernelsVersion 4, chantier SPAN 1). Native : noyaux de 48
// colonnes, K des convolutions 3×3 mis à plat (9 × 48 = 432, en tuiles de 64 sans queue par position) ;
// PerTap : 48 colonnes, K par position du filtre (le quart des tuiles K reste masqué) ; Padded : l'ancien
// calcul en 64 canaux dont 16 nuls. Les deux derniers ne servent qu'à la mesure (ajp_span_fp8_run).
enum class SpanFp8Width : std::uint8_t { Native, PerTap, Padded };

// Le contenu d'un gpu/span_fp8/package.h, sans ses vecteurs. Les pointeurs ne servent que pendant create().
struct SpanFp8Weights {
  static constexpr int kBlocks = 8;
  static constexpr int kScales = 1 + 3 * kBlocks;
  int blocks = kBlocks;           // 2 à kBlocks (étape 3.2c : le b5f48 en a 5)
  const float* scales = nullptr;  // 1 + 3·blocks échelles E4M3 d'activation (kScales places)
  SpanFp8ConvView first;
  SpanFp8ConvView conv[kBlocks][3];
  SpanFp8ConvView attention[kBlocks];
  SpanFp8ConvView concat;
  SpanFp8ConvView pixel;
  bool acc16 =
      false;  // conv_cat, to_pixel (étape 3.5a) et les blocs FP8 (kKernelsVersion 3) accumulent en f16
  // Canaux réels du modèle (span_fp8::Package::real_channels) : 64, ou 48 pour un b5f48, dont le paquet
  // range les poids en 64 avec des zéros. Les poids ci-dessus sont toujours ceux du paquet, rangés en 64.
  int channels = 64;
  SpanFp8Width width48 = SpanFp8Width::Native;
  // Les blocs en FP16, sans quantification : le SPAN FP16 en noyaux CUTLASS (lot 2 du fil SPAN), à juger
  // contre TensorRT FP16. Les échelles sont alors ignorées ; acc16 garde son sens (accumulateur f16).
  bool fp16 = false;
};

// Un emplacement RGBA16 UNORM de l'anneau où to_pixel écrit aussi (étape 3.4a) : surface 0, pas d'anneau.
// Seuls les pixels de 0 à width-1 et de 0 à height-1 sont écrits.
struct SpanFp8Ring {
  RingSurface surface{};  // emplacement de l'anneau (ring_surface.h) ; object 0 : aucun
  int width = 0;
  int height = 0;
};

enum class SpanFp8Status : std::uint8_t { Ok, OutOfMemory, Failed };

// Ce que SpanFp8Net::poll_accumulator a changé.
enum class SpanFp8AccChange : std::uint8_t { None, ToF32, ToF16 };

class SpanFp8Net {
 public:
  // Vrai pour sm 8.9 (Ada) seulement : les noyaux emploient les mma FP8 de cette architecture. CMake les
  // compile aussi pour sm 12.0 (CUDA_ARCHITECTURES « 89-real;120 »), mais ils n'y sont pas pris : le FP8
  // reste éteint sur RTX 50 tant qu'aucune mesure ne l'y a jugé.
  static bool supported_device(int major, int minor);
  // Le SPAN FP16 natif (SpanFp8Weights::fp16) : sm 8.9 (RTX 40) et 12.0 (RTX 50), les architectures de la
  // bibliothèque des noyaux. Le chargement les juge sur la carte contre le calcul du processeur
  // (span_fp8_check.h) : en 12.0, personne ne les a vus tourner.
  static bool supported_fp16_device(int major, int minor);
  // La version des noyaux, écrite au reçu (étape 3.7b) : à augmenter à chaque changement de ce qu'ils
  // calculent (un paquet validé avec d'autres noyaux se revalide). 1 : état après les étapes 3.1 à 3.5 ;
  // 2 : poids FP8 des blocs à une échelle par canal de sortie (étape 10.3) ; 3 : la variante acc16 accumule
  // aussi les blocs FP8 en f16, poids rangés 16 fois plus bas (deuxième passe TensorRT, N3) ; 4 : un modèle à
  // 48 canaux se calcule en 48 (SpanFp8Width::Native) et non plus en 64 dont 16 nuls ; 5 : les tuiles de 48
  // colonnes chargent toutes leurs lignes (Cfg::kTM) et to_pixel reprend sa tuile de 64 : un paquet jugé par
  // la version 4 l'a été avec un troisième plan faux ; 6 : l'accumulateur f16 borne une somme débordée et
  // passe au f32, la porte juge aussi le blanc plein et la mire saturée. Le retour au f16 après un temps
  // (poll_accumulator) ne change pas la version : chaque image se calcule comme avant.
  static constexpr int kKernelsVersion = 6;

  // Envoie les poids sur l'appareil courant (FP8 pour les blocs) et prépare les lancements. height et width
  // sont déjà jugés par span_fp8::size_refusal. Nul si l'envoi échoue ; status et error disent pourquoi.
  static std::unique_ptr<SpanFp8Net> create(const SpanFp8Weights& weights, int height, int width,
                                            SpanFp8Status& status, std::string& error);
  ~SpanFp8Net();
  SpanFp8Net(const SpanFp8Net&) = delete;
  SpanFp8Net& operator=(const SpanFp8Net&) = delete;

  // Octets de travail que launch() demande (tampons intermédiaires), à fournir 256 octets alignés.
  std::int64_t workspace_bytes() const;

  // Pose la zone de travail : les paramètres des lancements portent ses adresses. À refaire quand la zone
  // change ; sans appel aux fonctions CUDA de flux, donc sans effet sur une capture de graphe en cours.
  void bind(void* workspace);

  // Lance les noyaux sur stream. input : 3×H×W FP16, output : 3×2H×2W FP16 (adresses d'appareil). Faux si un
  // lancement échoue (cudaGetLastError), avec le motif dans error.
  bool launch(cudaStream_t stream, const void* input, void* output, std::string& error);
  // launch en deux parts (étape 3.3) : le corps (dégroupage, conv_1, blocs, conv_cat ; 3·blocs + 3
  // lancements) ne dépend que de l'entrée et de la zone, une capture de graphe le prend par adresse d'entrée
  // ; la queue (to_pixel, DepthToSpace fondu : 1 lancement) écrit la sortie, lancée en direct avec son
  // adresse. output peut être nul si ring porte une surface : l'image ne va alors qu'à l'anneau, au bit près
  // de ce que write_rgba16 écrirait de la sortie FP16.
  bool launch_body(cudaStream_t stream, const void* input, std::string& error);
  bool launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring, std::string& error);

  // Accumulateur f16 (SpanFp8Weights::acc16) : une image dont une somme déborde sort avec cette somme bornée
  // au plus grand f16 (rien d'infini ni de NaN). poll_accumulator, une fois avant chaque corps, la compte et
  // applique span_fp8::Acc16Policy : des débordements rapprochés passent le réseau au f32 pour un temps, puis
  // il revient au f16. Autre chose que None : les lancements ont changé, un graphe capturé avant est à
  // refaire. Toujours None sans acc16. launch l'appelle.
  SpanFp8AccChange poll_accumulator();
  // Les images débordées depuis la création (exact une fois le flux attendu) ; 0 sans acc16.
  int overflow_frames() const;

 private:
  struct Impl;
  explicit SpanFp8Net(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::gpu::cuda
